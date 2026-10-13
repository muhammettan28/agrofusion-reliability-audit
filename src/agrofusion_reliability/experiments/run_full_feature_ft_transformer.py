"""Validation-selected, full-feature FT-Transformer sensitivity experiment.

Uses the authors' rtdl-revisiting-models implementation with Linformer compression.
Historical predictions are never overwritten. Test labels do not select settings.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import polars as pl
import rtdl_revisiting_models as rtdl
import torch
from scipy.special import softmax
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.preprocessing import QuantileTransformer


def predict(model, x, batch_size):
    model.eval()
    with torch.no_grad():
        return np.concatenate(
            [
                model(x[i : i + batch_size], None).cpu().numpy()
                for i in range(0, len(x), batch_size)
            ]
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=Path(
            "data/study/features_full_lc_family/pilot_fusion.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/experiments/full_feature_ft_transformer"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-3, 3e-4])
    parser.add_argument(
        "--trial-index",
        type=int,
        help="Train one prespecified trial in an isolated directory",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Write validation results without evaluating test labels",
    )
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; no silent device fallback")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.trial_index is not None:
        if not 0 <= args.trial_index < len(args.learning_rates):
            parser.error("--trial-index must index --learning-rates")
        if not args.train_only:
            parser.error(
                "Isolated trials require --train-only to avoid premature test evaluation"
            )
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pl.read_parquet(args.feature_path)
    columns = [c for c in frame.columns if c.startswith("f_")]
    assert len(columns) == 568
    split = frame["class_balanced_split"].to_numpy()
    y = frame["target_label"].to_numpy().astype(str)
    train, val, test = [split == s for s in ["train", "validation", "test"]]
    # Coarse B/E annotations remain in training; downstream evaluation uses fine labels.
    val &= np.char.str_len(y) == 2
    test &= np.char.str_len(y) == 2
    classes = np.unique(y[train])
    encoded = np.searchsorted(classes, y)
    assert np.all(classes[encoded[val | test]] == y[val | test])
    x = frame.select(columns).to_numpy().astype(np.float32)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputer.fit(x[train])
    x = imputer.transform(x)
    scaler = QuantileTransformer(
        n_quantiles=1000,
        output_distribution="normal",
        subsample=100000,
        random_state=42,
    )
    scaler.fit(x[train])
    x = scaler.transform(x).astype(np.float32)
    xt, xv, xe = [
        torch.from_numpy(x[mask]).to(args.device) for mask in [train, val, test]
    ]
    yt = torch.from_numpy(encoded[train]).long().to(args.device)
    config = dict(
        n_cont_features=len(columns),
        cat_cardinalities=[],
        d_out=len(classes),
        n_blocks=2,
        d_block=48,
        attention_n_heads=4,
        attention_dropout=0.2,
        ffn_d_hidden=96,
        ffn_d_hidden_multiplier=None,
        ffn_dropout=0.1,
        residual_dropout=0.0,
        linformer_kv_compression_ratio=0.1,
        linformer_kv_compression_sharing="headwise",
    )
    manifest = dict(
        feature_sha256=hashlib.sha256(args.feature_path.read_bytes()).hexdigest(),
        feature_columns=columns,
        counts=dict(
            train=int(train.sum()), validation=int(val.sum()), test=int(test.sum())
        ),
        seed=42,
        configuration=config,
        learning_rates=args.learning_rates,
        max_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        threads=args.threads,
        device=args.device,
        cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        matrix_precision="float32; TF32 disabled; deterministic algorithms enabled",
        preprocessing="training-only median imputation and Gaussian quantile transformation; all 568 features retained",
        selection="minimum fine-target validation NLL, then fixed learning-rate order; no test selection",
        scope="one normal partition, one seed, two prespecified learning rates, compact compressed architecture; not an exhaustive tuned neural-model comparison",
        versions={
            m: importlib.metadata.version(m)
            for m in [
                "torch",
                "numpy",
                "scikit-learn",
                "polars",
                "rtdl-revisiting-models",
            ]
        },
        python=platform.python_version(),
    )
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        assert all(previous.get(k) == v for k, v in manifest.items()), (
            "Existing run differs; use a fresh output directory"
        )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    trials, history = [], []
    for trial, lr in enumerate(args.learning_rates):
        if args.trial_index is not None and trial != args.trial_index:
            continue
        np.random.seed(42)
        torch.manual_seed(42)
        model = rtdl.FTTransformer(**config).to(args.device)
        optimizer = torch.optim.AdamW(
            model.make_parameter_groups(), lr=lr, weight_decay=1e-5
        )
        checkpoint = args.output_dir / f"trial_{trial}.pt"
        history_path = args.output_dir / f"trial_{trial}_history.csv"
        complete_path = args.output_dir / f"trial_{trial}_complete.json"
        # A best-so-far checkpoint is not evidence that a training trial finished.
        # Interrupted trials restart with the same seed rather than silently becoming
        # candidates after fewer epochs than their prespecified stopping rule.
        if complete_path.exists() and checkpoint.exists() and history_path.exists():
            saved = torch.load(checkpoint, weights_only=False)
            trials.append(
                dict(
                    trial=trial,
                    learning_rate=lr,
                    best_epoch=saved["epoch"],
                    validation_nll=saved["validation_nll"],
                )
            )
            continue
        best, stale = np.inf, 0
        start = time.monotonic()
        trial_history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(len(xt), device=args.device)
            total = 0.0
            for i in range(0, len(order), args.batch_size):
                ids = order[i : i + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(xt[ids], None), yt[ids])
                loss.backward()
                optimizer.step()
                total += loss.item() * len(ids)
            logits = predict(model, xv, args.batch_size)
            nll = log_loss(
                encoded[val], softmax(logits, axis=1), labels=np.arange(len(classes))
            )
            acc = accuracy_score(encoded[val], logits.argmax(axis=1))
            trial_history.append(
                dict(
                    epoch=epoch,
                    training_nll=total / len(xt),
                    validation_nll=nll,
                    validation_accuracy=acc,
                    elapsed_seconds=time.monotonic() - start,
                )
            )
            print(
                f"FT trial={trial} lr={lr} epoch={epoch} val_NLL={nll:.5f} val_acc={acc:.5f} elapsed={time.monotonic() - start:.0f}s",
                flush=True,
            )
            if nll < best - 1e-6:
                best, stale = nll, 0
                torch.save(
                    dict(state=model.state_dict(), epoch=epoch, validation_nll=nll),
                    checkpoint,
                )
            else:
                stale += 1
            pl.DataFrame(trial_history).write_csv(history_path)
            if stale >= args.patience:
                break
        saved = torch.load(checkpoint, weights_only=False)
        complete_path.write_text(
            json.dumps(
                dict(
                    epochs_completed=epoch,
                    stopping_reason=(
                        "validation_patience"
                        if stale >= args.patience
                        else "epoch_limit"
                    ),
                    best_epoch=saved["epoch"],
                    validation_nll=saved["validation_nll"],
                ),
                indent=2,
            )
            + "\n"
        )
        trials.append(
            dict(
                trial=trial,
                learning_rate=lr,
                best_epoch=saved["epoch"],
                validation_nll=saved["validation_nll"],
            )
        )
        history.extend(trial_history)
    pl.DataFrame(trials).write_csv(args.output_dir / "validation_trials.csv")
    if args.train_only:
        print(
            "Completed validation-only training; test labels were not evaluated",
            flush=True,
        )
        return
    chosen = min(trials, key=lambda r: (r["validation_nll"], r["trial"]))
    model = rtdl.FTTransformer(**config).to(args.device)
    model.load_state_dict(
        torch.load(args.output_dir / f"trial_{chosen['trial']}.pt", weights_only=False)[
            "state"
        ]
    )
    logits = np.concatenate(
        [predict(model, xv, args.batch_size), predict(model, xe, args.batch_size)]
    )
    output_weight = model.backbone.output.linear.weight.detach().cpu().numpy()
    geometry = logits / np.maximum(np.linalg.norm(output_weight, axis=1), 1e-12)
    np.savez_compressed(
        args.output_dir / "posterior.npz",
        classes=classes,
        probabilities=softmax(logits, axis=1),
        logits=logits,
        geometric_distances=geometry,
    )
    records = pl.concat(
        [
            frame.filter(pl.Series(mask))
            .select("sample_id", "year", "target_label")
            .with_columns(pl.lit(name).alias("split"))
            for name, mask in [("validation", val), ("test", test)]
        ]
    )
    records.write_parquet(args.output_dir / "records.parquet")
    prob = softmax(logits[len(xv) :], axis=1)
    prediction = classes[prob.argmax(axis=1)]
    confidence = prob.max(axis=1)
    bins = np.minimum((confidence * 10).astype(int), 9)
    correct = prediction == y[test]
    ece = sum(
        abs(confidence[bins == b].mean() - correct[bins == b].mean())
        * (bins == b).mean()
        for b in range(10)
        if (bins == b).any()
    )
    metrics = dict(
        model="FT-Transformer compressed",
        accuracy=accuracy_score(y[test], prediction),
        balanced_accuracy=balanced_accuracy_score(y[test], prediction),
        macro_f1=f1_score(
            y[test],
            prediction,
            average="macro",
            labels=np.unique(y[test]),
            zero_division=0,
        ),
        ece_10=float(ece),
        nll=log_loss(encoded[test], prob, labels=np.arange(len(classes))),
    )
    pl.DataFrame([metrics]).write_csv(args.output_dir / "test_summary.csv")
    manifest.update(
        chosen_trial=chosen,
        posterior_sha256=hashlib.sha256(
            (args.output_dir / "posterior.npz").read_bytes()
        ).hexdigest(),
        records_sha256=hashlib.sha256(
            (args.output_dir / "records.parquet").read_bytes()
        ).hexdigest(),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
