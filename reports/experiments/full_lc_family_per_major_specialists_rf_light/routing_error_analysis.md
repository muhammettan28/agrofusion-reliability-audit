# Routing Error Analysis

## Predicted-Major Routing Correctness

| Route Correct | N | Share | Family Accuracy | Mean Family Conf. | Mean Major Conf. |
|---|---:|---:|---:|---:|---:|
| False | 5212 | 0.1643 | 0.0000 | 0.6260 | 0.6552 |
| True | 26513 | 0.8357 | 0.8345 | 0.7186 | 0.8028 |

## Major Routing Confusion

| Target Major | Routed Major | N | Family Accuracy |
|---|---|---:|---:|
| B | B | 16582 | 0.7994 |
| B | E | 1915 | 0.0000 |
| B | F | 99 | 0.0000 |
| E | E | 9638 | 0.8897 |
| E | B | 2175 | 0.0000 |
| E | F | 61 | 0.0000 |
| F | B | 920 | 0.0000 |
| F | F | 293 | 1.0000 |
| F | E | 42 | 0.0000 |
