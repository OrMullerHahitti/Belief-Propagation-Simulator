# exp2: the two ingredients of the split (50 seeds, 2000 iterations)

## random dense

| update rule | damping | frozen within 2000 | median freeze | median t(commit >= 95%) | final commitment | final cost | best cost | period 1 / 2 / other |
|---|---|---|---|---|---|---|---|---|
| MS (cavity) | 0.0 | 0% | nan | 2000 | 0.47 | 107902 +- 2815 | 103638 | 0 / 0 / 50 |
| doubling only (Q = 2 cavity) | 0.0 | 0% | nan | 2000 | 0.71 | 108219 +- 2686 | 104054 | 0 / 0 / 50 |
| echo only (Q = belief) | 0.0 | 0% | nan | 52 | 0.92 | 106576 +- 2878 | 105240 | 0 / 50 / 0 |
| split = cavity + belief | 0.0 | 0% | nan | 20 | 0.96 | 107117 +- 3654 | 105436 | 0 / 50 / 0 |
| MS (cavity) | 0.9 | 80% | 630 | 2000 | 0.81 | 99575 +- 2709 | 99172 | 40 / 0 / 10 |
| doubling only (Q = 2 cavity) | 0.9 | 60% | 916 | 2000 | 0.87 | 100372 +- 3556 | 99197 | 30 / 0 / 20 |
| echo only (Q = belief) | 0.9 | 100% | 208 | 2000 | 0.90 | 99817 +- 2445 | 99813 | 50 / 0 / 0 |
| split = cavity + belief | 0.9 | 100% | 62 | 81 | 0.95 | 99758 +- 2439 | 99758 | 50 / 0 / 0 |

## random sparse

| update rule | damping | frozen within 2000 | median freeze | median t(commit >= 95%) | final commitment | final cost | best cost | period 1 / 2 / other |
|---|---|---|---|---|---|---|---|---|
| MS (cavity) | 0.0 | 4% | 165 | 2000 | 0.07 | 17306 +- 1955 | 15638 | 2 / 0 / 48 |
| doubling only (Q = 2 cavity) | 0.0 | 0% | nan | 2000 | 0.33 | 17769 +- 1660 | 16254 | 0 / 0 / 50 |
| echo only (Q = belief) | 0.0 | 0% | nan | 2000 | 0.57 | 16487 +- 1478 | 16010 | 0 / 46 / 4 |
| split = cavity + belief | 0.0 | 0% | nan | 2000 | 0.77 | 17040 +- 1577 | 16557 | 0 / 50 / 0 |
| MS (cavity) | 0.9 | 38% | 599 | 2000 | 0.14 | 15055 +- 1567 | 14505 | 19 / 0 / 31 |
| doubling only (Q = 2 cavity) | 0.9 | 2% | 1227 | 2000 | 0.30 | 15998 +- 1612 | 14666 | 1 / 0 / 49 |
| echo only (Q = belief) | 0.9 | 100% | 224 | 2000 | 0.59 | 14704 +- 1411 | 14698 | 50 / 0 / 0 |
| split = cavity + belief | 0.9 | 98% | 71 | 2000 | 0.77 | 14633 +- 1417 | 14633 | 49 / 0 / 1 |

## graph coloring

| update rule | damping | frozen within 2000 | median freeze | median t(commit >= 95%) | final commitment | final cost | best cost | period 1 / 2 / other |
|---|---|---|---|---|---|---|---|---|
| MS (cavity) | 0.0 | 0% | nan | 2000 | 0.00 | 668 +- 162 | 238 | 0 / 43 / 7 |
| doubling only (Q = 2 cavity) | 0.0 | 0% | nan | 2000 | 0.00 | 779 +- 210 | 305 | 0 / 42 / 8 |
| echo only (Q = belief) | 0.0 | 0% | nan | 2000 | 0.52 | 455 +- 218 | 221 | 0 / 47 / 3 |
| split = cavity + belief | 0.0 | 2% | 19 | 2000 | 0.19 | 666 +- 259 | 284 | 1 / 48 / 1 |
| MS (cavity) | 0.9 | 16% | 380 | 2000 | 0.04 | 128 +- 88 | 33 | 8 / 0 / 42 |
| doubling only (Q = 2 cavity) | 0.9 | 22% | 412 | 2000 | 0.54 | 117 +- 90 | 30 | 10 / 0 / 40 |
| echo only (Q = belief) | 0.9 | 100% | 162 | 2000 | 0.73 | 37 +- 19 | 37 | 49 / 0 / 1 |
| split = cavity + belief | 0.9 | 80% | 130 | 2000 | 0.83 | 34 +- 18 | 33 | 40 / 0 / 10 |
