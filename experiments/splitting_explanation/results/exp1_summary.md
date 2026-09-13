# exp1: freeze time, commitment and cost (50 seeds, 2000 iterations)

## random dense

| algorithm | frozen within 2000 | median freeze | median t(commit >= 95%) | final commitment | final cost | best cost | period 1 / 2 / other |
|---|---|---|---|---|---|---|---|
| MS | 0% | nan | 2000 | 0.47 | 107902 +- 2815 | 103638 | 0 / 0 / 50 |
| DMS | 80% | 630 | 2000 | 0.81 | 99575 +- 2709 | 99172 | 40 / 0 / 10 |
| MS + split | 0% | nan | 20 | 0.96 | 107117 +- 3654 | 105436 | 0 / 50 / 0 |
| DMS + split | 100% | 62 | 81 | 0.95 | 99758 +- 2439 | 99758 | 50 / 0 / 0 |
| DMS($\lambda$=0.5) + split | 94% | 19 | 28 | 0.95 | 99659 +- 2413 | 99658 | 47 / 0 / 3 |

## random sparse

| algorithm | frozen within 2000 | median freeze | median t(commit >= 95%) | final commitment | final cost | best cost | period 1 / 2 / other |
|---|---|---|---|---|---|---|---|
| MS | 4% | 165 | 2000 | 0.07 | 17306 +- 1955 | 15638 | 2 / 0 / 48 |
| DMS | 38% | 599 | 2000 | 0.14 | 15055 +- 1567 | 14505 | 19 / 0 / 31 |
| MS + split | 0% | nan | 2000 | 0.77 | 17040 +- 1577 | 16557 | 0 / 50 / 0 |
| DMS + split | 98% | 71 | 2000 | 0.77 | 14633 +- 1417 | 14633 | 49 / 0 / 1 |
| DMS($\lambda$=0.5) + split | 96% | 21 | 2000 | 0.76 | 14632 +- 1405 | 14631 | 48 / 0 / 2 |

## graph coloring

| algorithm | frozen within 2000 | median freeze | median t(commit >= 95%) | final commitment | final cost | best cost | period 1 / 2 / other |
|---|---|---|---|---|---|---|---|
| MS | 0% | nan | 2000 | 0.00 | 668 +- 162 | 238 | 0 / 43 / 7 |
| DMS | 16% | 380 | 2000 | 0.04 | 128 +- 88 | 33 | 8 / 0 / 42 |
| MS + split | 2% | 19 | 2000 | 0.19 | 666 +- 259 | 284 | 1 / 48 / 1 |
| DMS + split | 80% | 130 | 2000 | 0.83 | 34 +- 18 | 33 | 40 / 0 / 10 |
| DMS($\lambda$=0.5) + split | 88% | 38 | 2000 | 0.85 | 32 +- 19 | 32 | 44 / 0 / 6 |

## scale free

| algorithm | frozen within 2000 | median freeze | median t(commit >= 95%) | final commitment | final cost | best cost | period 1 / 2 / other |
|---|---|---|---|---|---|---|---|
| MS | 2% | 72 | 2000 | 0.11 | 19594 +- 879 | 17617 | 1 / 0 / 49 |
| DMS | 44% | 360 | 2000 | 0.23 | 17171 +- 827 | 16591 | 22 / 0 / 28 |
| MS + split | 0% | nan | 2000 | 0.76 | 19292 +- 1101 | 18731 | 0 / 50 / 0 |
| DMS + split | 98% | 44 | 2000 | 0.76 | 16762 +- 312 | 16760 | 49 / 0 / 1 |
| DMS($\lambda$=0.5) + split | 98% | 14 | 2000 | 0.75 | 16772 +- 289 | 16771 | 49 / 0 / 1 |

## meeting scheduling

| algorithm | frozen within 2000 | median freeze | median t(commit >= 95%) | final commitment | final cost | best cost | period 1 / 2 / other |
|---|---|---|---|---|---|---|---|
| MS | 0% | nan | 2000 | 0.01 | 88 +- 7 | 42 | 0 / 50 / 0 |
| DMS | 6% | 447 | 2000 | 0.06 | 20 +- 8 | 9 | 3 / 0 / 47 |
| MS + split | 0% | nan | 2000 | 0.07 | 81 +- 15 | 45 | 0 / 50 / 0 |
| DMS + split | 86% | 107 | 2000 | 0.56 | 8 +- 2 | 8 | 43 / 0 / 7 |
| DMS($\lambda$=0.5) + split | 60% | 30 | 2000 | 0.61 | 27 +- 31 | 16 | 30 / 14 / 6 |
