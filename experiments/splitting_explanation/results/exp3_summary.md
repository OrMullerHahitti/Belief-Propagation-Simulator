# exp3: period-2 end states vs density (n=50, domain 10, 20 seeds, 1000 iterations)

## random instances

### MS + split

| density | edges | period 1 / 2 / other | flip fraction | class-2 edges | cost class 0 / 1 / 2 | 1-opt cost class 2 | layer cost | greedy from layer | improving single moves | BR steps | cost_2 increases | 
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 64 | 0 / 20 / 0 | 0.45 | 0.38 | 107 / 114 / 155 | 111 | 8074 | 7076 | 22.3 | nan | -1.0 |
| 0.1 | 121 | 0 / 20 / 0 | 0.68 | 0.55 | 113 / 119 / 154 | 120 | 16575 | 14414 | 34.2 | nan | -1.0 |
| 0.2 | 244 | 0 / 20 / 0 | 0.74 | 0.58 | 117 / 127 / 152 | 128 | 34318 | 30891 | 37.0 | nan | -1.0 |
| 0.3 | 366 | 0 / 19 / 1 | 0.78 | 0.62 | 122 / 131 / 152 | 132 | 52468 | 47730 | 38.6 | nan | -1.0 |
| 0.4 | 489 | 0 / 20 / 0 | 0.80 | 0.65 | 124 / 134 / 151 | 134 | 70693 | 64980 | 39.9 | nan | -1.0 |
| 0.6 | 732 | 0 / 20 / 0 | 0.76 | 0.61 | 127 / 137 / 151 | 137 | 106166 | 99315 | 38.2 | 1.000 | -0.8 |
| 0.8 | 978 | 0 / 20 / 0 | 0.83 | 0.70 | 127 / 137 / 151 | 138 | 143349 | 134615 | 41.4 | 1.000 | -0.9 |
| 1.0 | 1225 | 0 / 20 / 0 | 0.79 | 0.66 | 132 / 140 / 151 | 139 | 179319 | 169863 | 39.5 | 0.999 | -0.8 |

### DMS + split

| density | edges | period 1 / 2 / other | flip fraction | class-2 edges | cost class 0 / 1 / 2 | 1-opt cost class 2 | layer cost | greedy from layer | improving single moves | BR steps | cost_2 increases | 
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 64 | 19 / 0 / 1 | nan | nan | nan / nan / nan | nan | 6993 | 6993 | 0.0 | nan | -1.0 |
| 0.1 | 121 | 20 / 0 / 0 | nan | nan | nan / nan / nan | nan | 14232 | 14232 | 0.0 | nan | -1.0 |
| 0.2 | 244 | 20 / 0 / 0 | nan | nan | nan / nan / nan | nan | 30652 | 30652 | 0.0 | nan | -1.0 |
| 0.3 | 366 | 20 / 0 / 0 | nan | nan | nan / nan / nan | nan | 47546 | 47546 | 0.0 | nan | -1.0 |
| 0.4 | 489 | 20 / 0 / 0 | nan | nan | nan / nan / nan | nan | 64725 | 64725 | 0.0 | nan | -1.0 |
| 0.6 | 732 | 20 / 0 / 0 | nan | nan | nan / nan / nan | nan | 99150 | 99150 | 0.0 | nan | -1.0 |
| 0.8 | 978 | 20 / 0 / 0 | nan | nan | nan / nan / nan | nan | 134401 | 134401 | 0.0 | nan | -1.0 |
| 1.0 | 1225 | 20 / 0 / 0 | nan | nan | nan / nan / nan | nan | 169816 | 169816 | 0.0 | 1.000 | -0.8 |

## bipartite instances

### MS + split

| density | edges | period 1 / 2 / other | flip fraction | class-2 edges | cost class 0 / 1 / 2 | 1-opt cost class 2 | layer cost | greedy from layer | improving single moves | BR steps | cost_2 increases | rephased cost | rephased single moves |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 50 | 1 / 19 / 0 | 0.37 | 0.34 | 104 / 110 / 151 | 109 | 5989 | 5330 | 17.6 | nan | -1.0 | 5283 | 0.0 |
| 0.1 | 66 | 0 / 20 / 0 | 0.54 | 0.46 | 107 / 112 / 152 | 112 | 8489 | 7292 | 26.8 | nan | -1.0 | 7184 | 0.0 |
| 0.2 | 126 | 0 / 20 / 0 | 0.72 | 0.58 | 109 / 121 / 153 | 120 | 17372 | 14993 | 36.0 | nan | -1.0 | 14782 | 0.0 |
| 0.3 | 187 | 0 / 20 / 0 | 0.77 | 0.63 | 116 / 124 / 152 | 125 | 26297 | 23112 | 38.5 | nan | -1.0 | 22840 | 0.0 |
| 0.4 | 249 | 0 / 20 / 0 | 0.80 | 0.65 | 117 / 127 / 152 | 127 | 35692 | 31614 | 39.9 | nan | -1.0 | 31321 | 0.0 |
| 0.6 | 374 | 0 / 19 / 1 | 0.75 | 0.59 | 120 / 132 / 152 | 131 | 53481 | 48732 | 37.6 | nan | -1.0 | 48239 | 0.0 |
| 0.8 | 501 | 0 / 20 / 0 | 0.72 | 0.56 | 122 / 135 / 152 | 133 | 71665 | 66532 | 36.1 | nan | -1.0 | 66239 | 0.0 |
| 1.0 | 625 | 0 / 20 / 0 | 0.81 | 0.66 | 126 / 135 / 151 | 135 | 90814 | 84060 | 40.3 | nan | -1.0 | 83774 | 0.0 |

### DMS + split

| density | edges | period 1 / 2 / other | flip fraction | class-2 edges | cost class 0 / 1 / 2 | 1-opt cost class 2 | layer cost | greedy from layer | improving single moves | BR steps | cost_2 increases | rephased cost | rephased single moves |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 50 | 20 / 0 / 0 | nan | nan | nan / nan / nan | nan | 5282 | 5282 | 0.0 | nan | -1.0 | - | - |
| 0.1 | 66 | 20 / 0 / 0 | nan | nan | nan / nan / nan | nan | 7191 | 7191 | 0.0 | nan | -1.0 | - | - |
| 0.2 | 126 | 20 / 0 / 0 | nan | nan | nan / nan / nan | nan | 14853 | 14853 | 0.0 | nan | -1.0 | - | - |
| 0.3 | 187 | 20 / 0 / 0 | nan | nan | nan / nan / nan | nan | 22917 | 22917 | 0.0 | nan | -1.0 | - | - |
| 0.4 | 249 | 20 / 0 / 0 | nan | nan | nan / nan / nan | nan | 31432 | 31432 | 0.0 | nan | -1.0 | - | - |
| 0.6 | 374 | 20 / 0 / 0 | nan | nan | nan / nan / nan | nan | 48599 | 48599 | 0.0 | nan | -1.0 | - | - |
| 0.8 | 501 | 20 / 0 / 0 | nan | nan | nan / nan / nan | nan | 66371 | 66371 | 0.0 | nan | -1.0 | - | - |
| 1.0 | 625 | 20 / 0 / 0 | nan | nan | nan / nan / nan | nan | 83935 | 83935 | 0.0 | nan | -1.0 | - | - |
