# What changed after the compute_R fix, and what I propose for the paper

Summary for the meeting on 2026-09-16. Deadline: three weeks (AAAI-27).
The Sept 15 research (temporary split pulse, controllers, the 3-variable
path example, the damping theory) is summarized separately in
`results/professor_briefing_20260916.md`; this note covers the paper.

## In five sentences

1. A bug in how factors read their cost tables (fixed on Sept 9) affected the
   three random benchmark families. All min-sum lines on those families were
   rerun; graph coloring, meeting scheduling and DABP were never affected.
2. After the fix, the "merge" heuristic of Section 5 is no longer competitive:
   it loses to DMS-SCFG on all five benchmarks and to plain DMS on dense graphs.
   I propose keeping it only as a diagnostic that confirms the two-solution
   theory, and I rewrote Section 5 accordingly.
3. On the random families, damping alone now reaches the same final cost as
   splitting plus damping. What splitting adds there is speed: DMS-SCFG settles
   at a median iteration of 43 to 70, DMS at 359 to 720, and DMS does not
   settle at all in 10 to 30 of the 50 runs. I propose making this the stated
   claim for the random families, with a new settling table.
4. Delayed splitting improves on immediate splitting on every family in the
   mean, but with honest held-out selection of K the gain is significant only
   on random dense and scale-free graphs.
5. Two new split variants ran on all five benchmarks tonight: a fixed
   0.95/0.05 split (DABP's ratio, no network) and the temporary 0.95/0.05
   pulse. On the three random families the fixed 0.95 split is the best
   distributed line: it ties DABP on sparse and scale-free graphs (p = 0.19
   and 0.30) and trails it by 0.1 percent on dense graphs (p = 0.010). On the
   structured benchmarks the permanent 0.95 split hurts (meeting scheduling
   13.16 versus 8.10 for the symmetric split), while the pulse is the best
   line on graph coloring (27.23 versus 31.63 for the delayed split) and ties
   the symmetric split on meeting scheduling.

## 1. The bug, with one example

Take a binary factor between x3 and x7 whose table rows belong to x3. Before
the fix, if x7's message arrived first in the factor's inbox, the factor read
the table with x7 as the row index, as if the table were transposed. It then
sent messages computed from the wrong costs. The cost of the selected
assignment was always evaluated on the correct table, so the numbers in the
paper were real costs of assignments chosen from wrong messages.

Only factors whose inbox order differed from their variable order were hit.
Coloring and meeting-scheduling tables are symmetric, so a transposed read is
the same table, and DABP builds its own tensors. So: random dense, random
sparse and scale-free changed; graph coloring, meeting scheduling and all DABP
lines did not.

The rerun (Sept 9, 15 algorithms, 50 instances, 2000 iterations) replaced
every pre-fix row of those three families. The theory measurements
(commitment fractions, period census, damping grid) were produced by a
separate vectorized engine that always read the tables correctly; today I
checked that it reproduces the fixed engine exactly on a real 50-variable
sparse instance (zero difference over 2000 iterations, both MS-SCFG and
DMS-SCFG). Those sections need no change.

## 2. Table 1, before and after (mean final cost, 50 instances)

| Family | Line | Before | After |
|---|---|---:|---:|
| Random dense | MS | 108076.51 | 107935.73 |
| | DMS | 101904.84 | 99557.33 |
| | MS-SCFG | 107330.16 | 107196.39 |
| | DMS-SCFG | 101571.10 | 99758.48 |
| | best delayed split | 101335.62 (K=1000) | 99315.49 (K=1500) |
| | DABP (unchanged) | 99223.68 | 99223.68 |
| Random sparse | MS | 17611.47 | 17204.93 |
| | DMS | 15749.35 | 15100.29 |
| | MS-SCFG | 17288.43 | 17073.27 |
| | DMS-SCFG | 15357.87 | 14633.43 |
| | best delayed split | 15330.27 (K=1000) | 14554.99 (K=1000) |
| | DABP (unchanged) | 14519.01 | 14519.01 |
| Scale-free | MS | 19924.49 | 19630.55 |
| | DMS | 18229.40 | 17079.39 |
| | MS-SCFG | 19703.28 | 19291.73 |
| | DMS-SCFG | 17952.74 | 16761.71 |
| | best delayed split | 17900.34 (K=1000) | 16636.74 (K=1000) |
| | DABP (unchanged) | 16628.46 | 16628.46 |

Graph coloring and meeting scheduling are unchanged (DMS-SCFG 34.42 and 8.10,
delayed split 30.62 and 7.82, DABP 39.24 and 27.22).

Every damped line improved by about 2 to 4 percent. The merge lines improved
less, and the ordering flipped.

## 3. The four conclusions that change

### 3a. The merge is no longer a heuristic

| Family | MS-SCFG | MGM merge | exact merge | DMS-SCFG | DMS |
|---|---:|---:|---:|---:|---:|
| Graph coloring | 668.61 | 98.19 | 83.19 | 34.42 | 138.81 |
| Meeting scheduling | 81.80 | 20.98 | 20.14 | 8.10 | 20.70 |
| Random dense | 107196.39 | 100419.69 | 100419.21 | 99758.48 | 99557.33 |
| Random sparse | 17073.27 | 14800.23 | 14761.37 | 14633.43 | 15100.29 |
| Scale-free | 19291.73 | 16942.89 | 16878.09 | 16761.71 | 17079.39 |

DMS-SCFG beats the exact merge on every family (Wilcoxon p below 1e-4
everywhere). Plain DMS beats it on dense graphs (p = 1.7e-6), ties it on
meeting scheduling and scale-free graphs, and loses to it on graph coloring
(p = 3e-5) and sparse graphs (p = 0.028, not significant after Holm
correction). Before the fix the merges tied DMS-SCFG on dense and beat it on
sparse and scale-free graphs; that is gone.

What survives is the diagnostic value. The merges still close 82 to 95 percent
of the gap between raw MS-SCFG and DMS-SCFG while seeing only the two branch
assignments, and inverting every decision of the MGM merge is significantly
worse (p below 1e-9 on every family) yet still far better than the raw
snapshots. Both facts are what the two-solution theory predicts: the raw
snapshots mix two coherent solutions out of phase, and either coherent choice
beats the mixture.

What I changed: Section 5 no longer "proposes two versions of the algorithm".
It ends with a paragraph "Reading the two branches" that states the
diagnostic, its result, and its limit ("the menus are only as good as the two
branches; damping resolves the alternation inside the message passing, and
does better than any selection made after the fact"). The intro no longer
promises a search-based heuristic, and the abstract now says the split-damped
variants "match or significantly outperform" DABP on the structured
benchmarks (on graph coloring the difference to DABP is not significant,
p = 0.31).

### 3b. On random graphs, damping alone reaches the same cost; splitting makes it settle fast

Final cost, DMS versus DMS-SCFG: dense 99557 versus 99758 (DMS better,
Wilcoxon p = 1.3e-5, t-test p = 0.24); sparse 15100 versus 14633 (split
better, p = 6e-4); scale-free 17079 versus 16762 (not significant,
p = 0.25).

Settling, measured on the stored cost curves (runs of 50 whose cost is
constant over the last 100 iterations / median iteration of the last change):

| Line | Coloring | Meeting | Dense | Sparse | Scale-free |
|---|---:|---:|---:|---:|---:|
| DMS | 8 / 379 | 2 / 1386 | 40 / 720 | 20 / 606 | 22 / 359 |
| DMS-SCFG | 40 / 129 | 44 / 154 | 50 / 61 | 49 / 70 | 49 / 43 |
| DMS-SCFG @1000 | 39 / 1064 | 45 / 1082 | 50 / 990 | 50 / 1023 | 49 / 1006 |
| DABP | 30 / 110 | 14 / 1114 | 48 / 1087 | 46 / 31 | 47 / 1026 |
| DABP-NoSplit | 6 / 346 | 6 / 447 | 37 / 467 | 21 / 406 | 24 / 199 |

MS-SCFG settles in at most 6 of the 50 runs on any benchmark and in none on
the random families; plain MS settles in at most 2. This table is the quantitative form of the paper's central
claim ("splitting triggers rapid convergence of DMS"), and the paper did not
have it. I added it to Section 6 as Table "settling" with a short paragraph.

### 3c. Delayed splitting, with honest selection of K

The paper picked the best K per benchmark on the same 50 instances it
reports. The July review asked for held-out selection. Selecting K on seeds
0 to 24 and evaluating on seeds 25 to 49:

| Family | K selected | held-out gain over DMS-SCFG | Wilcoxon p |
|---|---:|---:|---:|
| Random dense | 1500 | 431 | 1.5e-6 |
| Scale-free | 500 | 104 | 0.002 |
| Random sparse | 100 | 14 | 0.34 |
| Graph coloring | 100 | 4.0 | 0.37 |
| Meeting scheduling | 100 | 0.16 | 0.25 |

So the delayed split improves on immediate splitting everywhere in the mean,
but its gain is established out of sample only on dense and scale-free
graphs; tonight's fixed 0.95 split overtakes it on the random families and
the pulse overtakes it on graph coloring (Section 6). Table 1 now reports the fixed choice K = 1000 for every
family (better than DMS-SCFG with p below 1e-5 on the three random families,
ties on the two structured ones), and a short paragraph reports the held-out
protocol. The old sentence "significant only on meeting scheduling and random
dense" was wrong after the fix and is replaced.

### 3d. DABP

Unchanged numbers, changed comparisons. DABP is best on random dense and
sparse graphs (against the delayed split at K = 1000: p = 4e-4 and p = 0.007),
tied with the delayed split on scale-free graphs (p = 0.75), tied with every
split line on graph coloring, and worse than every split line on meeting
scheduling (p below 1e-6). DABP-NoSplit is statistically indistinguishable
from DMS on all three random families (p = 0.67 to 0.86).

The new fixed 0.95/0.05 split (Section 6 below) is what DABP does to the graph
without the network. Against it, DABP's advantage shrinks to 0.1 percent on
dense graphs (p = 0.010, 32 wins to 18) and disappears on sparse (p = 0.19)
and scale-free graphs (p = 0.30). The honest statement for the paper becomes
"most of DABP's advantage on random graphs is the split ratio, which needs no
learning". That is a stronger paper than the current text, which concedes the
random families to DABP.

## 4. Section 5, what the new text says

Kept unchanged: the lemniscate figure, the example, Observation 5.1, and the
bridge sentence to the theorems. Removed: the proposal of the two merge
algorithms. Added: one paragraph, "Reading the two branches":

- the diagnostic (menus from iterations 198 and 199; MGM or exact selection),
- the two predictions and their outcome (selection far better than the raw
  oscillation; inverted selection significantly worse yet coherent, as
  Corollary bipartite leads one to expect),
- the limit (falls short of DMS-SCFG on every benchmark; used as evidence, not
  as a solver),
- the hand-off to Sections 6 and 7 (why exactly two solutions; why damping
  removes the alternation).

The two theory sections that follow (two-solution structure; damping) are
untouched. They now carry the section instead of the merge.

## 5. Section 6, what I changed

- Table 1 has the post-fix numbers, the fixed K = 1000 column, and two new
  columns for the fixed 0.95 split and the pulse (numbers fill in tonight).
  Ten numeric columns do not fit: in tonight's substitute two-column build the
  table overflows the text width by 139 pt (about two inches). Two columns
  have to go; my suggestion is MS-SCFG (it already appears in the merge table)
  and DABP-SymSplit (one sentence in the DABP paragraph). Both new lines earn
  their columns (Section 6).
- New settling table (Section 3b above) and a paragraph interpreting the
  random families as "same cost, ten times earlier".
- Statistical significance paragraph rewritten with the post-fix p-values.
  It promises that every cited p below 0.05 survives Holm correction within
  its benchmark except where marked; I will check that promise against the
  key-comparison script once tonight's run is in and mark the exceptions.
- New paragraph on held-out selection of K (Section 3c).
- The merge subsection became "Selecting within the Two-Branch Menus", with the
  inverted selection and DMS-SCFG added to its table and the interpretation
  rewritten as a diagnostic.
- Discussion and Limitations updated: DABP better on dense and sparse graphs
  by 0.1 to 0.2 percent, tied on scale-free; delayed split gain established out
  of sample on two families.
- Algorithm list: the two new split variants described; the pulse schedule is
  stated as fixed in advance on seeds 6000 to 6031.

Remaining placeholders in the text are marked `\todofill{}` (twelve of them,
all waiting for tonight's numbers).

## 6. New experiments

Run tonight through the same harness, seeds and 2000-iteration budget as
Table 1 (tmux `bp-sim-aaai`). Mean final cost over 50 instances:

| Family | DMS-SCFG | DMS-SCFG 0.95 | DMS-SCFG pulse | DMS-SCFG @1000 | DABP |
|---|---:|---:|---:|---:|---:|
| Graph coloring | 34.42 | 42.43 | 27.23 | 31.63 | 39.24 |
| Meeting scheduling | 8.10 | 13.16 | 8.14 | 8.04 | 27.22 |
| Random dense | 99758.48 | 99324.38 | 99409.53 | 99445.15 | 99223.68 |
| Random sparse | 14633.43 | 14522.63 | 14539.99 | 14554.99 | 14519.01 |
| Scale-free | 16761.71 | 16637.29 | 16658.59 | 16636.74 | 16628.46 |

The fixed 0.95 split is a 0.95/0.05 split with damping 0.9. The pulse starts
at 0.5, switches to 0.95/0.05 at iteration 64, and returns to 0.5 at
iteration 256 with the messages kept.

What the tests say on the random families (Wilcoxon, 50 paired instances):

- 0.95 split versus the symmetric split: lower by 0.4 to 0.8 percent on all
  three families, p below 1e-8, at least 44 wins of 50.
- 0.95 split versus the delayed split at K = 1000: lower on dense and sparse
  graphs (p = 0.03 each, not significant after Holm correction), a tie on
  scale-free graphs.
- 0.95 split versus DABP: tie on sparse (p = 0.19) and scale-free (p = 0.30);
  DABP better by 0.1 percent on dense (p = 0.010).
- pulse: between the symmetric and the 0.95 split everywhere; behind the
  0.95 split by 17 to 85 (p = 0.004 to 0.04, only scale-free survives Holm);
  behind DABP on all three families (p at most 0.004).
- settling: the 0.95 split settles later than the symmetric split (median
  100 to 185 versus 43 to 70) but still far earlier than DMS; 44 to 48 of 50
  runs settle. The pulse settles at 206 to 238; 48 to 50 of 50.

On the structured benchmarks the picture reverses:

- graph coloring: the pulse is the best line of all, 27.23 against 34.42 for
  the symmetric split (p = 0.0009, survives Holm) and 39.24 for DABP
  (p = 0.039, not after Holm). The permanent 0.95 split averages 42.43, but
  that is a heavy tail, not a shift: 24 wins against 21 losses, p = 0.90.
- meeting scheduling: the permanent 0.95 split is clearly worse, 13.16 against
  8.10 (p below 1e-5, 11 wins to 39 losses); the pulse ties the symmetric
  split at 8.14 (differences of hundredths; 9 wins to 41 losses, p = 0.008,
  Holm 0.057). Both stay far below DABP's 27.22.
- settling: the permanent 0.95 split settles in only 27 of 50 runs on
  coloring and 15 of 50 on meeting scheduling; the pulse settles in 43 of 50
  on both, at median iteration 267 and 303.

What this says: the split ratio is a knob with a sign that depends on the
family. Moving it away from 0.5 helps on the random families, where DABP's
learned weights sit, and hurts on the structured ones, where the symmetric
split has the strongest commitment certificate (Proposition on asymmetric
splits, and the Sept 15 certificate result). The temporary pulse takes most
of the random-family gain and none of the structured-family loss. Both lines
earn a column in Table 1; my recommendation is to keep both and drop MS-SCFG
and DABP-SymSplit to make room.

Earlier evidence for these lines: the 20-instance split-ratio sweep of Sept 13
(dense 0.95 split 99798 versus 100300 for 0.5; sparse 14388 versus 14518, on
the reference engine), and the pre-registered pulse confirmation of Sept 15
on 32 held-out sparse instances (0.58 percent lower cost than the 0.5 split,
25 wins, 7 ties, 0 losses).

What I did not run: a held-out selection of the pulse schedule on the other
families (the schedule was chosen on sparse graphs), learned controllers (they
did not beat fixed schedules on Sept 15), and DABP reruns (unaffected).

## 7. Plan for the three weeks

Week 1 (by Sept 22)
- Fill the twelve placeholders from tonight's runs; rerun the analysis scripts
  (done automatically at the end of the tmux job). Half a day.
- Regenerate the paper plots with the two new lines, copy them into
  `publish/plots/` (the copies there are still the July placeholders), and
  recompile with the AAAI kit to see the page count. One day.
- Write the Conclusion section (the July review asked for it; still missing).
  Half a day.
- Decide the Table 1 layout and whether the pulse line earns its column.

Week 2 (by Sept 29)
- Reproducibility checklist, anonymized repository, remove the placeholder
  URL. One day.
- Page budget: the current draft is over length. First cuts: census table to
  prose, fold Lemma decode into Lemma sibling (per README_BUILD), the
  DABP-SymSplit subsection to the appendix. One to two days.
- Optional experiment if space allows: settling curves as a figure (fraction
  of settled runs versus iteration) for the random dense family.

Week 3 (by Oct 6)
- Internal read-through against the July review's list, metadata cleaning,
  submission.

## 8. Decisions I need from you

1. Merge material: keep the diagnostic table in the main text (my
   recommendation, it is the empirical evidence for the two-solution reading)
   or move it to the appendix to save half a page.
2. Headline for the random families: "same final cost as DMS, settled ten
   times earlier" (my recommendation) versus keeping a final-cost claim, which
   the numbers no longer support on dense graphs.
3. The two new split lines. Both earn their place (the 0.95 split on the
   random families, the pulse on the structured ones); my recommendation is
   to keep both in Table 1 and drop MS-SCFG and DABP-SymSplit to make room.
4. Whether the intro should keep "delayed split improves on most benchmarks";
   the honest version is "best distributed line on every family; the gain
   over immediate splitting is out-of-sample significant on two".

## 9. Open items in the sources (found by building the paper tonight)

I compiled the split build with a substitute two-column article preamble
(tectonic; the AAAI kit is not in the repo, so margins and fonts are only
close to the real ones).

- It builds: no undefined references, no missing figures.
- 22 pages for main text plus both appendices. The 7-page main text plus
  supplement split the review response promised is still the biggest job.
- Three of the four bibliography files are not in `publish/` (`refs`,
  `DisCSP_refs`, `DampIns_AIJ`), so 69 citations are unresolved in the build.
  They are presumably in the Overleaf project; they need to be copied in.
- Table 1 overflows (see Section 6 above).
- Still to do from the July list: the anonymized repository URL (placeholder
  in the abstract's links block), the reproducibility checklist file (input
  by `main_revised.tex` but absent), stripping the `\todofill` macro, and
  clearing the PDF metadata before upload.

## 10. Where everything is

- Paper sources: `publish/` (committed today as a baseline, then edited;
  `git diff bc31f82 -- publish/` shows every change).
- Post-fix numbers: `experiments/aaai/data/*_summary.csv`,
  `*_significance.csv`; new scripts `key_comparisons.py` (Holm, bootstrap
  CI, wins/ties/losses, held-out K) and `settling.py` in
  `experiments/aaai/code/`, writing `*_key_comparisons.csv`,
  `*_heldout_k.csv`, `*_settling.csv` next to the data.
- New lines: `experiments/aaai/code/run_asym_pulse.sh`; the runner now knows
  the labels `DMS_split_0.95` and `DMS_split_pulse`.
- Logs: `experiments/aaai/logs/append_asym_pulse_20260915.log` and
  `rerun_ternary_axis_fix_20260915.log`.
- Engine equivalence check: run today on random sparse seed 0 with
  `experiments/splitting_explanation/lab.py` against the stored harness curve.
