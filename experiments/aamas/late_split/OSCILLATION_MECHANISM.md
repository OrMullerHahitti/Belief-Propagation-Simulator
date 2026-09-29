# Why Min-sum can keep selecting a few different values

This explains the general mechanism, before asking which conditions produce
exactly two or three selected values. It does not identify the ultimate cause
of the particular seed-0 tail. The benchmark protocol and saved runs are unchanged.

## 1. A selected value is the output of a changing score vector

For variable X_i, let D_i be its domain, F(i) its incident factors, and
R^t_(f→i)(v) the score that factor f sends for candidate v at update t.
Min-sum adds the incoming scores and selects a minimum:

\[
B_i^t(v)=\sum_{f\in F(i)}R^t_{f\to i}(v),\qquad
x_i^t=\arg\min_{v\in D_i}B_i^t(v).
\]

Unary costs, when present, are included as unary-factor messages. These are
cost scores, not probabilities. Only score differences matter for selection;
adding the same constant to all entries changes no winner in exact arithmetic.

The domain stays the same, and the algorithm still evaluates all its values.
It does not deliberately restrict itself to the values observed in a tail.
An assignment can stay fixed while the underlying scores keep changing.

## 2. What makes the scores change when the cost tables are fixed?

The implemented synchronous update first sends every variable-to-factor Q
message, then computes every factor-to-variable R message. For a pairwise
factor f with fixed table C_f(v,w):

\[
Q^t_{i\to f}(v)=\sum_{g\in F(i)\setminus\{f\}}R^{t-1}_{g\to i}(v),
\qquad
R^t_{f\to i}(v)=\min_{w\in D_j}\{C_f(v,w)+Q^t_{j\to f}(w)\}.
\]

The factor asks: **if i takes v, which value of j gives the cheapest
combination of this table and j's current incoming estimates?** The best w
can differ for different v, and can change at the next update. It is not
necessarily j's currently selected assignment.

Thus the table is fixed, but the input Q and the resulting conditional
estimate R change. The next iteration feeds these new estimates back into
the same equations. In these ordinary Min-sum updates, selected assignments
are readouts; they are not inserted as hard neighbor choices into the next
message calculation.

On a tree, influences cannot circulate indefinitely. On a graph with cycles,
they can return along another path. Excluding the recipient factor removes
the immediate echo through that same factor; it does not remove longer
feedback paths. Synchronous updates can therefore react to one another's
previous estimates repeatedly. Feedback can die out, settle without changing
the selected values, or keep reversing preferences. Having a loop alone
does not imply oscillation.

Equivalently, unrolling the recursion gives a growing computation tree.
Different copies of an original variable can take inconsistent values in
that tree. Locally minimal conditional estimates need not describe one
consistent joint assignment on the original graph. This is an interpretation
of the recursion, not an additional algorithm.

## 3. Why a jump occurs, and why it can persist

For two candidates a and b, define the preference gap

\[
G_i^t(b,a)=B_i^t(b)-B_i^t(a)
=\sum_f[R^t_{f\to i}(b)-R^t_{f\to i}(a)].
\]

A positive gap favors a over b; a negative gap favors b. To become the actual
winner, b must also beat every other candidate. A continuous change in the
scores can therefore produce an abrupt change in the selected discrete value.

Continued switching requires repeated changes in which candidate attains
the lowest score. A feedback loop can sustain those changes when returning
message differences remain large enough to overturn the current winning
margin. There is no acceptance rule requiring an update to lower the
decoded joint assignment's original cost. The exact example below demonstrates
both persistent switching and an increase in that cost.

Mathematically, the undamped message update is built from sums and minima of
affine expressions, so it is a continuous, piecewise-affine map. Different
minimizing choices select different pieces. The update need not contract
relative-message differences; recurring input patterns can sustain recurring
output patterns. A finite domain does not by itself make the full message
state finite, or establish a period for its decoded assignments.

## 4. Why only a few values may keep appearing

Feedback changes relative scores, but does not necessarily reverse every
comparison. A value with a sufficiently large disadvantage stays above the
minimum throughout those changes. The competing values are those whose
score curves actually reach the lower envelope of all candidate scores.

There is a precise sufficient condition. Write a relative score profile as
B_i^t(v)=b_i(v)+e_i^t(v), up to a common offset. Suppose over the time range
under discussion |e_i^t(v)|≤epsilon_v. If for some competing value a,

\[
b_i(v)-b_i(a)>\epsilon_v+\epsilon_a,
\]

then v never beats a in that range, hence v is never selected. Such a bound
must be established before using it to exclude future values; an observed
finite tail does not establish a permanent restriction.

Fixed cost tables also bound each factor's relative output, in exact
arithmetic. For finite table entries and finite incoming messages,

\[
\min_w[C_f(v,w)-C_f(a,w)]\le
R_{f\to i}(v)-R_{f\to i}(a)\le
\max_w[C_f(v,w)-C_f(a,w)].
\]

To see this, bound C_f(v,w) by C_f(a,w) plus the minimum/maximum difference,
add the same Q(w), and minimize. If the sum of the lower bounds over all
incident factors is strictly positive, v is always worse than a after a
native R update, regardless of the other incoming estimates. This is a
sufficient dominance test, not a claim that all absent values in our runs
pass it.

Another source of stable score shapes is the factor minimization itself:
if the same neighbor value w* minimizes C_f(v,w)+Q(w) for every v, then
R(v)=C_f(v,w*)+Q(w*). Its relative shape is a fixed column of the table
(or a row under the opposite axis convention). Changes in Q(w*) only add
a common offset until a different minimizing value takes over. Some messages
can have this property while others remain sensitive to several candidates.

These mechanisms explain how a small competitive set can emerge. They do
not guarantee that the set is small, that it is permanent, or that its size
equals the period of the joint assignment sequence. The number of surviving
candidates depends on the tables, graph, message state and update rule.

## 5. An exact example that closes the feedback loop

This is a deliberately constructed teaching example, not a reconstruction of
seed 0 or a claim about zero-initialized runs. Two variables X and Y each
have values A, B and C. Their original shared cost table is:

| X \\ Y | A | B | C |
|---|---:|---:|---:|
| A | 4 | 0 | 20 |
| B | 0 | 6 | 20 |
| C | 20 | 20 | 20 |

Represent the factor by two identical copies, each holding half this table.
Initialize each of the four factor-to-variable messages explicitly to
R=[0,1,10], in A/B/C order. The two original optimal assignments, (A,B)
and (B,A), both have cost zero; the following message and belief minima
are nevertheless strict and do not rely on tie-breaking.

Each outgoing Q excludes one factor and includes the other, so Q=[0,1,10].
For either clone, the next R is

\[
\begin{aligned}
R(A)&=\min(2+0,\ 0+1,\ 10+10)=1,\\
R(B)&=\min(0+0,\ 3+1,\ 10+10)=0,\\
R(C)&=\min(10+0,\ 10+1,\ 10+10)=10.
\end{aligned}
\]

So R becomes [1,0,10]. Feeding that vector back through the same calculation
gives [0,1,10] again. Both variables follow this calculation simultaneously:

| Update | Each incoming R | Each variable's belief (sum of both copies) | Joint selection | Original cost |
|---|---|---|---|---:|
| Specified starting state | [0,1,10] | [0,2,20] | (A,A) | 4 |
| Next update | [1,0,10] | [2,0,20] | (B,B) | 6 |
| Following update | [0,1,10] | [0,2,20] | (A,A) | 4 |

The full incoming-message state has returned, so the recurrence continues
indefinitely in exact arithmetic. The cost tables never change. A and B
exchange preference through feedback; C always has score 20 while a leader
has score 0. Hence C is never selected. The conditional estimates favor
opposite values for the two ends, but the synchronized decoded choices do
not realize those compatible pairs. All minima in the displayed message
updates and beliefs are unique.

The construction illustrates the mechanism. It is not evidence that arbitrary
graphs or arbitrary initializations must produce this particular period.

Validation uses the repository's actual `MinSumComputator.compute_Q`,
`compute_R`, `compute_belief`, and `get_assignment`, with exact array assertions:

```sh
uv run --no-sync python experiments/aamas/runs/oscillation_explanation_20260921/verify_example.py
```

The companion `verification.json` records the checked vectors and source hashes.

## 6. What splitting and damping change

Splitting preserves each assignment's objective because C/2+C/2=C. It changes
the feedback: Q toward the first copy excludes that copy's R but still
contains the second copy's R. The path X→first copy→Y→second copy→X is a
four-edge factor-graph cycle. The exact example above uses that path.
Objective preservation is therefore not preservation of the message dynamics.

In the current late-split implementation, damping before the split mixes
outgoing Q messages as Q_sent=lambda*Q_old+(1-lambda)*Q_new, with lambda=0.9.
After splitting, lambda=0 and Q is undamped. Smoothing can reduce repeated
preference reversals, but it does not universally guarantee convergence.
The split checkpoint determines the starting message state for this changed
update rule; its good original cost alone does not establish stability.

## Evidence and scope

- Native schedule: `src/propflow/bp/engine_base.py`, `BPEngine.step`.
- Native equations and readout: `src/propflow/bp/computators.py`,
  `compute_Q`, `compute_R`, `compute_belief`, `get_assignment`;
  `src/propflow/core/agents.py`, `VariableAgent.curr_assignment`.
- Splitting: `src/propflow/policies/splitting.py`, `_split_factors`;
  `src/propflow/bp/engines.py`, `MidRunSplitEngine._transfer_messages`.
- Damping: `src/propflow/policies/damping.py`, `_apply_damping`;
  `experiments/aamas/late_split/core.py`, `ReleasedDampingSplitEngine`.
- [Yedidia, Message-Passing Algorithms for Inference and Optimization,
  MERL TR2011-087](https://www.merl.com/publications/docs/TR2011-087.pdf),
  Sections 5–6 and 9: Min-sum equations, loopy interpretation and splitting.
- See [primary-source scope notes](oscillation_explanation_sources.md) for
  computation-tree references and limits on periodicity claims.

The exact equations, conditional dominance arguments and constructed feedback
example establish the general mechanism. They do not establish which feedback
path is necessary in the observed seed-0 run, predict its eventual behavior,
or prove that a small number of values must occur in every instance. Those
are separate questions requiring additional hypotheses or instance evidence.
