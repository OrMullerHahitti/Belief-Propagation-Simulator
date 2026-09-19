# Research rationale and source boundaries

The main anchor remains the user's AAAI splitting work, locally titled
**How Does Function Splitting Improve Belief Propagation?** The inspected sources
are `publish/sec4_effect_of_splitting.tex` and `publish/sec5b_two_solutions.tex`,
with the September 9 saved manuscript audit and the September 15 source review
under `results/aaai_derived_control_20260915/theory_review.md`. These are distinct
source versions: the saved live audit identified `sub2.tex` as the configured
Overleaf main, and does not establish today's live manuscript contents. This
study makes no manuscript edits and does not assume all draft theorems survived
the prior audit.

## What the splitting argument actually permits

For an original factor table C, clone weight alpha, and the next actual incoming
Q message q, the factor sends

    r(v) = min_u {alpha C(u,v) + q(u)}.

A proposed unique sender u* remains the minimizer exactly when, for every
competitor z and receiver value v,

    q(z) - q(u*) + alpha [C(z,v) - C(u*,v)] > 0.

Equality is a possible branch boundary. It must be checked against the full
lower envelope; arbitrary pairwise line intersections need not be effective
boundaries. The experiment imports the saved verified boundary implementation
and compares weights immediately on either side of an effective boundary.

With identical active sender choices in the two complementary clones, their
combined C-dependent contribution cancels its dependence on w locally:
w C(u,v) + (1-w) C(u,v) = C(u,v). This explains why equal clones can absorb
weight changes without changing their combined belief. Merely preserving each
clone's own choices is not sufficient once the two clones choose different
senders. All these statements hold at fixed incoming q for the next update;
later message updates move the boundaries.

More precisely, on an open weight interval with fixed active labels a0(v),
a1(v) and fixed incoming q0,q1, the combined response is affine in w:

    s_w(v) = w C(a0(v),v) + (1-w) C(a1(v),v)
             + q0(a0(v)) + q1(a1(v)).

Thus its derivative is C(a0(v),v)-C(a1(v),v). The derivative of a normalized
message subtracts this value at the reference receiver label. This is a direct
substitution into the two minima, valid only while all active inequalities
remain satisfied. It proves cancellation when the clone choices coincide.
It also shows that, after their choices diverge, continuous reweighting can
change beliefs without crossing another boundary. The local action's crossing
score therefore detects one useful intervention channel, not every possible
effect of splitting. A future cost-directed rule could examine this affine
sensitivity as well as branch boundaries.

The implemented mechanism test branches from the same equal-split Q/R state
after 64 updates. It changes one edge for one update, restores .5/.5, and compares
both immediate beliefs and terminal original cost. It tests the immediate
mechanism. It does not establish that the fixed 192-update pulse obtains its
eventual gain specifically through a particular later boundary crossing.

Commitment, row churn, predicted row crossings under .95, and row margins enter
the controller explicitly. The local action attends to edges whose predicted
minimizers change, rather than treating every small weight adjustment as useful.
The ranking does not establish which changed row will improve original cost;
that missing connection is measured through actual outcomes.

## External angles checked

Elidan, McGraw and Koller's **Residual Belief Propagation** (UAI 2006) uses
current message residuals to inform asynchronous scheduling. It motivates
observing dynamical state instead of assigning theoretical significance to an
arbitrary update index. It does not justify terminal-cost improvement from our
synchronous split pulse, and this study does not implement their asynchronous
algorithm. [Primary paper](https://arxiv.org/abs/1206.6837)

Li, Chu, Langford and Schapire's **A Contextual-Bandit Approach to Personalized
News Article Recommendation** (WWW 2010) supplies a standard reference for
small linear action-outcome models and learning from selected-action feedback.
Our implementation uses ridge outcome heads and epsilon exploration, not
LinUCB's confidence-bound selection. Moreover, our actions change subsequent
solver states, so we do not import a contextual-bandit regret guarantee. The
paper is supporting methodology for a compact learner, not the splitting
theory anchor. [Primary paper](https://arxiv.org/pdf/1003.0146)

Damping is tested separately because small damped updates can mask a large
undamped fixed-point defect. A stable decoded assignment also need not imply
stable messages. The diagnostic checks pairwise and unary Q/R and the undamped
Q-map defect; removing damping throughout and switching it temporarily are
separate interventions. No generic convergence theorem is asserted.

## What would support a stronger paper claim

A robust pulse cost gain supports an empirical use of the splitting mechanism.
The inside/outside comparison supports the immediate active-row explanation.
Neither alone proves that crossing a boundary causes better final cost.
For that stronger claim, interventions would need matched controls for the
relevant route changes and their effect on original-objective quality.

A learned method must also beat the unchanged pulse and its matched frozen
exploration control on fresh inputs with comparable stability. Even that would
not establish that all four splitting-derived features are necessary: a
feature ablation and a fixed action-sequence control would still be needed.
Those attribution limits are kept explicit rather than labeling every adaptive
intervention a learning or attention improvement.
