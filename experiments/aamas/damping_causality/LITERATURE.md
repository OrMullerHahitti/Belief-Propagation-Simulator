# Relation to prior work

Inspected 15 September 2026. These references motivate the distinction between
message dynamics and solution quality; they do not certify our new fixtures.

- Cohen, Galiki and Zivan, *Governing convergence of Max-sum on DCOPs through
  damping and splitting*, Artificial Intelligence 279 (2020), 103212:
  <https://www.sciencedirect.com/science/article/pii/S0004370219302061>.
  This is prior work on the combined algorithm; we do not claim that combining
  splitting and damping is a new method.
- Zaed, Lev and Zivan, *Insights Regarding the Success of Damping in Improving
  Belief Propagation*, AAMAS 2025:
  <https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2025/pdfs/p2281.pdf>.
  The paper analyzes cost-tree coefficients and emphasizes early inconsistent
  beliefs. It also distinguishes convergence of beliefs to equal values from
  unambiguous assignment selection. Our exact small split-path example examines
  feedback and changes of minimizing rows directly. This is complementary
  evidence, not a claim that generic temporal averaging is newly discovered.
- Local splitting analysis: `publish/sec4_effect_of_splitting.tex` and
  `publish/sec5b_two_solutions.tex`. These are local draft sources. The saved
  September 9 Overleaf audit is a separate version and is not today's live
  manuscript. We retain conditional and pattern-specific claims and do not
  treat a scalar symmetric-subspace argument as global graph convergence.

The derivations in `theory.md`, native replay, and saved rational certificates
are the evidence for the conclusions of this investigation. No external paper
supplies a universal necessity or sufficiency guarantee for the present
arbitrary discrete loopy factor graphs.
