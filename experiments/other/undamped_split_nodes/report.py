"""Create inspectable figures and a compact inline dataset from saved traces."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx
import numpy as np

from experiments.aaai.code.utils.plot_helpers import remove_frame
from .run import OUTPUT, SELECTED, load_problem, write_csv

plt.switch_backend("Agg")


def plots(family: str, analysis: dict) -> None:
    """Plot real observations, with route labels chosen from the observed tail."""
    p = load_problem(family)
    graph = nx.Graph()
    graph.add_nodes_from(p.variable_names)
    graph.add_edges_from((p.variable_names[u], p.variable_names[v]) for u, v in p.edges)
    positions = {n["variable"]: n["position"] for n in analysis["nodes"]}
    colors = ["#bb5540" if n["unsettled"] else "#467b9b" for n in analysis["nodes"]]
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_edges(graph, positions, ax=ax, alpha=0.17, width=0.5)
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_color=colors, node_size=330)
    nx.draw_networkx_labels(graph, positions, ax=ax, font_size=9, font_color="white")
    ax.set_title(
        f"{family.replace('_', ' ')} | red: verdict alternates; blue: verdict fixed"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{family}_graph.pdf")
    fig.savefig(OUTPUT / f"{family}_graph.png", dpi=150)
    plt.close(fig)
    palette = plt.get_cmap("tab10")
    with PdfPages(OUTPUT / f"{family}_beliefs_and_cavities.pdf") as pdf:
        for name, detail in analysis["selected"].items():
            trace = np.load(OUTPUT / f"{family}_{name}.npz")
            a, b = detail["values"]
            fig, axes = plt.subplots(2, 2, figsize=(13, 8))
            for ax, indices in zip(axes[0], (np.arange(80), np.arange(1980, 2000))):
                belief = trace["belief"][indices]
                belief = belief - belief.min(axis=1, keepdims=True)
                for value in range(p.d):
                    ax.plot(
                        indices + 1,
                        belief[:, value],
                        label=str(value),
                        color=palette(value),
                        linewidth=1.4 if value in [a, b] else 0.7,
                        alpha=1 if value in [a, b] else 0.45,
                    )
                ax.set(
                    xlabel="Completed updates",
                    ylabel="Belief minus its minimum (cost units)",
                )
                remove_frame(ax)
            axes[0, 1].legend(title="Value", ncol=5, fontsize=9)
            contributions = trace["r"][..., b].sum(2) - trace["r"][..., a].sum(2)
            top = np.argsort(np.abs(contributions[-1] - contributions[-2]))[-4:][::-1]
            for f in top:
                axes[1, 0].plot(
                    np.arange(80) + 1,
                    contributions[:80, f],
                    label=f"{detail['labels'][f]} / {detail['partners'][f]}",
                )
            axes[1, 0].axhline(0, color="grey", linewidth=0.5)
            axes[1, 0].set(
                xlabel="Completed updates",
                ylabel=f"Pair contribution to B({b}) − B({a})",
            )
            axes[1, 0].legend(fontsize=8)
            remove_frame(axes[1, 0])
            e = detail["edge_indices"][int(top[0])]
            node = list(p.variable_names).index(name)
            table = p.costs[e] if p.edges[e, 0] == node else p.costs[e].T
            image = axes[1, 1].imshow(table, cmap="cividis", vmin=100, vmax=199)
            axes[1, 1].set(
                xlabel=f"{detail['partners'][int(top[0])]} value",
                ylabel=f"{name} value",
                title=f"Original {detail['labels'][int(top[0])]} table; each clone = half",
            )
            axes[1, 1].set_xticks(range(p.d))
            axes[1, 1].set_yticks(range(p.d))
            for y in range(p.d):
                for x in range(p.d):
                    axes[1, 1].text(
                        x,
                        y,
                        f"{table[y, x]:.0f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if table[y, x] < 150 else "black",
                    )
            fig.colorbar(image, ax=axes[1, 1], label="Original cost")
            fig.suptitle(
                f"{family}: {name} | observed route/competitor values {a}, {b}"
            )
            fig.tight_layout()
            pdf.savefig(fig)
            fig.savefig(OUTPUT / f"{family}_{name}_overview.png", dpi=140)
            plt.close(fig)
            # every incident original factor has its own incoming/cavity trace
            for start in range(0, len(detail["labels"]), 6):
                fig, axes = plt.subplots(3, 2, figsize=(13, 10), squeeze=False)
                for ax, f in zip(
                    axes.flat, range(start, min(start + 6, len(detail["labels"])))
                ):
                    total = trace["belief"][:80, b] - trace["belief"][:80, a]
                    clone = (
                        trace["cavity"][:80, f, 0, b] - trace["cavity"][:80, f, 0, a]
                    )
                    external = (
                        trace["external_cavity"][:80, f, b]
                        - trace["external_cavity"][:80, f, a]
                    )
                    for series, label in [
                        (total, "Full belief"),
                        (contributions[:80, f], "Both incoming clones"),
                        (clone, "Cavity excluding one clone"),
                        (external, "Cavity excluding both"),
                    ]:
                        ax.plot(np.arange(80) + 1, series, linewidth=0.8, label=label)
                    ax.set(
                        title=f"{detail['labels'][f]} / {detail['partners'][f]}",
                        xlabel="Completed updates",
                        ylabel=f"Gap: value {b} minus {a}",
                    )
                    ax.axhline(0, color="grey", linewidth=0.5)
                    remove_frame(ax)
                for ax in list(axes.flat)[min(6, len(detail["labels"]) - start) :]:
                    ax.set_visible(False)
                axes[0, 0].legend(fontsize=7, ncol=2)
                fig.suptitle(
                    f"{family} — {name}: every incident factor (positive gap favors {a})"
                )
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


def report() -> dict:
    """Build a durable report with all 100 node rows and auditable data links."""
    lines = [
        "# Undamped equal-split variable analysis",
        "",
        "Saved sparse and dense seed-5000 AAAI inputs; 50 variables, 10 values (0–9). "
        "Every original pairwise and unary factor is split 0.5/0.5; damping is zero. "
        "No controller, DABP, training, or cost-table modification is used.",
        "",
        "A verdict is unsettled when its preferred value changes in updates 1701–2000. "
        "We separately checked updates 9701–10000. Beliefs are cost vectors, so lower is better. "
        "This is a description of two saved instances, not a prevalence estimate or convergence theorem.",
        "",
        "## Main findings",
        "",
        "- Sparse: 35 variables alternate; dense: 39 alternate. Every unsettled variable uses "
        "exactly two values in the checked tails. The remaining 15 and 11 keep one verdict, "
        "while their beliefs still move.",
        "- Each unsettled set forms one biconnected component. Neither has an internal "
        "articulation vertex or bridge. Both exchange messages with settled variables.",
        "- Sparse settled and unsettled variables both have mean degree 5.0. Dense means are "
        "27.18 and 28.08. Mean betweenness and cost-table variability also overlap strongly; "
        "these summaries do not explain verdict stability on their own.",
        "- All 15,000 next-verdict observations per graph in the primary tail minimize the "
        "original local objective conditioned on the previous neighbor verdicts. Thus the "
        "observed tail is closed under synchronous local best responses, consistent with the "
        "paper's splitting analysis. This does not prove why the trajectory entered that pair.",
        "- Sparse x28 is the clearest cavity example: it has only two neighbors, x23 and x34, "
        "yet alternates 9 ↔ 4. The cavity excluding one clone differs from the external "
        "cavity excluding both; retaining the sibling changes the cavity winner on half the "
        "checked steps for each incident pair.",
        "- Sparse x7 (degree 9) keeps value 2 from the first recorded update, despite belief "
        "motion of about 334 cost units. Dense x48 (degree 34) keeps value 0 after update 11, "
        "despite belief motion of about 486 units.",
        "",
        "## Definitions and numerical limits",
        "",
        "Original degree counts distinct variable neighbors; split-factor degree includes "
        "both clones of each pairwise factor and both unary clones. Betweenness is "
        "normalized, unweighted shortest-path betweenness on the original variable graph. "
        "Core, triangles, clustering, PageRank, greedy-modularity community, and community "
        "participation are in the variable CSVs. Components are induced only by unsettled "
        "variables; their boundaries retain the original graph connections.",
        "",
        "Table interaction RMS removes the table's row and column means and restores its "
        "grand mean. It measures non-additive interaction, rather than just overall cost "
        "scale. Effective rank is the participation ratio of squared singular values. UU / US "
        "/ SS mean factors between two unsettled, mixed, or two settled variables. These "
        "comparisons are descriptive; graph nodes and edges are dependent observations.",
        "",
        "At completed update t, B(t) is the sum of incoming R(t). The candidate cavity to "
        "clone a is B(t)−R_a(t); the external cavity removes both siblings. Actual Q sent at "
        "update t+1 is recorded separately. Pair contributions and cavity plots use a "
        "difference between two actual route values; for fixed-verdict comparisons the second "
        "value is the lowest mean-belief alternative in the tail. Full 10-value vectors "
        "remain in the NPZ files and belief plots.",
        "",
        "Native replay verified every assignment and all pairwise Q/R and unary R differences "
        "for 2,000 updates per graph. Raw Q offsets can differ because returned native step "
        "objects precede scheduled normalization; all reference-label message differences "
        "match exactly. Native objective totals differ by at most floating-point summation "
        "error. Large native common offsets also cause small non-additivity in diagnostics: "
        "selected sparse belief reconstruction errors are at most 0.015625, dense at most "
        "0.000477. These residuals are retained in the node NPZ files and are much smaller "
        "than late preference margins. Numerical beliefs therefore are not claimed to be "
        "exact two-cycles.",
        "",
    ]
    inline = {}
    indices = np.r_[np.arange(60), np.arange(1980, 2000)]
    for family in SELECTED:
        a = json.loads((OUTPUT / f"{family}_analysis.json").read_text())
        p = load_problem(family)
        with np.load(OUTPUT / f"{family}_scout.npz") as source:
            x, beliefs = source["assignments"], source["beliefs"]
        pred = x[-2 + np.arange(len(x)) % 2]
        bad = np.flatnonzero((x != pred).any(axis=1))
        entry = int(bad[-1]) + 2 if len(bad) else 1
        lines += [
            f"## {family.replace('_', ' ').title()}",
            "",
            f"{a['edges']} edges; realized density {a['density']:.4f}. "
            f"The final assignment two-cycle begins at update {entry} and persists through update 10,000. "
            f"Its two original-objective costs are {a['phase_costs'][0]:.3f} and {a['phase_costs'][1]:.3f}.",
            "",
            f"**Unsettled:** {', '.join(a['unsettled'])}.",
            "",
            f"**Fixed verdict:** {', '.join(a['settled'])}.",
            "",
        ]
        component = a["components"][0]
        lines += [
            f"The unsettled component has {component['edges']} internal edges, diameter {component['diameter']}, "
            f"cycle rank {component['cycle_rank']}, and {component['cut_edges']} boundary edges to "
            f"{len(component['boundary_variables'])} fixed-verdict variables. "
            f"Conductance is {component['conductance']:.4f}. It has no articulation point or bridge.",
            "",
        ]
        comparisons = []
        for metric in [
            "degree",
            "betweenness",
            "clustering",
            "core",
            "incident_interaction_rms",
            "incident_table_std",
            "min_margin",
        ]:
            values = {"metric": metric}
            for group in [True, False]:
                v = [n[metric] for n in a["nodes"] if n["unsettled"] == group]
                values["unsettled_mean" if group else "settled_mean"] = float(
                    np.mean(v)
                )
            comparisons.append(values)
        write_csv(OUTPUT / f"{family}_group_comparison.csv", comparisons)
        lines += [
            "| Variable | Verdict A → B | Degree | Betweenness | Core | Unsettled neighbors | Minimum margin |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for n in a["nodes"]:
            lines.append(
                f"| {n['variable']} | {n['phase_A']} → {n['phase_B']} | {n['degree']} | "
                f"{n['betweenness']:.5f} | {n['core']} | {n['unsettled_neighbors']} | {n['min_margin']:.3f} |"
            )
        lines += [
            "",
            f"[Graph]({family}_graph.pdf) · "
            f"[Beliefs and every selected-node cavity]({family}_beliefs_and_cavities.pdf) · "
            f"[All node metrics]({family}_variables.csv) · [All factor metrics]({family}_factors.csv) · "
            f"[Original tables and ordered axes]({family}_cost_tables.json)",
            "",
        ]
        for name, d in a["selected"].items():
            leading = sorted(d["factors"], key=lambda f: abs(f["swing"]), reverse=True)[
                :3
            ]
            lines += [
                f"### {name}",
                "",
                f"Compared values {d['values'][0]} and {d['values'][1]}. "
                "Largest factor contribution changes between the final two phases:",
                "",
                "| Factor / neighbor | Phase A gap | Phase B gap | Change |",
                "|---|---:|---:|---:|",
            ]
            for f in leading:
                lines.append(
                    f"| {f['factor']} / {f['neighbor']} | {f['contribution_A']:.3f} | "
                    f"{f['contribution_B']:.3f} | {f['swing']:.3f} |"
                )
            lines += [
                "",
                f"[All factor contributions]({family}_{name}_factor_contributions.csv) · "
                f"[Overview]({family}_{name}_overview.png) · "
                f"[Full 2,000-step vectors]({family}_{name}.npz)",
                "",
            ]
        plots(family, a)
        details = {}
        for name, d in a["selected"].items():
            trace = np.load(OUTPUT / f"{family}_{name}.npz")
            va, vb = d["values"]
            contribution = trace["r"][indices, ..., vb].sum(2) - trace["r"][
                indices, ..., va
            ].sum(2)
            factors = []
            i = list(p.variable_names).index(name)
            for j, e in enumerate(d["edge_indices"]):
                c = (
                    p.unary[i : i + 1]
                    if e < 0
                    else (p.costs[e] if p.edges[e, 0] == i else p.costs[e].T)
                )
                factors.append(
                    dict(
                        name=d["labels"][j],
                        neighbor=d["partners"][j],
                        table=np.round(c, 4).tolist(),
                        gap=np.round(contribution[:, j], 2).tolist(),
                    )
                )
            details[name] = dict(values=d["values"], factors=factors)
        inline[family] = dict(
            nodes=a["nodes"],
            edges=p.edges.tolist(),
            steps=(indices + 1).tolist(),
            beliefs=np.round(beliefs[indices], 2).tolist(),
            details=details,
            component=a["components"][0],
        )
    lines += [
        "## Validation and reproduction",
        "",
        "Focused snapshot and dynamics suite: 17 passed. Native parity: 4,000 updates, zero "
        "assignment mismatches, zero gauge-message discrepancy. The explicit `tests/` suite "
        "has 327 passed, 3 preexisting Figure 5/8 failures, and 2 skipped. `make ci` stops on "
        "seven existing Black-format failures. Repository-wide discovery also encounters "
        "duplicate test names in the two preexisting submission-copy directories. Standalone "
        "type-checking encounters missing NetworkX stubs; checking this module with missing "
        "imports ignored passes.",
        "",
        "Run from the repository with `.venv/bin/python -m "
        "experiments.other.undamped_split_nodes.run all`, then `.venv/bin/python -m "
        "experiments.other.undamped_split_nodes.report`. Inputs are loaded from the earlier "
        "study's saved NPZ files; their hashes are in each native-validation JSON. All "
        "simulation arrays retain every update; only the inline view limits its display to "
        "updates 1–60 and 1981–2000.",
        "",
    ]
    report_text = "\n".join(lines)
    for section_name in ["THRESHOLD_SECTION.md", "THRESHOLD_TRANSITIONS.md"]:
        threshold_section = OUTPUT / section_name
        if threshold_section.exists():
            report_text = report_text.replace(
                "## Random Sparse",
                threshold_section.read_text() + "\n## Random Sparse",
                1,
            )
    (OUTPUT / "REPORT.md").write_text(report_text)
    (OUTPUT / "inline_data.json").write_text(
        json.dumps(inline, separators=(",", ":"), allow_nan=False)
    )
    return inline


if __name__ == "__main__":
    report()
