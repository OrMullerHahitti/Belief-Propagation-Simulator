"""Package exact trajectories in a self-contained experiment inspector."""

from __future__ import annotations

import base64
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np
from plotly.offline import get_plotlyjs

from .analysis import (
    applied_coefficients,
    node_changes,
    node_split_changes,
    paired_sources,
)
from .graph import canonical_json, fingerprint


def encode_array(array: np.ndarray) -> dict:
    """Losslessly encode column-contiguous float64 trajectories for the reader."""
    packed = np.asarray(array.T, dtype="<f8", order="C")
    return {
        "shape": list(array.shape),
        "data": base64.b64encode(zlib.compress(packed.tobytes(), level=6)).decode(),
    }


def load_run(path: Path, graph_sha256: str) -> tuple[dict, dict]:
    """Validate saved provenance before deriving any displayed measurement."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        if metadata["graph_sha256"] != graph_sha256:
            raise ValueError(f"{path.name} belongs to a different graph")
        damped = data["damped"]
        attention = data["attention"]
        iterations = data["iteration"]
        if not np.array_equal(iterations, np.arange(1, len(damped) + 1)):
            raise ValueError("missing or reordered iterations")
        if attention.shape[0] != len(damped):
            raise ValueError("damping and attention iteration counts differ")
        if metadata["outcome"]["iterations"] != len(damped):
            raise ValueError("outcome does not match recorded trajectory length")
        source_target = np.asarray(metadata["src_trg_idxes"], dtype=np.int32)
        damping, shares, coefficients = applied_coefficients(
            damped, attention, source_target
        )
        names = metadata["ordered_names"]
        by_name = {name: index for index, name in enumerate(names)}
        target_owners = np.array([by_name[name] for name in metadata["trg_var_names"]])
        source_owners = target_owners[source_target]
        arrays = {
            "damping": encode_array(damping),
            "new_weight": encode_array(damped[..., 0, :].mean(axis=-1)),
            "attention": encode_array(shares),
        }
        summary = {
            "damping": node_changes(damping, target_owners, len(names)),
            "coefficient": node_changes(coefficients, source_owners, len(names)),
        }
        overview = {
            "max_damping_change": np.abs(np.diff(damping, axis=0)).max(axis=1).tolist(),
            "max_coefficient_change": np.abs(np.diff(coefficients, axis=0))
            .max(axis=1)
            .tolist(),
        }
        pairs = paired_sources(metadata)
        return {
            "meta": metadata,
            "arrays": arrays,
            "summary": summary,
            "split_summary": node_split_changes(
                coefficients, pairs, target_owners, len(names), metadata["split_ratio"]
            ),
            "overview": overview,
            "pairs": pairs,
            "target_owners": target_owners.tolist(),
            "source_count": np.bincount(
                source_target, minlength=len(target_owners)
            ).tolist(),
        }, metadata["outcome"]


def build_report(output: Path) -> Path:
    """Build the offline reader solely from the four saved experiment files."""
    problem = json.loads((output / "graph.json").read_text())
    manifest = json.loads((output / "run.json").read_text())
    graph_hash = fingerprint(problem)
    if manifest["graph_sha256"] != graph_hash:
        raise ValueError("saved graph checksum does not match run.json")
    payload = {"graph": problem, "manifest": manifest, "runs": {}}
    for variant in ("symmetric", "asymmetric"):
        run, outcome = load_run(output / f"{variant}.npz", graph_hash)
        if outcome != manifest["runs"][variant]:
            raise ValueError(f"{variant} outcome does not match run.json")
        if run["meta"]["settings"] != manifest["settings"]:
            raise ValueError(f"{variant} settings do not match run.json")
        payload["runs"][variant] = run
    runs = payload["runs"]
    left, right = runs["symmetric"]["meta"], runs["asymmetric"]["meta"]
    if left["initial_model_sha256"] != right["initial_model_sha256"]:
        raise ValueError("network initialization differs between variants")
    for key in (
        "ordered_names",
        "fn_factor_names",
        "fn_half",
        "trg_fn_idxes",
        "src_fn_idxes",
        "src_trg_idxes",
        "trg_var_names",
    ):
        if left[key] != right[key]:
            raise ValueError(f"paired provenance mismatch: {key}")
    root = Path(__file__).parent
    manifest["report_source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            root / name
            for name in ("report.py", "reader.template.html", "reader.css", "reader.js")
        )
    }
    template = (root / "reader.template.html").read_text()
    html = template.replace("__STYLE__", (root / "reader.css").read_text())
    html = html.replace("__PLOTLY__", get_plotlyjs())
    html = html.replace("__DATA__", canonical_json(payload).replace("</", "<\\/"))
    html = html.replace("__SCRIPT__", (root / "reader.js").read_text())
    result = output / "report.html"
    result.write_text(html)
    (output / "run.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(f"report ready: {result} ({result.stat().st_size / 1e6:.1f} MB)", flush=True)
    return result
