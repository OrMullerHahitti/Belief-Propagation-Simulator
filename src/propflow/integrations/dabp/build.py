"""Convert a PropFlow ``FactorGraph`` into the tensor inputs DABP consumes.

This is a port of ``DABP-main/alg/data.py`` ``COP_Dataset.__getitem__`` (and its
``label_variables`` helper): instead of parsing an XML problem file, it reads the
variables and binary cost tables directly off a ``FactorGraph``. The message /
scatter index construction is reproduced verbatim so the vendored model runs
unchanged.

DABP only supports **binary** (arity-2) factors with a single uniform domain. We
validate both and raise a clear error otherwise. Costs are divided by ``scale``
(training magnitude only — reported cost is always evaluated on PropFlow's
original tables), and every factor is split into two clones distributing the
cost via ``split_ratio`` (DABP's built-in SCFG step).
"""

from __future__ import annotations

import numpy as np

from .constant import MAX_COLOR_NUM, FUN_ID, SCALE, SPLIT_RATIO


def label_variables(adj_list: dict) -> dict:
    """greedy graph coloring used for DABP's variable embeddings."""
    ordered_keys = sorted(adj_list.keys())
    labels: dict = {}
    for vn in ordered_keys:
        all_colors = {labels[nb] for nb in adj_list[vn] if nb in labels}
        color = 0
        for color in range(MAX_COLOR_NUM):
            if color not in all_colors:
                break
        labels[vn] = color
    return labels


def build_dabp_inputs(fg, node_embed_dim: int = 12, scale: float = SCALE, split_ratio: float = SPLIT_RATIO):
    """build the DABP input dict from a FactorGraph.

    Returns ``(data, ordered_names, domain)`` where ``data`` feeds
    ``AttentiveBP.preprocess_single`` and ``ordered_names[k]`` is the
    VariableAgent name corresponding to belief row ``k``.
    """
    variables = list(fg.variables)
    domains = {v.domain for v in variables}
    if len(domains) != 1:
        raise ValueError(f"DABP requires a single uniform variable domain; got {sorted(domains)}")
    domain = domains.pop()
    max_dom_size = domain
    ordered_names = [v.name for v in variables]

    # --- collect binary cost matrices + unary potentials (both scaled) --------
    # DABP only models binary factors. Unary factors (e.g. tie-break prefs) are
    # folded into one incident binary factor so the total cost is preserved.
    unary = {name: np.zeros(domain, dtype=float) for name in ordered_names}
    base = []  # mutable [matrix(scaled), row_name, col_name]
    for f in fg.factors:
        cn = getattr(f, "connection_number", {}) or {}
        if f.cost_table is None or not cn:
            continue
        ct = np.asarray(f.cost_table, dtype=float)
        if len(cn) == 1:
            ((vname, _dim),) = cn.items()
            unary[vname] = unary[vname] + ct.reshape(-1) / scale
        elif len(cn) == 2 and set(cn.values()) == {0, 1}:
            inv = {dim: name for name, dim in cn.items()}
            base.append([ct / scale, inv[0], inv[1]])
        else:
            raise ValueError(f"DABP supports only unary/binary factors; factor '{f.name}' " f"has arity {len(cn)}")

    # fold each variable's accumulated unary cost into one incident binary factor
    folded = set()
    for entry in base:
        m, row, col = entry
        if row not in folded and np.any(unary[row]):
            m = m + unary[row][:, None]
            entry[0] = m
            folded.add(row)
        if col not in folded and np.any(unary[col]):
            m = m + unary[col][None, :]
            entry[0] = m
            folded.add(col)
    missing = [n for n in ordered_names if np.any(unary[n]) and n not in folded]
    if missing:
        raise ValueError(
            f"DABP: variables {missing} carry unary costs but have no incident " "binary factor to fold them into."
        )

    all_matrix = []
    for m, row, col in base:
        all_matrix.append((m * split_ratio, row, col))
        all_matrix.append((m * (1.0 - split_ratio), row, col))

    NV = len(variables)
    NF = len(all_matrix)

    # ---------------- adjacency & ordered vars (ported from data.py) ----------
    adj_list: dict = {}
    var_index: dict = {}
    adj_func_list: dict = {}
    rv_idxes = []
    cv_idxes = []
    for idx, vn in enumerate(ordered_names):
        adj_list[vn] = []
        adj_func_list[vn] = []
        var_index[vn] = idx

    for idx, (_, row, col) in enumerate(all_matrix):
        assert row != col
        if row not in adj_list[col]:
            adj_list[row].append(col)
            adj_list[col].append(row)
        adj_func_list[row].append(idx)
        adj_func_list[col].append(idx)
        rv_idxes.append(var_index[row])
        cv_idxes.append(var_index[col])

    if len(set(rv_idxes + cv_idxes)) != NV:
        raise ValueError("DABP requires every variable to appear in at least one factor " "(no isolated variables).")

    vn_color_dict = label_variables(adj_list)
    var_embed = [vn_color_dict[vn] for vn in ordered_names]

    padding = [0 for _ in range(node_embed_dim - 4)]
    func_embed = [FUN_ID + padding for _ in range(NF)]

    # ---------------- message buffers & per-direction indices -----------------
    msg_hidden = [[], []]
    msgs = []
    msg_rv2f_idxes, msg_cv2f_idxes, msg_f2rv_idxes, msg_f2cv_idxes = [], [], [], []
    i = 0
    for _ in range(NF):
        msg_hidden[0].append(list(padding))
        msgs.append([0] * max_dom_size)
        msg_rv2f_idxes.append(i)
        i += 1
    for _ in range(NF):
        msg_hidden[0].append(list(padding))
        msgs.append([0] * max_dom_size)
        msg_cv2f_idxes.append(i)
        i += 1
    for _ in range(NF):
        msg_hidden[1].append(list(padding))
        msgs.append([0] * max_dom_size)
        msg_f2rv_idxes.append(i)
        i += 1
    for _ in range(NF):
        msg_hidden[1].append(list(padding))
        msgs.append([0] * max_dom_size)
        msg_f2cv_idxes.append(i)
        i += 1
    msg_v2f_idxes = msg_rv2f_idxes + msg_cv2f_idxes
    msg_f2v_idxes = msg_f2rv_idxes + msg_f2cv_idxes

    # ---------------- node-feature layout & edges -----------------------------
    x_f_start_idx = NV
    x_rv2f_start_idx = NV + NF
    x_cv2f_start_idx = NV + 2 * NF
    x_f2rv_start_idx = NV + 3 * NF
    x_f2cv_start_idx = NV + 4 * NF

    src, dst = [], []
    for k, (_, row, col) in enumerate(all_matrix):
        row_idx = var_index[row]
        col_idx = var_index[col]
        # rv -> m -> f
        src.append(row_idx)
        dst.append(k + x_rv2f_start_idx)
        src.append(k + x_rv2f_start_idx)
        dst.append(k + x_f_start_idx)
        # cv -> m -> f
        src.append(col_idx)
        dst.append(k + x_cv2f_start_idx)
        src.append(k + x_cv2f_start_idx)
        dst.append(k + x_f_start_idx)
        # f -> m -> rv
        src.append(k + x_f_start_idx)
        dst.append(k + x_f2rv_start_idx)
        src.append(k + x_f2rv_start_idx)
        dst.append(row_idx)
        # f -> m -> cv
        src.append(k + x_f_start_idx)
        dst.append(k + x_f2cv_start_idx)
        src.append(k + x_f2cv_start_idx)
        dst.append(col_idx)
    edge_index = [src, dst]

    # ---------------- attention target/source + belief scatter indices --------
    msg_trg_idxes = []
    msg_src_idxes = []
    embed_trg_idxes = []
    embed_src_idxes = []
    v2f_scatter_idxes = []
    trg_scatter_idx = 0

    msg_f2v_per_v_idxes = []
    f2v_per_v_scatter_idxes = []
    f2v_per_v_scatter_idx = 0

    degrees = []
    unary_var_cnt = 0
    for vn in ordered_names:
        msg_vn2f_idxes = []
        msg_f2vn_idxes = []
        for fn_idx in adj_func_list[vn]:
            _, row, col = all_matrix[fn_idx]
            if row == vn:
                msg_vn2f_idxes.append(fn_idx)  # rv -> f
                msg_f2vn_idxes.append(fn_idx + 2 * NF)  # f -> rv
            else:
                assert col == vn
                msg_vn2f_idxes.append(fn_idx + NF)  # cv -> f
                msg_f2vn_idxes.append(fn_idx + 3 * NF)  # f -> cv
        msg_f2v_per_v_idxes.append(msg_f2vn_idxes)
        f2v_per_v_scatter_idxes.append([f2v_per_v_scatter_idx] * len(msg_f2vn_idxes))
        f2v_per_v_scatter_idx += 1

        func_list = adj_func_list[vn]
        if len(func_list) == 1:
            unary_var_cnt += 1
            continue
        degree = len(func_list)
        degrees.append(degree)
        for j in range(degree):
            msg_trg_idxes.append(msg_vn2f_idxes[j])
            tmp = list(msg_f2vn_idxes)
            tmp.pop(j)
            msg_src_idxes.append(tmp)

            fn_idx = func_list[j]
            embed_trg_idxes.append(fn_idx + NV)
            embed_src_idxes.append([t + NV for t in func_list if t != fn_idx])
            v2f_scatter_idxes.append([trg_scatter_idx] * len(embed_src_idxes[-1]))
            trg_scatter_idx += 1

    msg_src_idxes = [j for sub in msg_src_idxes for j in sub]
    embed_src_idxes = [j for sub in embed_src_idxes for j in sub]
    v2f_scatter_idxes = [j for sub in v2f_scatter_idxes for j in sub]
    msg_f2v_per_v_idxes = [j for sub in msg_f2v_per_v_idxes for j in sub]
    f2v_per_v_scatter_idxes = [j for sub in f2v_per_v_scatter_idxes for j in sub]

    # ---------------- padded cost tensors -------------------------------------
    cost_tensors = np.empty((NF, max_dom_size, max_dom_size), dtype=float)
    for k, (m, _, _) in enumerate(all_matrix):
        m = np.asarray(m, dtype=float)
        fill = float(m.max()) + 1.0 / scale
        block = np.full((max_dom_size, max_dom_size), fill, dtype=float)
        block[: m.shape[0], : m.shape[1]] = m
        cost_tensors[k] = block

    data = {
        "edge_index": edge_index,
        "var_embed": var_embed,
        "func_embed": func_embed,
        "msg_hidden": msg_hidden,
        "msgs": msgs,
        "msg_rv2f_idxes": msg_rv2f_idxes,
        "msg_cv2f_idxes": msg_cv2f_idxes,
        "msg_f2rv_idxes": msg_f2rv_idxes,
        "msg_f2cv_idxes": msg_f2cv_idxes,
        "msg_v2f_idxes": msg_v2f_idxes,
        "msg_f2v_idxes": msg_f2v_idxes,
        "msg_trg_idxes": msg_trg_idxes,
        "msg_src_idxes": msg_src_idxes,
        "msg_f2v_per_v_idxes": msg_f2v_per_v_idxes,
        "embed_trg_idxes": embed_trg_idxes,
        "embed_src_idxes": embed_src_idxes,
        "v2f_scatter_idxes": v2f_scatter_idxes,
        "f2v_per_v_scatter_idxes": f2v_per_v_scatter_idxes,
        "f_batch": [0] * NF,
        "cost_tensors": cost_tensors,
        "rv_idxes": rv_idxes,
        "cv_idxes": cv_idxes,
        "NF": NF,
        "NV": NV,
    }
    return data, ordered_names, domain
