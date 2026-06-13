"""Vendored DABP neural model (``DABP-main/alg/model.py``).

The ``Attention`` and ``AttentiveBP`` modules are kept faithful to the original
Deep Attentive Belief Propagation implementation. Local changes:

- imports come from the vendored ``.constant`` / ``.scatter`` modules
- device and dtype are configurable (``configure``) so the model can run in
  float32 on Apple MPS, which does not support float64
- ``preprocess_single`` replaces the batched ``preprocess`` for a single graph,
  avoiding any dependency on PyTorch Geometric batching internals
- ``step`` additionally returns the per-variable belief tensor so the engine can
  read off assignments; ``step_once``/``detach_state`` drive it iteratively
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from .constant import VAR_ID, V2F_ID, F2V_ID
from .scatter import scatter_add, scatter_softmax, scatter_mean


class Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.query_proj = nn.Linear(hidden, hidden, bias=False)
        self.key_proj = nn.Linear(hidden, hidden, bias=False)
        self.score_proj = nn.Linear(2 * hidden, 1)

    def forward(self, query, key):
        query = self.query_proj(query)
        key = self.key_proj(key)
        return torch.sigmoid(self.score_proj(torch.cat([query, key], dim=1)))


class AttentiveBP(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=1, prefix_dim=4, msg_dim=15, max_color=100):
        super().__init__()
        self.prefix_dim = prefix_dim
        self.vn_color_embed = nn.Embedding(max_color, in_channels - prefix_dim)
        self.gru1 = nn.GRUCell(msg_dim, in_channels - prefix_dim)
        self.gru2 = nn.GRUCell(msg_dim, in_channels - prefix_dim)
        self.conv1 = GATConv(in_channels, 8, heads=4, concat=True)
        self.conv2 = GATConv(32, 8, 4, concat=True)
        self.conv3 = GATConv(32, 8, 4, concat=True)
        self.conv4 = GATConv(32, out_channels, 4, concat=False)
        self.attentions = nn.ModuleList()
        for _ in range(num_heads):
            self.attentions.append(Attention(out_channels))
        # device/dtype the preprocessed tensors live on (set via configure)
        self.device = torch.device("cpu")
        self.dtype = torch.float64

    def configure(self, device: torch.device, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype

    def step(self, *param):
        edge_index = param[0]
        var_embed = param[1]
        func_embed = param[2]
        msg_hidden = param[3]
        msgs = param[4]
        msg_rv2f_idxes = param[5]
        msg_cv2f_idxes = param[6]
        msg_f2rv_idxes = param[7]
        msg_f2cv_idxes = param[8]
        msg_v2f_idxes = param[9]
        msg_f2v_idxes = param[10]
        msg_trg_idxes = param[11]
        msg_src_idxes = param[12]
        msg_f2v_per_v_idxes = param[13]
        embed_trg_idxes = param[14]
        embed_src_idxes = param[15]
        v2f_scatter_idxes = param[16]
        f2v_per_v_scatter_idxes = param[17]
        f_batch = param[18]
        cost_tensors = param[19]
        rv_idxes = param[20]
        cv_idxes = param[21]
        first_iteration = param[22]

        # msg f -> v
        msg_rv2f = msgs[msg_rv2f_idxes]
        msg_cv2f = msgs[msg_cv2f_idxes]
        msg_f2rv = torch.min(cost_tensors + msg_cv2f.unsqueeze(1), dim=2)[0]
        msg_f2cv = torch.min(cost_tensors + msg_rv2f.unsqueeze(2), dim=1)[0]
        msgs[msg_f2rv_idxes] = msg_f2rv
        msgs[msg_f2cv_idxes] = msg_f2cv

        _var_embed = []
        for ve in var_embed:
            _var_embed.append(self.vn_color_embed(ve))

        for i in range(len(msg_hidden)):
            v2f_hidden, f2v_hidden = msg_hidden[i]
            v2f_msgs = msgs[msg_v2f_idxes[i]]
            f2v_msgs = msgs[msg_f2v_idxes[i]]
            v2f_hidden = self.gru1(v2f_msgs.detach(), v2f_hidden)
            f2v_hidden = self.gru2(f2v_msgs.detach(), f2v_hidden)
            msg_hidden[i] = [v2f_hidden, f2v_hidden]

        x = []
        for i in range(len(msg_hidden)):
            ve = _var_embed[i]
            pref = torch.tensor(VAR_ID, dtype=self.dtype, device=self.device)
            pref = pref.repeat(ve.shape[0], 1)
            ve = torch.cat([pref, ve], dim=1)

            fe = func_embed[i]

            v2f_hidden = msg_hidden[i][0]
            pref = torch.tensor(V2F_ID, dtype=self.dtype, device=self.device)
            pref = pref.repeat(v2f_hidden.shape[0], 1)
            v2f_hidden = torch.cat([pref, v2f_hidden], dim=1)

            f2v_hidden = msg_hidden[i][1]
            pref = torch.tensor(F2V_ID, dtype=self.dtype, device=self.device)
            pref = pref.repeat(f2v_hidden.shape[0], 1)
            f2v_hidden = torch.cat([pref, f2v_hidden], dim=1)

            x.append(torch.cat([ve, fe, v2f_hidden, f2v_hidden], dim=0))
        x = torch.cat(x, dim=0)

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv4(x, edge_index)
        x = F.leaky_relu(x)

        msg_src = msgs[msg_src_idxes]
        msg_trg = msgs[msg_trg_idxes]

        embed_src = x[embed_src_idxes]
        embed_trg = x[embed_trg_idxes]
        var_degree = torch.unique(v2f_scatter_idxes, return_counts=True)[-1]  # Degree(x) - 1
        embed_trg_repeated = torch.repeat_interleave(embed_trg, var_degree, dim=0)
        assert embed_src.shape == embed_trg_repeated.shape

        attention_scores = []
        attention_trg_scores = []
        for m in self.attentions:
            attention_scores.append(m(embed_trg_repeated, embed_src))
            attention_trg_scores.append(m(embed_trg, embed_trg))
        attention_scores = torch.cat(attention_scores, dim=1)
        attention_trg_scores = torch.cat(attention_trg_scores, dim=1)
        attention_weight = scatter_softmax(attention_scores, v2f_scatter_idxes, dim=0)
        weighted_msg = msg_src.unsqueeze(2) * attention_weight.unsqueeze(1)
        weighted_msg = scatter_add(weighted_msg, v2f_scatter_idxes, dim=0)
        weighted_msg = weighted_msg.mean(-1)
        weighted_msg = weighted_msg * var_degree.unsqueeze(1)

        attention_scores = scatter_mean(attention_scores, v2f_scatter_idxes, dim=0)
        damped_weights = torch.softmax(
            torch.cat([attention_scores.unsqueeze(1), attention_trg_scores.unsqueeze(1)], dim=1), dim=1
        )
        v2f_msgs = weighted_msg.unsqueeze(2) * damped_weights[:, 0, :].unsqueeze(1) + msg_trg.unsqueeze(
            2
        ) * damped_weights[:, 1, :].unsqueeze(1)
        v2f_msgs = v2f_msgs.mean(-1)
        v2f_msgs = v2f_msgs - v2f_msgs.min(dim=1, keepdim=True)[0]
        msgs[msg_trg_idxes] = v2f_msgs

        f2v_per_v_msgs = msgs[msg_f2v_per_v_idxes]
        belief = scatter_add(f2v_per_v_msgs, f2v_per_v_scatter_idxes, dim=0)
        dist = torch.softmax(-belief, dim=1)
        entropy = dist * torch.log2(dist + 1e-6)
        entropy = -entropy.sum(dim=1)
        entropy = entropy.mean()
        dist_rv = dist[rv_idxes]
        dist_cv = dist[cv_idxes]
        if not first_iteration:
            loss = torch.bmm(torch.bmm(dist_rv.unsqueeze(1), cost_tensors), dist_cv.unsqueeze(2)).squeeze()
            loss = scatter_add(loss, f_batch)
            loss = loss.mean()
            loss = loss + 0.1 * entropy
        else:
            loss = 0
        val_idx_rv = dist_rv.argmax(dim=1, keepdim=False).tolist()
        val_idx_cv = dist_cv.argmax(dim=1, keepdim=False).tolist()
        cost = []
        for i in range(cost_tensors.shape[0]):
            cost.append(cost_tensors[i, val_idx_rv[i], val_idx_cv[i]].item())
        cost = torch.tensor(cost, dtype=self.dtype, device=self.device)
        cost = scatter_add(cost, f_batch, dim=0)
        beliefs = belief.detach().to("cpu").numpy()
        return loss, cost.mean().item(), beliefs

    # ------------------------------------------------------------------
    # single-graph driving helpers (replace the original batched pipeline)
    # ------------------------------------------------------------------
    def preprocess_single(self, d: dict) -> None:
        """load one graph's tensors (built by build.py) onto device/dtype.

        Resets message and hidden state to their initial (zero) values, mirroring
        a DABP restart. Index tensors that the model indexes per-graph
        (``msg_v2f_idxes``/``msg_f2v_idxes``) are wrapped in a one-element list.
        """
        dev, dt = self.device, self.dtype

        def L(arr):
            return torch.tensor(arr, dtype=torch.long, device=dev)

        self.edge_index = L(d["edge_index"])
        self.var_embed = [L(d["var_embed"])]
        self.func_embed = [torch.tensor(d["func_embed"], dtype=dt, device=dev)]
        self.msg_hidden = [
            [
                torch.tensor(d["msg_hidden"][0], dtype=dt, device=dev),
                torch.tensor(d["msg_hidden"][1], dtype=dt, device=dev),
            ]
        ]
        self.msgs = torch.tensor(d["msgs"], dtype=dt, device=dev)
        self.msg_rv2f_idxes = L(d["msg_rv2f_idxes"])
        self.msg_cv2f_idxes = L(d["msg_cv2f_idxes"])
        self.msg_f2rv_idxes = L(d["msg_f2rv_idxes"])
        self.msg_f2cv_idxes = L(d["msg_f2cv_idxes"])
        self.msg_v2f_idxes = [L(d["msg_v2f_idxes"])]
        self.msg_f2v_idxes = [L(d["msg_f2v_idxes"])]
        self.msg_trg_idxes = L(d["msg_trg_idxes"])
        self.msg_src_idxes = L(d["msg_src_idxes"])
        self.msg_f2v_per_v_idxes = L(d["msg_f2v_per_v_idxes"])
        self.embed_trg_idxes = L(d["embed_trg_idxes"])
        self.embed_src_idxes = L(d["embed_src_idxes"])
        self.v2f_scatter_idxes = L(d["v2f_scatter_idxes"])
        self.f2v_per_v_scatter_idxes = L(d["f2v_per_v_scatter_idxes"])
        self.f_batch = torch.zeros(d["NF"], dtype=torch.long, device=dev)
        self.cost_tensors = torch.tensor(d["cost_tensors"], dtype=dt, device=dev)
        self.rv_idxes = L(d["rv_idxes"])
        self.cv_idxes = L(d["cv_idxes"])

    def detach_state(self) -> None:
        """cut the autograd graph at a phase boundary (keep current values)."""
        self.msgs = self.msgs.detach()
        for i in range(len(self.msg_hidden)):
            for j in range(2):
                self.msg_hidden[i][j] = self.msg_hidden[i][j].detach()

    def step_once(self, first_iteration: bool):
        """run a single BP iteration; returns (loss, cost_mean, beliefs)."""
        return self.step(
            self.edge_index,
            self.var_embed,
            self.func_embed,
            self.msg_hidden,
            self.msgs,
            self.msg_rv2f_idxes,
            self.msg_cv2f_idxes,
            self.msg_f2rv_idxes,
            self.msg_f2cv_idxes,
            self.msg_v2f_idxes,
            self.msg_f2v_idxes,
            self.msg_trg_idxes,
            self.msg_src_idxes,
            self.msg_f2v_per_v_idxes,
            self.embed_trg_idxes,
            self.embed_src_idxes,
            self.v2f_scatter_idxes,
            self.f2v_per_v_scatter_idxes,
            self.f_batch,
            self.cost_tensors,
            self.rv_idxes,
            self.cv_idxes,
            first_iteration,
        )
