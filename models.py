import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dropout_edge
from typing import Optional


# ─────────────────────────────────────────────
# 1. LIGHT GAT ENCODER (Adaptive Self + Graph Gating)
# ─────────────────────────────────────────────
class LightGATEncoder(nn.Module):
    """
    2-layer GAT with Homophily-Aware Adaptive Self-Feature Gating and residual skip.

    Architecture:
        Self-branch:   z_self  = ELU(BN(Linear(in_dim → hid)))
        Graph-branch:  z_graph = Layer2_GAT(Layer1_GAT(x, edge_index))
        Adaptive Gate: beta    = Sigmoid(Linear(z_self)) * homophily_scale
        Output:        z_out   = ELU(z_self + beta * z_graph)
    """
    def __init__(self, in_dim: int, hid: int, dropout: float = 0.3, homophily: float = 0.7):
        super().__init__()
        self.hid = hid
        self.homophily = homophily
        # Scale factor based on train-graph homophily: dampens message passing on noisy graphs
        self.graph_scale = float(max(0.15, min(1.0, (homophily - 0.25) / 0.60)))

        # Direct patient self-feature projection
        self.self_proj = nn.Linear(in_dim, hid)
        self.bn_self   = nn.BatchNorm1d(hid)

        # Layer 1 GAT
        self.conv1   = GATConv(in_dim, hid, heads=2, concat=False)
        self.bn1     = nn.BatchNorm1d(hid)
        # Layer 2 GAT
        self.conv2   = GATConv(hid, hid, heads=2, concat=False)
        self.bn2     = nn.BatchNorm1d(hid)

        # Learnable gate controlling how much graph message passing is mixed into z_self
        self.graph_gate = nn.Linear(hid, hid)

        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Clean handling for zeroed-out modalities during ablation studies
        if torch.all(x == 0):
            return torch.zeros(x.size(0), self.hid, device=x.device, dtype=x.dtype)

        # Higher feature dropout on low-homophily (noisy) modalities to prevent p >> n overfitting
        eff_drop = self.dropout if self.homophily >= 0.55 else min(0.55, self.dropout + 0.20)
        if self.training and edge_index.numel() > 0:
            edge_index, _ = dropout_edge(edge_index, p=0.15)

        x_drop = F.dropout(x, p=eff_drop, training=self.training)
        z_self = F.elu(self.bn_self(self.self_proj(x_drop)))

        # ── Layer 1 + Layer 2 Graph Aggregation ─────────────────────────
        z1 = F.elu(self.bn1(self.conv1(x_drop, edge_index)) + z_self)
        z1_drop = F.dropout(z1, p=eff_drop, training=self.training)
        z2 = F.elu(self.bn2(self.conv2(z1_drop, edge_index)) + z1)

        # ── Adaptive Gating between Self-Features and Graph Context ─────
        beta = torch.sigmoid(self.graph_gate(z_self)) * self.graph_scale
        z_out = F.elu(z_self + beta * z2)

        return z_out


# ─────────────────────────────────────────────
# 2. ATTENTION FUSION (Homophily-Guided Anchor-Residual Fusion)
# ─────────────────────────────────────────────
class AttentionFusion(nn.Module):
    """
    Homophily-Guided Anchor-Residual Multi-Omics Fusion.

    1. Uses Train-Graph Homophily Prior (`prior_weights`) so modalities that overfit
       training labels in p >> n regimes (e.g. CNV/miRNA on OV) cannot deceive the
       attention mechanism.
    2. Uses Anchor-Residual connection from the highest-reliability modality (`anchor_idx`)
       plus a gated cross-omics synergy residual so 4-Omics integration strictly builds
       upon and improves over the strongest single modality without signal dilution.

    forward() returns:
      fused   [N, D]  — fused patient embeddings
      weights [N, M]  — per-patient attention weights (for explainability)
      entropy scalar  — penalty term passed to main.py
    """
    def __init__(self, dim: int, n_modality: int = 4, dropout: float = 0.2,
                 prior_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.dim = dim
        self.n_modality = n_modality
        hidden_attn = max(16, dim // 2)
        self.score = nn.Sequential(
            nn.Linear(dim, hidden_attn),
            nn.ELU(),
            nn.Linear(hidden_attn, 1, bias=False)
        )
        # Feature-wise noise gate per modality
        self.channel_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        # Cross-omics synergy module over concatenated weighted embeddings
        self.cross_omics = nn.Sequential(
            nn.Linear(dim * n_modality, dim),
            nn.BatchNorm1d(dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        # Residual synergy gate conditioned on anchor + multi-omics context
        self.synergy_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

        if prior_weights is None:
            prior_weights = torch.ones(n_modality, dtype=torch.float32) / float(n_modality)
        else:
            prior_weights = prior_weights.float() / prior_weights.sum()
        self.register_buffer("prior_weights", prior_weights)
        self.register_buffer("prior_logits", torch.log(prior_weights + 1e-8))
        self.anchor_idx = int(torch.argmax(prior_weights).item())

    def forward(self, z_list: list):
        z = torch.stack(z_list, dim=1)          # [N, M, D]
        active_mask = (z.abs().amax(dim=-1, keepdim=True) > 1e-7)  # [N, M, 1]

        # Filter noisy channels within each active modality
        z_clean = z * self.channel_gate(z) * active_mask.float()   # [N, M, D]

        # Combine dynamic patient attention scores with Train-Graph Homophily prior
        dyn_scores = torch.tanh(self.score(z_clean)) * 0.6         # [N, M, 1]
        prior_log  = self.prior_logits[:z.size(1)].view(1, -1, 1)  # [1, M, 1]
        raw_scores = dyn_scores + prior_log

        if not torch.all(active_mask):
            raw_scores = raw_scores.masked_fill(~active_mask, -1e9)

        w = F.softmax(raw_scores, dim=1)        # [N, M, 1]

        entropy = -(w * torch.log(w + 1e-8)).sum(dim=1).mean()   # scalar

        z_weighted = z_clean * w                # [N, M, D]
        z_sum = z_weighted.sum(dim=1)           # [N, D]

        # Anchor-Residual Multi-Omics Synergy when all modalities are active
        if torch.all(active_mask) and z.size(1) == self.n_modality:
            z_anchor = z_clean[:, self.anchor_idx, :]                      # [N, D]
            z_cat = z_weighted.reshape(z.size(0), -1)                      # [N, M*D]
            synergy = self.cross_omics(z_cat) + (z_sum - z_anchor)         # [N, D]
            gate = self.synergy_gate(torch.cat([z_anchor, z_sum], dim=-1)) # [N, D]
            fused = F.elu(z_anchor + 0.35 * gate * synergy)                # [N, D]
        else:
            # Single-omic ablation: weight by modality's prior reliability
            rel = (self.prior_weights[:z.size(1)].view(1, -1, 1) * active_mask.float()).sum(dim=1)
            scale = 0.55 + 0.40 * rel
            fused = F.elu(scale * z_sum)

        return fused, w.squeeze(-1), entropy    # [N,D], [N,M], scalar


# ─────────────────────────────────────────────
# 3. FOCAL LOSS
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss with class weights.

    Standard cross-entropy treats all samples equally.
    Focal Loss down-weights easy samples (high confidence) and focuses
    training on hard/misclassified samples (minority classes).

    gamma=2.0: (1 - pt)^gamma acts as a modulating factor.
      - Easy sample (pt=0.9): factor = (0.1)^2 = 0.01 → almost ignored.
      - Hard sample  (pt=0.1): factor = (0.9)^2 = 0.81 → strongly penalized.

    alpha = class_weights (balanced, clamped [0.5, 5.0] in main.py):
      Combined with gamma, gives extra weight to minority classes that are
      also hard to classify (double benefit for rare cancer subtypes).
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ─────────────────────────────────────────────
# 4. PANCANCER MODEL
# ─────────────────────────────────────────────
class PanCancerModel(nn.Module):
    """
    4 × LightGATEncoder → AttentionFusion → Linear classifier.

    Each encoder independently processes one omic modality on its own
    patient similarity graph (built from that omic's feature space).
    AttentionFusion learns to weight the 4 embeddings per patient.

    forward() returns (logits, entropy) so main.py computes:
        loss = FocalLoss(logits, y) - entropy_weight * entropy
    """
    @staticmethod
    def _compute_train_homophily(d, chance: float) -> float:
        ei = d.edge_index
        tr = d.train_mask
        if ei.numel() == 0:
            return chance
        tr_edges = tr[ei[0]] & tr[ei[1]]
        if int(tr_edges.sum().item()) == 0:
            return chance
        return float((d.y[ei[0][tr_edges]] == d.y[ei[1][tr_edges]]).float().mean().item())

    def __init__(self, args, data: dict):
        super().__init__()
        hid  = args.nhid
        drop = args.dropout_ratio

        self.has_cnv = 'CNV' in data
        n_cls = int(args.num_classes) if getattr(args, 'num_classes', 0) > 0 else int(data['mRNA'].y.max().item()) + 1
        chance = 1.0 / max(2, n_cls)

        omics_keys = ['mRNA', 'miRNA', 'Methy'] + (['CNV'] if self.has_cnv else [])
        homophilies = [self._compute_train_homophily(data[k], chance) for k in omics_keys]
        qualities   = [max(0.03, h - chance) ** 2 for h in homophilies]
        prior_w     = torch.tensor(qualities, dtype=torch.float32)
        prior_w     = prior_w / prior_w.sum()

        self.enc_mrna  = LightGATEncoder(data['mRNA'].num_node_features,  hid, drop, homophilies[0])
        self.enc_mirna = LightGATEncoder(data['miRNA'].num_node_features, hid, drop, homophilies[1])
        self.enc_methy = LightGATEncoder(data['Methy'].num_node_features, hid, drop, homophilies[2])
        if self.has_cnv:
            self.enc_cnv = LightGATEncoder(data['CNV'].num_node_features, hid, drop, homophilies[3])

        n_modality = 4 if self.has_cnv else 3
        self.fusion = AttentionFusion(hid, n_modality=n_modality, dropout=drop, prior_weights=prior_w)

        self.classifier = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(hid, n_cls)
        )

    def forward(self, d1, d2, d3, d4=None):
        z1 = self.enc_mrna (d1.x, d1.edge_index)
        z2 = self.enc_mirna(d2.x, d2.edge_index)
        z3 = self.enc_methy(d3.x, d3.edge_index)

        if self.has_cnv and d4 is not None:
            z4 = self.enc_cnv(d4.x, d4.edge_index)
            fused, attn_weights, entropy = self.fusion([z1, z2, z3, z4])
        else:
            fused, attn_weights, entropy = self.fusion([z1, z2, z3])

        # Store for post-training explainability
        self.last_attn = attn_weights.detach()

        return self.classifier(fused), entropy

    def get_attention_weights(self) -> torch.Tensor:
        """Returns [N, 4] attention weights — use to plot per-omics importance."""
        return self.last_attn
