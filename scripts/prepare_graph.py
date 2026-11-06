import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif

from torch_geometric.data import Data


# ─────────────────────────────────────────────
# 1. ARGPARSE
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "Cancer-classification-dataset")

parser = argparse.ArgumentParser()
parser.add_argument("--cancer",      required=True,
                    choices=["BRCA", "COAD", "GBM", "LGG", "OV"],
                    help="Cancer type. Supported: BRCA (N=671, 5-class), "
                         "COAD (N=260, 4-class), GBM (N=244, 5-class), "
                         "LGG (N=247, 3-class), OV (N=284, 4-class).")
parser.add_argument("--seed",        type=int,   default=777)
parser.add_argument("--train_ratio", type=float, default=0.7,
                    help="Train split ratio. Default 0.7 (70/30 split).")
parser.add_argument("--k",           type=int,   default=7,
                    help="Mutual KNN neighbors. k=7 works well for N≥200.")
parser.add_argument("--topk_mrna",   type=int,   default=1000)
parser.add_argument("--topk_mirna",  type=int,   default=None,
                    help="Top-k miRNA features by MAD (fit on train only). "
                         "Default=None → keep all features. "
                         "Recommended: ~100 for LGG/OV/GBM (high all-zero rates). "
                         "BRCA miRNA (366 features, 0%% zero) does not need reduction.")
parser.add_argument("--topk_methy",  type=int,   default=1000)
parser.add_argument("--topk_cnv",    type=int,   default=1000,
                    help="Top-k CNV features by ANOVA (fit on train only). Default=1000.")
parser.add_argument("--data_dir",    type=str,   default=DEFAULT_DATA_DIR,
                    help="Root directory of raw TCGA data folder.")
# FIX #2: PCA before KNN graph construction (optional, medium priority).
# PCA removes noise dimensions → cleaner graph topology → GATConv gets better neighborhood.
# Node features (Data.x) remain full-dimensional; PCA only affects WHICH patients are neighbors.
# Recommended: --pca_graph 64. Leave as None to skip PCA (backward-compatible).
parser.add_argument("--pca_graph",   type=int,   default=64,
                    help="If set, apply PCA to this many components before building KNN graph. "
                         "Cleaner topology, does not change node feature dimensions. "
                         "Recommended: 64. Set to 0 to disable PCA.")
parser.add_argument("--min_class_samples", type=int, default=5,
                    help="Filter out ultra-rare outlier classes with fewer than this many total samples "
                         "(e.g. COAD Class 3 with only 4 samples outside the 3 official GI subtypes). "
                         "Set to 0 to keep all classes.")
args = parser.parse_args()

cancer_type = args.cancer

# Locate the dataset folder containing the cancer subfolder cross-platform (Windows / Linux / Colab)
candidates = [
    args.data_dir,
    os.path.join(os.getcwd(), args.data_dir),
    os.path.join(PROJECT_ROOT, args.data_dir),
    DEFAULT_DATA_DIR,
    os.path.join(os.getcwd(), "data", "Cancer-classification-dataset"),
    os.path.join(os.getcwd(), cancer_type),
    r"c:\Users\Admin\Desktop\Machine learning\My-MOFNet-Research-main (Phuoc)\Classification"
]

repo_path = None
base_path = None
for cand in candidates:
    if cand and os.path.exists(cand):
        # 1. cand has cancer_type as a subfolder
        if os.path.exists(os.path.join(cand, cancer_type, f"{cancer_type}_mRNA.csv")):
            repo_path = cand
            base_path = os.path.join(cand, cancer_type)
            break
        # 2. cand itself is the cancer folder
        elif os.path.exists(os.path.join(cand, f"{cancer_type}_mRNA.csv")):
            repo_path = os.path.dirname(cand)
            base_path = cand
            break

if repo_path is None or base_path is None:
    raise FileNotFoundError(
        f"\n[ERROR] Could not find '{cancer_type}_mRNA.csv' for {cancer_type}.\n"
        f"Checked paths:\n" + "\n".join(f"  - {c}" for c in candidates if c) + "\n"
        f"Please provide the path using --data_dir, e.g.:\n"
        f"  python scripts/prepare_graph.py --cancer {cancer_type} --data_dir /path/to/Cancer-classification-dataset\n"
    )


# ─────────────────────────────────────────────
# 2. SEED
# ─────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(args.seed)


# ─────────────────────────────────────────────
# 3. FEATURE SELECTION (ANOVA, fit on train)
# ─────────────────────────────────────────────
def select_top_anova(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, k: int) -> np.ndarray:
    """
    Select k features with highest ANOVA F-value, computed strictly on TRAIN ONLY.
    Matches MOFNet paper methodology.
    """
    X_train = X[train_idx]
    y_train = y[train_idx]
    
    # Calculate ANOVA F-values
    f_values, _ = f_classif(X_train, y_train)
    
    # Handle NaN f_values (if variance is exactly 0 in all classes)
    f_values = np.nan_to_num(f_values, nan=0.0)
    
    top_idx = np.argsort(f_values)[-k:]
    return X[:, top_idx]


# ─────────────────────────────────────────────
# 4. BUILD KNN GRAPH
# ─────────────────────────────────────────────
def build_knn_graph(X: np.ndarray, k: int, train_idx: np.ndarray,
                    pca_components: int = None) -> torch.Tensor:
    """
    Mutual KNN graph from cosine similarity.

    Mutual KNN: keep edge(i,j) only if j ∈ topk(i) AND i ∈ topk(j).
      - Removes noisy one-directional edges.
      - GATConv learns actual neighborhood structure.

    If pca_components is set:
      - PCA is fit on train_idx ONLY (no leakage).
      - Transform all patients for graph construction.
      - Node features (Data.x) are NOT reduced — only topology is affected.
    """
    X_graph = X.copy()

    if pca_components is not None and pca_components > 0:
        n_comp = min(pca_components, len(train_idx) - 1, X_graph.shape[1])
        if n_comp < X_graph.shape[1]:
            pca = PCA(n_components=n_comp, random_state=42)
            pca.fit(X_graph[train_idx])          # fit on train only
            X_graph = pca.transform(X_graph)
            explained = pca.explained_variance_ratio_.sum()
            print(f"    PCA: {X.shape[1]} → {X_graph.shape[1]} dims "
                  f"(explained variance: {explained:.2%})")

    sim = cosine_similarity(X_graph)
    np.fill_diagonal(sim, -1)               # prevent self-loops

    sim_t       = torch.tensor(sim, dtype=torch.float32)
    _, topk_idx = torch.topk(sim_t, k=k, dim=1)    # [N, k]

    N   = X_graph.shape[0]
    adj = torch.zeros(N, N, dtype=torch.bool)
    row = torch.arange(N).unsqueeze(1).expand(-1, k)
    adj[row.reshape(-1), topk_idx.reshape(-1)] = True

    # Mutual: keep only edges present in both directions
    mutual = adj & adj.T

    # Rescue isolated nodes (degree == 0) by connecting them to their top-2 nearest neighbors
    deg = mutual.sum(dim=1)
    isolated = (deg == 0).nonzero(as_tuple=False).view(-1)
    if len(isolated) > 0:
        fallback_k = min(2, k)
        for node_i in isolated:
            nbs = topk_idx[node_i, :fallback_k]
            mutual[node_i, nbs] = True
            mutual[nbs, node_i] = True

    edge_index = mutual.nonzero(as_tuple=False).T    # [2, E]

    return edge_index


# ─────────────────────────────────────────────
# 5. LOAD DATA
# ─────────────────────────────────────────────
def load_omic(filename: str) -> pd.DataFrame:
    """
    Load omics CSV and transpose so rows = patients, columns = features.
    NaN handling is done downstream in preprocess() via train-median imputation
    for ALL omics (consistent, no leakage).
    """
    df = pd.read_csv(os.path.join(base_path, filename), index_col=0).T
    df.index = df.index.astype(str).str[:16]
    return df

print(f"\n[1/5] Loading omics data for {cancer_type}...")

df_mrna  = load_omic(f'{cancer_type}_mRNA.csv')
df_mirna = load_omic(f'{cancer_type}_miRNA.csv')
df_methy = load_omic(f'{cancer_type}_Methy.csv')

cnv_file = os.path.join(base_path, f'{cancer_type}_CNV.csv')
has_cnv = os.path.exists(cnv_file)
if has_cnv:
    df_cnv = load_omic(f'{cancer_type}_CNV.csv')
    common = sorted(
        set(df_mrna.index) & set(df_mirna.index) & set(df_methy.index) & set(df_cnv.index)
    )
else:
    df_cnv = None
    common = sorted(
        set(df_mrna.index) & set(df_mirna.index) & set(df_methy.index)
    )
print(f"  → Common patients: {len(common)}")


# ─────────────────────────────────────────────
# 6. LABEL & SPLIT (70 / 30)
# ─────────────────────────────────────────────
print("[2/5] Loading labels & splitting (70/30)...")

labels_df = pd.read_csv(os.path.join(base_path, f'{cancer_type}_label.csv'))
if labels_df.shape[1] > 10:
    labels_df = labels_df.T
labels_df.index = df_mrna.index

raw_y = LabelEncoder().fit_transform(labels_df.loc[common].values.ravel())
if args.min_class_samples > 0:
    counts = np.bincount(raw_y)
    valid_classes = np.where(counts >= args.min_class_samples)[0]
    if len(valid_classes) < len(counts):
        rare_classes = np.where(counts < args.min_class_samples)[0]
        for rc in rare_classes:
            print(f"  [Filter] Removing outlier Class {rc} with only {counts[rc]} total samples (< {args.min_class_samples}).")
        keep_mask = np.isin(raw_y, valid_classes)
        common = [p for p, keep in zip(common, keep_mask) if keep]
        raw_y = raw_y[keep_mask]
        print(f"  → Remaining patients after filtering ultra-rare classes: {len(common)}")

y = LabelEncoder().fit_transform(raw_y)
idx = np.arange(len(common))

# 70% Train, 30% Test (stratified)
train_idx, test_idx = train_test_split(
    idx, test_size=(1.0 - args.train_ratio), stratify=y, random_state=args.seed
)

train_mask = torch.zeros(len(common), dtype=torch.bool)
test_mask  = torch.zeros(len(common), dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx]   = True
y_tensor = torch.tensor(y, dtype=torch.long)

print(f"  → Train: {train_mask.sum().item()}, "
      f"Test: {test_mask.sum().item()}")
print(f"  → Class distribution (full): {np.bincount(y)}")


# ─────────────────────────────────────────────
# 7. PREPROCESS: IMPUTE → SCALE → FEATURE SELECT
# ─────────────────────────────────────────────
print("[3/5] Preprocessing features...")

def preprocess(df: pd.DataFrame, topk, name: str) -> np.ndarray:
    """
    Unified pipeline — no double normalization, no leakage:
      0. Train-median imputation for NaN (fit on train only)
      1. StandardScaler fit on TRAIN → transform train/val/test
      2. MAD feature selection fit on TRAIN (if topk specified)
    """
    X = df.loc[common].values.astype(np.float32)

    # Step 0: Impute NaN with train median (ALL omics, consistent)
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"  {name}: {nan_count} NaN values → train-median imputation")
        train_median = np.nanmedian(X[train_idx], axis=0)
        for j in range(X.shape[1]):
            nan_mask = np.isnan(X[:, j])
            if nan_mask.any():
                X[nan_mask, j] = train_median[j]

    # Step 1: Scale — fit on train only (zero-variance features → scale=1, stays 0)
    scaler = StandardScaler()
    X[train_idx] = scaler.fit_transform(X[train_idx])
    X[test_idx]  = scaler.transform(X[test_idx])

    # Step 2: Feature selection — ANOVA computed on train only
    if topk is not None and topk < X.shape[1]:
        X = select_top_anova(X, y, train_idx, topk)
        print(f"  {name}: {df.shape[1]} → {X.shape[1]} features (ANOVA top-{topk})")
    else:
        print(f"  {name}: {X.shape[1]} features (no reduction)")

    return X

X_mrna  = preprocess(df_mrna,  args.topk_mrna,  "mRNA")
X_mirna = preprocess(df_mirna, args.topk_mirna,  "miRNA")
X_methy = preprocess(df_methy, args.topk_methy,  "Methy")
if has_cnv:
    X_cnv = preprocess(df_cnv, args.topk_cnv, "CNV")


# ─────────────────────────────────────────────
# 8. BUILD GRAPHS
# ─────────────────────────────────────────────
pca_info = f", PCA→{args.pca_graph}d" if args.pca_graph else ""
print(f"[4/5] Building mutual KNN graphs (k={args.k}{pca_info})...")

def create_data(X: np.ndarray, name: str) -> Data:
    edge_index = build_knn_graph(X, args.k, train_idx, args.pca_graph)
    print(f"  {name}: nodes={X.shape[0]}, edges={edge_index.shape[1]}, feats={X.shape[1]}")
    return Data(
        x          = torch.tensor(X, dtype=torch.float32),
        edge_index = edge_index,
        y          = y_tensor,
        train_mask = train_mask,
        test_mask  = test_mask,
    )

dataset = {
    "mRNA":  create_data(X_mrna,  "mRNA"),
    "miRNA": create_data(X_mirna, "miRNA"),
    "Methy": create_data(X_methy, "Methy"),
}
if has_cnv:
    dataset["CNV"] = create_data(X_cnv, "CNV")


# ─────────────────────────────────────────────
# 9. SAVE
# ─────────────────────────────────────────────
print("[5/5] Saving...")
save_path = os.path.join(repo_path, f"{cancer_type}_graph.pt")
torch.save(dataset, save_path)
print(f"\n✅ DONE! Saved to: {save_path}")
print(f"   num_classes for {cancer_type}: {int(y_tensor.max().item()) + 1}")
print(f"   Split: {train_mask.sum().item()} train / "
      f"{test_mask.sum().item()} test")
print(f"   (pass num_classes to main.py via --num_classes, or leave 0 for auto-detect)")