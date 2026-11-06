"""
Trực quan hóa nâng cao chuẩn Publication (100% dữ liệu thực):
1. Confusion Matrix (Ma trận nhầm lẫn chuẩn hóa) từ dự đoán thực tế của mô hình.
2. t-SNE Embedding (Phân cụm biểu diễn bệnh nhân 2D từ vector ẩn thực tế của AttentionFusion).

Chạy: python run_viz_advanced.py
Output:
  - Result/pic/confusion_matrix_all5.svg and .png
  - Result/pic/confusion_matrix_{CANCER}.svg and .png
  - Result/pic/tsne_all5.svg and .png
  - Result/pic/tsne_{CANCER}.svg and .png
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from models import PanCancerModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, 'Result', 'pic')
os.makedirs(OUT_DIR, exist_ok=True)

ckpt_candidates = [
    os.environ.get('CKPT_DIR', ''),
    os.path.join(BASE_DIR, 'data', 'Cancer-classification-dataset', 'checkpoints'),
    os.path.join(BASE_DIR, 'checkpoints'),
    r'E:\Cancer-classification-dataset\checkpoints'
]
CKPT_DIR = next((d for d in ckpt_candidates if d and os.path.exists(d)), ckpt_candidates[1])

data_candidates = [
    os.path.join(BASE_DIR, 'data', 'Cancer-classification-dataset'),
    os.path.join(BASE_DIR, 'data'),
    BASE_DIR
]
DATA_DIR = next((d for d in data_candidates if os.path.exists(d)), data_candidates[0])

# Subtype names based on official Cancer Subtype labels.xlsx mapping
SUBTYPE_NAMES = {
    'BRCA': ['LumA', 'HER2', 'LumB', 'Normal', 'Basal'],
    'COAD': ['GI.CIN', 'GI.MSI', 'GI.GS', 'GI.HM-SNV'],
    'GBM':  ['Classical', 'Proneural', 'Mesenchymal', 'G-CIMP', 'Neural'],
    'LGG':  ['Classical', 'Mesenchymal', 'Neural', 'Proneural'],
    'OV':   ['Immunoreactive', 'Differentiated', 'Proliferative', 'Mesenchymal'],
}

SEEDS = [777, 42, 1234, 2024, 999]


def rescue_isolated_nodes(data):
    for k, d in data.items():
        N = d.x.size(0)
        deg = torch.zeros(N, dtype=torch.long)
        if d.edge_index.numel() > 0:
            deg.scatter_add_(0, d.edge_index[0], torch.ones(d.edge_index.size(1), dtype=torch.long))
        isolated = (deg == 0).nonzero(as_tuple=False).view(-1)
        if len(isolated) > 0:
            x_norm = torch.nn.functional.normalize(d.x, p=2, dim=1)
            sim = torch.mm(x_norm, x_norm.t())
            sim.fill_diagonal_(-1.0)
            _, topk_idx = torch.topk(sim, k=min(2, N - 1), dim=1)
            new_edges = []
            for node_i in isolated.tolist():
                for nb in topk_idx[node_i].tolist():
                    new_edges.append([node_i, nb])
                    new_edges.append([nb, node_i])
            if new_edges:
                add_ei = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
                combined = torch.cat([d.edge_index, add_ei], dim=1)
                d.edge_index = torch.unique(combined, dim=1)
    return data


def get_real_model_predictions_and_embeddings(cancer):
    """
    Load all available seed checkpoints and compute 5-seed ensemble test predictions
    and multi-seed patient embeddings. Strictly uses actual model weights and real graph data.
    """
    graph_path = os.path.join(DATA_DIR, f"{cancer}_graph.pt")
    if not os.path.exists(graph_path):
        print(f"  [SKIP] Graph {graph_path} not found.")
        return None

    data = rescue_isolated_nodes(torch.load(graph_path, map_location="cpu", weights_only=False))
    d1, d2, d3 = data['mRNA'], data['miRNA'], data['Methy']
    d4 = data.get('CNV', None)
    test_mask = data['mRNA'].test_mask

    probs_list = []
    emb_list   = []

    for s in SEEDS:
        ckpt_path = os.path.join(CKPT_DIR, f"{cancer}_seed{s}_best.pth")
        if not os.path.exists(ckpt_path):
            continue

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if 'model_state_dict' not in ck or 'args' not in ck:
            continue

        class MockArgs:
            def __init__(self, d):
                self.__dict__.update(d)

        args = MockArgs(ck['args'])
        model = PanCancerModel(args, data)
        model.load_state_dict(ck['model_state_dict'])
        model.eval()

        with torch.no_grad():
            z1 = model.enc_mrna(d1.x, d1.edge_index)
            z2 = model.enc_mirna(d2.x, d2.edge_index)
            z3 = model.enc_methy(d3.x, d3.edge_index)
            if model.has_cnv and d4 is not None:
                z4 = model.enc_cnv(d4.x, d4.edge_index)
                fused, _, _ = model.fusion([z1, z2, z3, z4])
            else:
                fused, _, _ = model.fusion([z1, z2, z3])

            logits = model.classifier(fused)
            probs  = torch.softmax(logits, dim=1)
            emb_norm = torch.nn.functional.normalize(fused, p=2, dim=1)

        probs_list.append(probs.cpu().numpy())
        emb_list.append(emb_norm.cpu().numpy())

    if not probs_list:
        print(f"  [SKIP] No checkpoint found for {cancer} in {CKPT_DIR}")
        return None

    mean_probs = np.mean(probs_list, axis=0)
    preds      = np.argmax(mean_probs, axis=1)
    emb_all    = np.concatenate(emb_list, axis=1) if len(emb_list) > 1 else emb_list[0]

    y_all  = d1.y.cpu().numpy()
    y_test = y_all[test_mask.cpu().numpy()]
    p_test = preds[test_mask.cpu().numpy()]

    return y_test, p_test, emb_all, y_all


def plot_confusion_matrices(real_data_dict):
    cancers = list(real_data_dict.keys())
    fig, axes = plt.subplots(1, len(cancers), figsize=(4.6 * len(cancers), 4.2))
    if len(cancers) == 1:
        axes = [axes]

    for ax, cancer in zip(axes, cancers):
        y_test, p_test, _, _ = real_data_dict[cancer]
        n_cls = len(np.unique(y_test))
        names = SUBTYPE_NAMES.get(cancer, [f"Class {i}" for i in range(n_cls)])[:n_cls]

        cm = confusion_matrix(y_test, p_test)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])

        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.0%})"

        sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues', cbar=False,
                    xticklabels=names, yticklabels=names, ax=ax,
                    annot_kws={"size": 8.5})

        ax.set_title(f"{cancer} Confusion Matrix", fontsize=11, fontweight='bold')
        ax.set_ylabel("True Subtype", fontsize=10)
        ax.set_xlabel("Predicted Subtype", fontsize=10)
        ax.tick_params(axis='x', rotation=30, labelsize=9)
        ax.tick_params(axis='y', rotation=0, labelsize=9)

        # Also save individually
        fig_single, ax_s = plt.subplots(figsize=(5.5, 4.6))
        sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues', cbar=True,
                    xticklabels=names, yticklabels=names, ax=ax_s, annot_kws={"size": 10})
        ax_s.set_title(f"{cancer} — Subtype Confusion Matrix", fontsize=12, fontweight='bold')
        ax_s.set_ylabel("True Subtype", fontsize=11)
        ax_s.set_xlabel("Predicted Subtype", fontsize=11)
        plt.tight_layout()
        single_svg = os.path.join(OUT_DIR, f"confusion_matrix_{cancer}.svg")
        single_png = os.path.join(OUT_DIR, f"confusion_matrix_{cancer}.png")
        fig_single.savefig(single_svg, format='svg', bbox_inches='tight')
        fig_single.savefig(single_png, dpi=150, bbox_inches='tight')
        plt.close(fig_single)

    plt.tight_layout()
    all_svg = os.path.join(OUT_DIR, "confusion_matrix_all5.svg")
    all_png = os.path.join(OUT_DIR, "confusion_matrix_all5.png")
    fig.savefig(all_svg, format='svg', bbox_inches='tight')
    fig.savefig(all_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved Confusion Matrix Figures:")
    print(f"   SVG: {all_svg}")
    print(f"   PNG: {all_png}")


def plot_tsne_embeddings(real_data_dict):
    cancers = list(real_data_dict.keys())
    fig, axes = plt.subplots(1, len(cancers), figsize=(4.6 * len(cancers), 4.2))
    if len(cancers) == 1:
        axes = [axes]

    palette = sns.color_palette("tab10", 6)

    for ax, cancer in zip(axes, cancers):
        _, _, emb_all, y_all = real_data_dict[cancer]
        names = SUBTYPE_NAMES.get(cancer, [f"Class {i}" for i in range(len(np.unique(y_all)))])

        # Compute 2D t-SNE projection strictly from real embeddings
        tsne = TSNE(n_components=2, perplexity=min(30, len(emb_all) - 1), random_state=42)
        emb_2d = tsne.fit_transform(emb_all)

        for c_idx in np.unique(y_all):
            mask = (y_all == c_idx)
            lbl = names[c_idx] if c_idx < len(names) else f"Class {c_idx}"
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                       label=lbl, color=palette[c_idx % len(palette)], s=35,
                       alpha=0.85, edgecolors='white', linewidth=0.6)

        ax.set_title(f"{cancer} (t-SNE)", fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
        ax.legend(fontsize=8, frameon=True, facecolor='white', framealpha=0.85, loc='best')

        # Also save individually
        fig_single, ax_s = plt.subplots(figsize=(5.5, 4.6))
        for c_idx in np.unique(y_all):
            mask = (y_all == c_idx)
            lbl = names[c_idx] if c_idx < len(names) else f"Class {c_idx}"
            ax_s.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                         label=lbl, color=palette[c_idx % len(palette)], s=50,
                         alpha=0.85, edgecolors='white', linewidth=0.7)
        ax_s.set_title(f"{cancer} — Multi-Omics Patient Embeddings (t-SNE)", fontsize=12, fontweight='bold')
        ax_s.set_xticks([])
        ax_s.set_yticks([])
        ax_s.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
        ax_s.legend(fontsize=9, frameon=True, facecolor='white', framealpha=0.9, loc='best')
        plt.tight_layout()
        single_svg = os.path.join(OUT_DIR, f"tsne_{cancer}.svg")
        single_png = os.path.join(OUT_DIR, f"tsne_{cancer}.png")
        fig_single.savefig(single_svg, format='svg', bbox_inches='tight')
        fig_single.savefig(single_png, dpi=150, bbox_inches='tight')
        plt.close(fig_single)

    plt.tight_layout()
    all_svg = os.path.join(OUT_DIR, "tsne_all5.svg")
    all_png = os.path.join(OUT_DIR, "tsne_all5.png")
    fig.savefig(all_svg, format='svg', bbox_inches='tight')
    fig.savefig(all_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved t-SNE Embedding Figures:")
    print(f"   SVG: {all_svg}")
    print(f"   PNG: {all_png}")


def main():
    print("=" * 65)
    print("  RUNNING REAL SCIENTIFIC VISUALIZATIONS (NO HARDCODED DATA)")
    print("  1. Confusion Matrices")
    print("  2. t-SNE Embeddings")
    print("=" * 65)

    cancers = ['BRCA', 'COAD', 'GBM', 'LGG', 'OV']
    real_data_dict = {}

    for c in cancers:
        res = get_real_model_predictions_and_embeddings(c)
        if res is not None:
            real_data_dict[c] = res

    if not real_data_dict:
        print("\n[ERROR] Không tìm thấy dữ liệu đồ thị hoặc checkpoint thực tế nào.")
        print("Vui lòng chạy 'prepare_graph.py' và 'main.py' trước:")
        print("  python scripts/prepare_graph.py --cancer BRCA")
        print("  python main.py --cancer BRCA --epochs 150 --nhid 64")
        return

    plot_confusion_matrices(real_data_dict)
    plot_tsne_embeddings(real_data_dict)


if __name__ == "__main__":
    main()
