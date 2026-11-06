"""
Biểu đồ nghiên cứu loại bỏ Omics (Ablation Study)
Tính toán và so sánh hiệu năng THỰC TẾ giữa Đơn Omics (mRNA, miRNA, Methy, CNV)
và Đa Omics (All 4-Omics) dựa trên đồ thị và checkpoint thực của mô hình.

Chạy: python run_ablation_study.py
Output:
  - Result/pic/ablation_study_all5.svg and .png
  - Result/pic/ablation_study_{CANCER}.svg and .png
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
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

CANCERS    = ['BRCA', 'COAD', 'GBM', 'LGG', 'OV']
MODALITIES = ['mRNA', 'miRNA', 'Methy', 'CNV', '4-Omics']
COLORS     = ['#E07B6A', '#6BBFCF', '#5DBFA8', '#8B9DC3', '#E65100']
SEEDS      = [777, 42, 1234, 2024, 999]


def evaluate_feature_ablation(model, data, modality, test_mask):
    """
    Đánh giá mô hình thực tế khi loại bỏ hoặc cô lập các tầng omics (Zero-out Feature Ablation).
    """
    d_mrna  = data['mRNA'].clone()
    d_mirna = data['miRNA'].clone()
    d_methy = data['Methy'].clone()
    d_cnv   = data['CNV'].clone() if 'CNV' in data else None

    # Zero-out other modalities to isolate single-omic contribution
    if modality == 'mRNA':
        d_mirna.x = torch.zeros_like(d_mirna.x)
        d_methy.x = torch.zeros_like(d_methy.x)
        if d_cnv is not None:
            d_cnv.x = torch.zeros_like(d_cnv.x)
    elif modality == 'miRNA':
        d_mrna.x  = torch.zeros_like(d_mrna.x)
        d_methy.x = torch.zeros_like(d_methy.x)
        if d_cnv is not None:
            d_cnv.x = torch.zeros_like(d_cnv.x)
    elif modality == 'Methy':
        d_mrna.x  = torch.zeros_like(d_mrna.x)
        d_mirna.x = torch.zeros_like(d_mirna.x)
        if d_cnv is not None:
            d_cnv.x = torch.zeros_like(d_cnv.x)
    elif modality == 'CNV':
        if d_cnv is None:
            return 0.0, 0.0, 0.0
        d_mrna.x  = torch.zeros_like(d_mrna.x)
        d_mirna.x = torch.zeros_like(d_mirna.x)
        d_methy.x = torch.zeros_like(d_methy.x)
    elif modality == '4-Omics':
        pass  # Keep all original features untouched

    with torch.no_grad():
        logits, _ = model(d_mrna, d_mirna, d_methy, d_cnv)
        preds = logits.argmax(dim=1)

    y_true = data['mRNA'].y[test_mask].cpu().numpy()
    y_pred = preds[test_mask].cpu().numpy()

    acc  = accuracy_score(y_true, y_pred)
    w_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    m_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return acc, w_f1, m_f1


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


def compute_real_ablation_for_cancer(cancer):
    graph_path = os.path.join(DATA_DIR, f"{cancer}_graph.pt")
    if not os.path.exists(graph_path):
        print(f"  [SKIP] Graph {graph_path} not found.")
        return None

    data = rescue_isolated_nodes(torch.load(graph_path, map_location="cpu", weights_only=False))
    test_mask = data['mRNA'].test_mask

    # Collect results across available checkpoint seeds
    mod_metrics = {m: {'acc': [], 'weighted_f1': [], 'macro_f1': []} for m in MODALITIES}
    loaded_seeds = 0

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

        for m in MODALITIES:
            acc, w_f1, m_f1 = evaluate_feature_ablation(model, data, m, test_mask)
            mod_metrics[m]['acc'].append(acc)
            mod_metrics[m]['weighted_f1'].append(w_f1)
            mod_metrics[m]['macro_f1'].append(m_f1)

        loaded_seeds += 1

    if loaded_seeds == 0:
        print(f"  [SKIP] No checkpoints found for {cancer} in {CKPT_DIR}")
        return None

    # Calculate real mean and std across evaluated seeds
    res = {}
    for m in MODALITIES:
        res[m] = {
            'acc':         (np.mean(mod_metrics[m]['acc']),         np.std(mod_metrics[m]['acc'])),
            'weighted_f1': (np.mean(mod_metrics[m]['weighted_f1']), np.std(mod_metrics[m]['weighted_f1'])),
            'macro_f1':    (np.mean(mod_metrics[m]['macro_f1']),    np.std(mod_metrics[m]['macro_f1'])),
        }
    print(f"  → {cancer}: Successfully evaluated real ablation over {loaded_seeds} seeds.")
    return res


def plot_single_cancer_ablation(cancer, data):
    metrics = ['acc', 'weighted_f1', 'macro_f1']
    metric_labels = ['Accuracy', 'Weighted-F1', 'Macro-F1']

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(metric_labels))
    width = 0.15

    for i, mod in enumerate(MODALITIES):
        means = [data[mod][m][0] for m in metrics]
        stds  = [data[mod][m][1] for m in metrics]
        pos   = x - (len(MODALITIES) - 1) * width / 2 + i * width

        bars = ax.bar(pos, means, width, yerr=stds, label=mod, color=COLORS[i],
                      capsize=4, alpha=0.9 if mod == '4-Omics' else 0.75,
                      edgecolor='black' if mod == '4-Omics' else 'none',
                      linewidth=1.2 if mod == '4-Omics' else 0)

        if mod == '4-Omics':
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.035,
                        f"{val:.3f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#D84315')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Omics Ablation Study — {cancer}', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    out_svg = os.path.join(OUT_DIR, f"ablation_study_{cancer}.svg")
    out_png = os.path.join(OUT_DIR, f"ablation_study_{cancer}.png")
    plt.savefig(out_svg, format='svg', bbox_inches='tight')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_svg}")


def plot_all5_ablation(ablation_dict):
    cancers = list(ablation_dict.keys())
    n = len(cancers)

    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5.0), sharey=True)
    if n == 1:
        axes = [axes]

    metrics = ['acc', 'weighted_f1', 'macro_f1']
    metric_labels = ['ACC', 'Weighted-F1', 'Macro-F1']
    width = 0.15
    x = np.arange(len(metric_labels))

    for ax, cancer in zip(axes, cancers):
        data = ablation_dict[cancer]
        for i, mod in enumerate(MODALITIES):
            means = [data[mod][m][0] for m in metrics]
            stds  = [data[mod][m][1] for m in metrics]
            pos   = x - (len(MODALITIES) - 1) * width / 2 + i * width

            bars = ax.bar(pos, means, width, yerr=stds, label=mod if ax == axes[0] else "",
                          color=COLORS[i], capsize=3,
                          alpha=0.95 if mod == '4-Omics' else 0.75,
                          edgecolor='black' if mod == '4-Omics' else 'none',
                          linewidth=1.1 if mod == '4-Omics' else 0)

            if mod == '4-Omics':
                for bar, val in zip(bars, means):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                            f"{val:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#D84315')

        ax.set_title(cancer, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].set_ylabel('Performance Metric', fontsize=12)
    fig.suptitle('Pan-Cancer Omics Ablation Study (Single-Omics vs. 4-Omics Integration)', fontsize=14, y=1.02, fontweight='bold')
    fig.legend(MODALITIES, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=5, fontsize=10.5, frameon=False)
    plt.tight_layout()

    out_svg = os.path.join(OUT_DIR, "ablation_study_all5.svg")
    out_png = os.path.join(OUT_DIR, "ablation_study_all5.png")
    plt.savefig(out_svg, format='svg', bbox_inches='tight')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved Pan-Cancer Ablation Figure:")
    print(f"   SVG: {out_svg}")
    print(f"   PNG: {out_png}")


def main():
    print("=" * 65)
    print("  RUNNING REAL OMICS ABLATION STUDY (NO HARDCODED DATA)")
    print("=" * 65)

    real_ablation_results = {}
    for c in CANCERS:
        res = compute_real_ablation_for_cancer(c)
        if res is not None:
            real_ablation_results[c] = res
            plot_single_cancer_ablation(c, res)

    if not real_ablation_results:
        print("\n[ERROR] Không có checkpoint hoặc đồ thị thực tế nào để tính toán Ablation.")
        print("Vui lòng chạy 'prepare_graph.py' và 'main.py' trước để tạo dữ liệu và mô hình thực:")
        print("  python scripts/prepare_graph.py --cancer BRCA")
        print("  python main.py --cancer BRCA --epochs 150 --nhid 64")
        return

    plot_all5_ablation(real_ablation_results)


if __name__ == "__main__":
    main()
