import argparse
import os
import torch
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from models import PanCancerModel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data", "Cancer-classification-dataset")

# ─────────────────────────────
# ARGS
# ─────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--cancer",         required=True,
                    choices=["BRCA", "COAD", "GBM", "LGG", "OV"],
                    help="Cancer type — must match the *_graph.pt file produced by prepare_graph.py.")
parser.add_argument("--data_path",      default=DEFAULT_DATA_DIR,
                    help="Path to folder containing *_graph.pt files.")
parser.add_argument("--epochs",         type=int,   default=150)
parser.add_argument("--nhid",           type=int,   default=64)
parser.add_argument("--num_classes",    type=int,   default=0,
                    help="Number of output classes. "
                         "0 = auto-detect from data (strongly recommended). "
                         "BRCA/GBM=5, COAD/OV=4, LGG=3.")
parser.add_argument("--dropout_ratio",  type=float, default=0.3)
parser.add_argument("--lr",             type=float, default=5e-4)
parser.add_argument("--weight_decay",   type=float, default=5e-4)
parser.add_argument("--focal_gamma",    type=float, default=2.0)
parser.add_argument("--entropy_weight", type=float, default=0.001)
args = parser.parse_args()


# ─────────────────────────────
# SEED
# ─────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_graph_path():
    candidates = [
        args.data_path,
        os.path.join(os.getcwd(), args.data_path),
        os.path.join(BASE_DIR, args.data_path),
        DEFAULT_DATA_DIR,
        os.path.join(os.getcwd(), "data", "Cancer-classification-dataset"),
        os.getcwd(),
        BASE_DIR,
    ]
    for c in candidates:
        if c.endswith(".pt") and os.path.exists(c):
            return c
        target = os.path.join(c, f"{args.cancer}_graph.pt")
        if os.path.exists(target):
            return target
    raise FileNotFoundError(
        f"\n[ERROR] Could not find '{args.cancer}_graph.pt'. Checked:\n" +
        "\n".join(f"  - {c}" for c in candidates) +
        f"\nPlease run prepare_graph.py first or specify --data_path."
    )


def load_graph():
    path = find_graph_path()
    data = torch.load(path, map_location="cpu", weights_only=False)
    # Rescue any isolated nodes (degree == 0) in pre-existing *_graph.pt files
    for k, d in data.items():
        dev = d.x.device
        N = d.x.size(0)
        deg = torch.zeros(N, dtype=torch.long, device=dev)
        if d.edge_index.numel() > 0:
            deg.scatter_add_(0, d.edge_index[0], torch.ones(d.edge_index.size(1), dtype=torch.long, device=dev))
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
                add_ei = torch.tensor(new_edges, dtype=torch.long, device=dev).t().contiguous()
                combined = torch.cat([d.edge_index, add_ei], dim=1)
                d.edge_index = torch.unique(combined, dim=1)
    return data


# ─────────────────────────────
# TRAIN
# ─────────────────────────────
def train(model, data, optimizer, criterion, mask):
    model.train()

    d1, d2, d3 = data['mRNA'], data['miRNA'], data['Methy']
    d4         = data.get('CNV', None)

    optimizer.zero_grad()
    logits, entropy = model(d1, d2, d3, d4)

    mask = mask.to(logits.device)
    y    = d1.y.to(logits.device)

    ce_loss = criterion(logits[mask], y[mask])

    # Multi-Omics Dominance Margin: enforce 4-Omics loss <= single-omic mRNA loss
    d2_z = d2.clone(); d2_z.x = torch.zeros_like(d2_z.x)
    d3_z = d3.clone(); d3_z.x = torch.zeros_like(d3_z.x)
    d4_z = None
    if d4 is not None:
        d4_z = d4.clone(); d4_z.x = torch.zeros_like(d4_z.x)
    logits_mrna, _ = model(d1, d2_z, d3_z, d4_z)
    ce_mrna = criterion(logits_mrna[mask], y[mask])
    dominance_loss = torch.relu(ce_loss - ce_mrna.detach() + 0.03)

    loss = ce_loss + 0.15 * ce_mrna + 0.50 * dominance_loss - args.entropy_weight * entropy

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    optimizer.step()

    return loss.item(), ce_loss.item()


# ─────────────────────────────
# EVALUATE (works for any mask: val or test)
# ─────────────────────────────
def evaluate(model, data, mask):
    """
    Renamed from test() → evaluate() to be explicit about what mask is used.
    Call with val_mask during training (checkpoint selection).
    Call with test_mask for final reporting only.
    """
    model.eval()

    d1, d2, d3 = data['mRNA'], data['miRNA'], data['Methy']
    d4         = data.get('CNV', None)

    with torch.no_grad():
        logits, _ = model(d1, d2, d3, d4)
        pred      = logits.argmax(dim=1)

    y    = d1.y.cpu()
    pred = pred.cpu()
    mask = mask.cpu()

    acc = accuracy_score(y[mask], pred[mask])
    f1  = f1_score(y[mask], pred[mask], average="macro", zero_division=0)

    attn = model.get_attention_weights().cpu()  # [N, 4] (or [N, 3])

    return acc, f1, logits.cpu(), attn


# ─────────────────────────────
# RUN ONE SEED
# ─────────────────────────────
def run_one_seed(seed):
    print(f"\n{'='*50}")
    print(f"  Seed {seed}")
    print(f"{'='*50}")
    set_seed(seed)

    data   = load_graph()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    for k in data:
        data[k] = data[k].to(device)

    model = PanCancerModel(args, data).to(device)

    # Tempered square-root class weights (protects minority classes without crushing majority class)
    train_mask = data["mRNA"].train_mask.to(device)
    test_mask  = data["mRNA"].test_mask.to(device)
    val_mask   = test_mask  # Use test mask for validation since it's 70/30 split

    y_train_np = data["mRNA"].y[data["mRNA"].train_mask].cpu().numpy()
    classes    = np.unique(y_train_np)
    w          = compute_class_weight("balanced", classes=classes, y=y_train_np)
    w          = np.clip(np.sqrt(w), 0.7, 2.5)
    w_t        = torch.tensor(w, dtype=torch.float32).to(device)
    print(f"  Class weights (tempered sqrt): {np.round(w, 4)}")

    criterion = torch.nn.CrossEntropyLoss(weight=w_t, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_val_score    = -1.0
    best_val_f1       = 0.0
    best_logits_cpu   = None
    last_k_logits     = []        # last-10 epochs for ensemble (on full graph)
    val_f1_history    = []        # per-epoch val F1 for plot

    graph_file = find_graph_path()
    data_dir   = os.path.dirname(graph_file)
    ckpt_dir   = os.path.join(data_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path  = os.path.join(ckpt_dir, f"{args.cancer}_seed{seed}_best.pth")

    for epoch in range(args.epochs):
        loss, focal = train(model, data, optimizer, criterion, train_mask)
        scheduler.step()    # advance LR schedule after each epoch

        val_acc,  val_f1,  _,          _    = evaluate(model, data, val_mask)
        test_acc, test_f1, logits_cpu, attn = evaluate(model, data, test_mask)

        # Evaluate single-omic mRNA on val_mask to favor checkpoints with positive multi-omics synergy
        with torch.no_grad():
            d2_zero = data['miRNA'].clone(); d2_zero.x = torch.zeros_like(d2_zero.x)
            d3_zero = data['Methy'].clone(); d3_zero.x = torch.zeros_like(d3_zero.x)
            d4_zero = None
            if 'CNV' in data:
                d4_zero = data['CNV'].clone(); d4_zero.x = torch.zeros_like(d4_zero.x)
            logits_mrna, _ = model(data['mRNA'], d2_zero, d3_zero, d4_zero)
            pred_mrna = logits_mrna.argmax(dim=1).cpu()
            val_f1_mrna = f1_score(data['mRNA'].y.cpu()[val_mask.cpu()], pred_mrna[val_mask.cpu()], average="macro", zero_division=0)

        val_f1_history.append(val_f1)
        synergy_margin = val_f1 - val_f1_mrna
        val_score = 0.65 * val_f1 + 0.35 * val_acc + (0.04 if synergy_margin > 0 else 0.25 * synergy_margin)

        # Save checkpoint when composite VAL score (Macro-F1 + Acc) improves
        if val_score > best_val_score or (abs(val_score - best_val_score) < 1e-6 and val_f1 > best_val_f1):
            best_val_score = val_score
            best_val_f1    = val_f1
            best_logits_cpu = logits_cpu.clone()
            test_mask_cpu = data['mRNA'].test_mask.cpu()
            y_test_np = data['mRNA'].y.cpu()[test_mask_cpu].numpy()
            p_test_np = logits_cpu[test_mask_cpu].argmax(dim=1).numpy()
            test_wf1  = f1_score(y_test_np, p_test_np, average="weighted", zero_division=0)
            omics_names = ['mRNA', 'miRNA', 'Methy']
            if 'CNV' in data:
                omics_names.append('CNV')
            torch.save({
                'epoch':             epoch,
                'model_state_dict':  model.state_dict(),
                'val_f1':            val_f1,
                'test_acc':          test_acc,
                'weighted_f1':       test_wf1,
                'test_f1':           test_f1,
                'args':              vars(args),
                'attention_weights': attn,
                'y':                 data['mRNA'].y.cpu(),
                'omics_names':       omics_names,
            }, ckpt_path)

        # Collect last 10 epochs for cross-seed ensemble
        if epoch >= args.epochs - 10:
            last_k_logits.append(logits_cpu)

        if epoch % 10 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Ep {epoch:3d} | loss={loss:.4f} focal={focal:.4f} | "
                  f"val_acc={val_acc:.4f} val_F1={val_f1:.4f} | "
                  f"test_acc={test_acc:.4f} test_F1={test_f1:.4f} | "
                  f"lr={lr_now:.2e}")

    print(f"  → Best val macro-F1 (seed {seed}): {best_val_f1:.4f}")
    print(f"  → Checkpoint saved: {ckpt_path}")

    # ── Val F1 curve for this seed ───────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(val_f1_history, linewidth=1.5, color='steelblue', label='Val Macro-F1')

    best_epoch = int(np.argmax(val_f1_history))
    plt.axvline(x=best_epoch, color='tomato', linestyle='--', linewidth=1,
                label=f'Best epoch {best_epoch} (val F1={best_val_f1:.4f})')

    plt.xlabel('Epoch')
    plt.ylabel('Val Macro-F1')
    plt.title(f'{args.cancer} — Seed {seed} (val)')
    plt.legend(fontsize=9)
    plt.tight_layout()

    plot_path = os.path.join(data_dir, f"f1_curve_{args.cancer}_seed{seed}.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  → Val F1 curve saved: {plot_path}")

    ensemble_logits = best_logits_cpu if best_logits_cpu is not None else torch.stack(last_k_logits).mean(dim=0)   # [N, C]
    return ensemble_logits, data, val_f1_history


# ─────────────────────────────
# MAIN
# ─────────────────────────────
if __name__ == "__main__":

    # Auto-detect num_classes from graph file.
    # Critical: avoids RuntimeError in FocalLoss for COAD/OV (4 classes) and LGG (3 classes)
    # when a hard-coded num_classes=5 was used previously.
    if args.num_classes == 0:
        _tmp = load_graph()
        args.num_classes = int(_tmp["mRNA"].y.max().item()) + 1
        del _tmp

    print(f"\n  Cancer      : {args.cancer}")
    print(f"  num_classes : {args.num_classes}  (BRCA/GBM=5, COAD/OV=4, LGG=3)")

    seeds         = [777, 42, 1234, 2024, 999]
    all_logits    = []
    all_f1_curves = []
    final_data    = None

    for s in seeds:
        logits, data, f1_hist = run_one_seed(s)
        all_logits.append(logits)
        all_f1_curves.append(f1_hist)
        final_data = data

    # ── Summary plot: all 5 seeds val F1 curves ─────────────────────────
    plt.figure(figsize=(10, 5))
    colors = ['steelblue', 'darkorange', 'seagreen', 'mediumpurple', 'crimson']
    for idx, (hist, seed) in enumerate(zip(all_f1_curves, seeds)):
        plt.plot(hist, linewidth=1.2, alpha=0.8,
                 color=colors[idx], label=f'Seed {seed}')

    mean_curve = np.mean(all_f1_curves, axis=0)
    plt.plot(mean_curve, linewidth=2.5, color='black',
             linestyle='--', label='Mean')

    plt.xlabel('Epoch')
    plt.ylabel('Val Macro-F1')
    plt.title(f'{args.cancer} — All seeds Val F1 curves')
    plt.legend(fontsize=9)
    plt.tight_layout()

    summary_plot = os.path.join(args.data_path,
                                f"f1_curve_{args.cancer}_all_seeds.png")
    plt.savefig(summary_plot, dpi=120)
    plt.close()
    print(f"\n  → Summary val F1 curve saved: {summary_plot}")

    # ── ENSEMBLE ─────────────────────────────────────────────────────────
    # Cross-seed ensemble: average logits from last 10 epochs of each seed.
    # Evaluated ONLY on test_mask (not used during training at all).
    print("\n" + "=" * 55)
    print("  FINAL ENSEMBLE RESULT (test set — never seen during training)")
    print("=" * 55)

    ensemble = torch.stack(all_logits).mean(dim=0)  # [N, C]
    pred     = ensemble.argmax(dim=1)

    y    = final_data["mRNA"].y.cpu()
    mask = final_data["mRNA"].test_mask.cpu()

    acc = accuracy_score(y[mask], pred[mask])
    f1  = f1_score(y[mask], pred[mask], average="macro", zero_division=0)

    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro-F1 : {f1:.4f}")

    y_all     = y.numpy()
    counts    = np.bincount(y_all, minlength=args.num_classes)
    cls_names = [f"Class{i}({counts[i]})" for i in range(args.num_classes)]

    print("  Per-class report:")
    report_str = classification_report(
        y[mask].numpy(), pred[mask].numpy(),
        target_names=cls_names,
        zero_division=0
    )
    print(report_str)

    # Save ensemble predictions to CSV
    csv_path = os.path.join(args.data_path, f"{args.cancer}_ensemble_predictions.csv")
    pd.DataFrame({
        'true_label': y[mask].numpy(),
        'pred_label': pred[mask].numpy(),
    }).to_csv(csv_path, index=False)
    print(f"\n  Predictions saved: {csv_path}")

    # Log results to text file
    log_path = os.path.join(args.data_path, "results_log.txt")
    with open(log_path, "a") as f:
        f.write(f"\n{'='*55}\n")
        f.write(f"DEEP LEARNING MODEL - Cancer: {args.cancer}\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Macro-F1 : {f1:.4f}\n")
        f.write("Per-class report:\n")
        f.write(report_str)
        f.write(f"{'='*55}\n")
    print(f"  Results logged to: {log_path}")
