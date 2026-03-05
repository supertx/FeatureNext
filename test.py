import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.convnext import convnext_tiny
from torch.utils.data import DataLoader

from dataset import FusionDataset, get_default_transfrom
from model import get_mbf, MultiModalQualityNet


def process_state_dict(state_dict):
    processed_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone"):
            new_key = k.replace("backbone.", "")
            processed_dict[new_key] = v
    return processed_dict


class MultiModalExtractor(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.face_net = get_mbf(fp16=False, num_features=feature_dim)
        self.face_net.load_state_dict(torch.load("model/data/model.pt", weights_only=True))

        self.palm_net = convnext_tiny(pretrained=False)
        self.palm_net.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(768, eps=1e-6, elementwise_affine=True),
            nn.Linear(768, feature_dim),
        )
        palm_state_dict = torch.load(
            "checkpoints/palm_finetune/palm_finetuned_final.pt",
            weights_only=True,
        )
        self.palm_net.load_state_dict(process_state_dict(palm_state_dict), strict=False)

        self.quality_net = MultiModalQualityNet()

    def load_quality_weight(self, quality_weight_path):
        self.quality_net.load_state_dict(torch.load(quality_weight_path, weights_only=True))

    @torch.no_grad()
    def forward(self, face_img, palm_img):
        face_feat = F.normalize(self.face_net(face_img), p=2, dim=1)
        palm_feat = F.normalize(self.palm_net(palm_img), p=2, dim=1)
        q_f, q_p = self.quality_net(face_img, palm_img)
        return face_feat, palm_feat, q_f, q_p


@torch.no_grad()
def compute_acc_and_roc(similarity, labels):
    """
    计算全量 1:1 验证的 ROC 和 ACC。
    提取相似度矩阵的上三角部分（排除对角线），实现严谨的交叉比对。
    """
    n = labels.numel()
    # 提取上三角索引，offset=1 完美避开自身匹配
    r, c = torch.triu_indices(n, n, offset=1)

    scores = similarity[r, c].float()
    is_same = (labels[r] == labels[c]).long()

    pos_scores = scores[is_same == 1]
    neg_scores = scores[is_same == 0]

    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return {
            "acc": 0.0, "best_threshold": 0.0, "auc": 0.0,
            "fpr": torch.tensor([0.0, 1.0]), "tpr": torch.tensor([0.0, 1.0]),
            "pos_mean": 0.0, "neg_mean": 0.0,
        }

    # 按相似度降序排列
    order = torch.argsort(scores, descending=True)
    y_true = is_same[order]
    y_score = scores[order]

    pos_num = pos_scores.numel()
    neg_num = neg_scores.numel()
    total_num = pos_num + neg_num

    tp = torch.cumsum(y_true, dim=0).float()
    fp = torch.cumsum(1 - y_true, dim=0).float()

    tpr = tp / pos_num
    fpr = fp / neg_num
    tpr = torch.cat([torch.tensor([0.0]), tpr])
    fpr = torch.cat([torch.tensor([0.0]), fpr])
    auc = torch.trapz(tpr, fpr).item()

    # 计算最佳 ACC
    tp_ext = torch.cat([torch.tensor([0.0]), tp])
    fp_ext = torch.cat([torch.tensor([0.0]), fp])
    acc_curve = (tp_ext + (neg_num - fp_ext)) / total_num
    best_idx = int(torch.argmax(acc_curve).item())
    best_acc = float(acc_curve[best_idx].item())

    if best_idx == 0:
        best_threshold = float(y_score[0].item()) + 1e-6
    elif best_idx == total_num:
        best_threshold = float(y_score[-1].item()) - 1e-6
    else:
        # 取最佳分割点的中间值更为严谨
        best_threshold = float((y_score[best_idx - 1] + y_score[best_idx]) / 2.0)

    return {
        "acc": best_acc,
        "best_threshold": best_threshold,
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "pos_mean": float(pos_scores.mean().item()),
        "neg_mean": float(neg_scores.mean().item()),
    }


def plot_roc_curves(face_result, palm_result, weighted_result, save_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, skip ROC curve plotting.")
        return

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(7, 6))
    plt.plot(
        face_result["fpr"].numpy(),
        face_result["tpr"].numpy(),
        linewidth=2,
        label=f"Face ROC (AUC={face_result['auc']:.4f})",
    )
    plt.plot(
        palm_result["fpr"].numpy(),
        palm_result["tpr"].numpy(),
        linewidth=2,
        label=f"Palm ROC (AUC={palm_result['auc']:.4f})",
    )
    plt.plot(
        weighted_result["fpr"].numpy(),
        weighted_result["tpr"].numpy(),
        linewidth=2,
        label=f"Weighted ROC (AUC={weighted_result['auc']:.4f})",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="gray", label="Random")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multimodal Biometrics ROC Curve")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"ROC curve saved to: {save_path}")


@torch.no_grad()
def evaluate_topk(model, dataloader, device, topk=(1, 5)):
    model.eval()
    all_face_feat = []
    all_palm_feat = []
    all_qf = []
    all_qp = []
    all_labels = []

    print("Extracting features (moving directly to CPU to prevent OOM)...")
    for face_img, palm_img, labels in dataloader:
        face_img = face_img.to(device, non_blocking=True)
        palm_img = palm_img.to(device, non_blocking=True)

        face_feat, palm_feat, q_f, q_p = model(face_img, palm_img)
        
        all_face_feat.append(face_feat.cpu())
        all_palm_feat.append(palm_feat.cpu())
        all_qf.append(q_f.cpu())
        all_qp.append(q_p.cpu())
        all_labels.append(labels.cpu())

    print("Computing similarity matrix on CPU...")
    face_feat = torch.cat(all_face_feat, dim=0)
    palm_feat = torch.cat(all_palm_feat, dim=0)
    q_f = torch.cat(all_qf, dim=0).view(-1)
    q_p = torch.cat(all_qp, dim=0).view(-1)
    labels = torch.cat(all_labels, dim=0)

    n = labels.size(0)

    face_sim = torch.mm(face_feat, face_feat.t())
    palm_sim = torch.mm(palm_feat, palm_feat.t())
    
    # 使用对称权重进行质量融合
    q_f_sym = (q_f[:, None] + q_f[None, :]) / 2.0
    q_p_sym = (q_p[:, None] + q_p[None, :]) / 2.0
    sim_weighted = q_f_sym * face_sim + q_p_sym * palm_sim

    # 为 Top-K 检索准备带有遮罩的相似度矩阵
    diag_mask = torch.eye(n, dtype=torch.bool)
    face_sim_topk = face_sim.clone()
    palm_sim_topk = palm_sim.clone()
    sim_weighted_topk = sim_weighted.clone()

    face_sim_topk.masked_fill_(diag_mask, -float('inf'))
    palm_sim_topk.masked_fill_(diag_mask, -float('inf'))
    sim_weighted_topk.masked_fill_(diag_mask, -float('inf'))

    print("Calculating Top-K Accuracies...")
    max_k = max(topk)

    _, indices_face = torch.topk(face_sim_topk, k=max_k, dim=1, largest=True)
    pred_face = labels[indices_face]
    
    _, indices_palm = torch.topk(palm_sim_topk, k=max_k, dim=1, largest=True)
    pred_palm = labels[indices_palm]
    
    _, indices_weighted = torch.topk(sim_weighted_topk, k=max_k, dim=1, largest=True)
    pred_weighted = labels[indices_weighted]

    metrics = {}
    for k in topk:
        metrics[f"top{k}_face"] = (pred_face[:, :k] == labels[:, None]).any(dim=1).float().mean().item()
        metrics[f"top{k}_palm"] = (pred_palm[:, :k] == labels[:, None]).any(dim=1).float().mean().item()
        metrics[f"top{k}_weighted"] = (pred_weighted[:, :k] == labels[:, None]).any(dim=1).float().mean().item()

    print("Calculating Binary ACC and ROC Metrics...")
    binary_face = compute_acc_and_roc(face_sim, labels)
    binary_palm = compute_acc_and_roc(palm_sim, labels)
    binary_weighted = compute_acc_and_roc(sim_weighted, labels)

    metrics.update({
        "acc_face": binary_face["acc"], "acc_palm": binary_palm["acc"], "acc_weighted": binary_weighted["acc"],
        "auc_face": binary_face["auc"], "auc_palm": binary_palm["auc"], "auc_weighted": binary_weighted["auc"],
        "best_threshold_face": binary_face["best_threshold"],
        "best_threshold_palm": binary_palm["best_threshold"],
        "best_threshold_weighted": binary_weighted["best_threshold"],
        "pos_mean_face": binary_face["pos_mean"], "neg_mean_face": binary_face["neg_mean"],
        "pos_mean_palm": binary_palm["pos_mean"], "neg_mean_palm": binary_palm["neg_mean"],
        "pos_mean_weighted": binary_weighted["pos_mean"], "neg_mean_weighted": binary_weighted["neg_mean"],
        "roc_face": binary_face, "roc_palm": binary_palm, "roc_weighted": binary_weighted,
    })

    return metrics, n


def main():
    parser = argparse.ArgumentParser(description="Evaluate fusion test-set accuracy")
    parser.add_argument("--quality-ckpt", type=str, default="checkpoints/fusion/quality_net_v3_epoch_39.pth")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--roc-save-path", type=str, default="roc_curve.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FusionDataset(
        palm_data_dir="/data/tx/palm_data",
        face_data_dir="/data/tx/IJB",
        transform=get_default_transfrom(train=False),
        train=False,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True,
    )

    model = MultiModalExtractor(feature_dim=512).to(device)
    model.load_quality_weight(args.quality_ckpt)

    metrics, sample_num = evaluate_topk(model, dataloader, device, topk=(1, 5))

    print(f"\n--- Evaluation Results (Test samples: {sample_num}) ---")
    print(f"Top-1 accuracy (face):      {metrics['top1_face'] * 100:.2f}%")
    print(f"Top-1 accuracy (palm):      {metrics['top1_palm'] * 100:.2f}%")
    print(f"Top-1 accuracy (weighted):  {metrics['top1_weighted'] * 100:.2f}%\n")
    
    print(f"Binary ACC (face):          {metrics['acc_face'] * 100:.2f}%")
    print(f"Binary ACC (palm):          {metrics['acc_palm'] * 100:.2f}%")
    print(f"Binary ACC (weighted):      {metrics['acc_weighted'] * 100:.2f}%\n")
    
    print(f"ROC AUC (face):             {metrics['auc_face']:.6f}")
    print(f"ROC AUC (palm):             {metrics['auc_palm']:.6f}")
    print(f"ROC AUC (weighted):         {metrics['auc_weighted']:.6f}\n")
    
    print(f"Best threshold (face):      {metrics['best_threshold_face']:.6f}")
    print(f"Best threshold (palm):      {metrics['best_threshold_palm']:.6f}")
    print(f"Best threshold (weighted):  {metrics['best_threshold_weighted']:.6f}")

    plot_roc_curves(metrics["roc_face"],  metrics["roc_palm"], metrics["roc_weighted"], args.roc_save_path)


if __name__ == "__main__":
    main()