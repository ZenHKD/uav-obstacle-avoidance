import os
import argparse
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as T

# Ensure src can be imported
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.model.dataloader import ODADataset, PyBulletDataset
from src.model.net_fast import ODANetFast

# =====================================================================
# 1. Custom Loss Functions
# =====================================================================

class SILogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic (SILog) Loss for depth map supervision.
    Focuses training on shape-agnostic, relative depth variations.
    """
    def __init__(self, alpha: float = 0.5, scale: float = 10.0, eps: float = 1e-4) -> None:
        super().__init__()
        self.alpha = alpha
        self.scale = scale
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # Clamp GT to max 10.0 to enforce sky/far distance supervision
        gt = torch.clamp(gt, max=10.0)
        mask = (gt > 0.0) & (gt <= 10.0)
        if not mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
        pred_valid = torch.clamp(pred[mask], min=self.eps)
        gt_valid = torch.clamp(gt[mask], min=self.eps)
        
        d = torch.log(pred_valid) - torch.log(gt_valid)
        loss = torch.mean(d ** 2) - self.alpha * (torch.mean(d) ** 2)
        return self.scale * torch.sqrt(torch.clamp(loss, min=1e-9))


class MetricDepthLoss(nn.Module):
    """
    Hybrid Loss combining SILogLoss for geometry structures and L1 Loss for absolute metric scale.
    """
    def __init__(self, alpha: float = 0.5, scale: float = 1.0, w_l1: float = 0.1) -> None:
        super().__init__()
        self.silog = SILogLoss(alpha=alpha, scale=scale)
        self.l1 = nn.L1Loss()
        self.w_l1 = w_l1

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        gt = torch.clamp(gt, max=10.0)
        mask = (gt > 0.0) & (gt <= 10.0)
        if not mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
        loss_silog = self.silog(pred, gt)
        loss_l1 = self.l1(pred[mask], gt[mask])
        return loss_silog + self.w_l1 * loss_l1


# =====================================================================
# 2. Helper Functions
# =====================================================================

def get_optimizer(model: nn.Module, lr: float, backbone_lr: float, is_frozen: bool) -> torch.optim.Optimizer:
    """
    Returns an AdamW optimizer configured for the given backbone freeze state.
    """
    if is_frozen:
        params = [p for name, p in model.named_parameters() if p.requires_grad and "backbone" not in name]
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    else:
        backbone_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
                
        param_groups = [
            {"params": other_params, "lr": lr},
            {"params": backbone_params, "lr": backbone_lr}
        ]
        return torch.optim.AdamW(param_groups, weight_decay=0.01)


def save_prediction_plot(epoch: int, val_idx: int, rgb: torch.Tensor, gt_depth: torch.Tensor, 
                         pred_depth: torch.Tensor, save_dir: str) -> None:
    """
    Saves a visualization comparison of RGB, GT depth, and predicted depth.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    rgb_img = rgb[0].permute(1, 2, 0).cpu().numpy()
    if rgb_img.max() > 1.0 or rgb_img.min() < 0.0:
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
        
    gt_d = gt_depth[0].squeeze(0).cpu().numpy() if gt_depth.ndim == 4 else gt_depth[0].cpu().numpy()
    pred_d = pred_depth[0, 0].cpu().numpy()
    
    fig = plt.figure(figsize=(12, 4))
    
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.imshow(rgb_img)
    ax0.set_title("Input RGB/DVS")
    ax0.axis("off")
    
    ax1 = fig.add_subplot(1, 3, 2)
    im1 = ax1.imshow(gt_d, cmap="inferno", vmin=0.0, vmax=10.0)
    ax1.set_title("GT Depth (Metric 0-10m)")
    ax1.axis("off")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    ax2 = fig.add_subplot(1, 3, 3)
    im2 = ax2.imshow(pred_d, cmap="inferno", vmin=0.0, vmax=10.0)
    ax2.set_title("Pred Depth (Metric 0-10m)")
    ax2.axis("off")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_val_{val_idx}.png"), dpi=150)
    plt.close()


# =====================================================================
# 3. Main Training Execution
# =====================================================================

def train_phase4(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("Accelerations enabled: TF32 matmul, TF32 cuDNN, cuDNN benchmark.")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    vis_dir = os.path.join(args.checkpoint_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    print("Loading ODA Dataset...")
    dataset = ODADataset(
        dataset_dir=args.dataset_dir,
        target_size=(224, 224),
        cache_data=args.cache_data
    )
    print(f"Total subsequences built: {len(dataset)}")
    
    unique_seqs = sorted(list(dataset.trial_overview.keys()))
    np.random.seed(42)
    np.random.shuffle(unique_seqs)
    
    val_size = int(len(unique_seqs) * 0.20)
    val_seqs = set(unique_seqs[:val_size])
    train_seqs = set(unique_seqs[val_size:])
    
    train_indices = [i for i, s in enumerate(dataset.samples) if s['seq_id'] in train_seqs]
    val_indices = [i for i, s in enumerate(dataset.samples) if s['seq_id'] in val_seqs]
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    print(f"Train split: {len(train_subset)} samples ({len(train_seqs)} sequences)")
    print(f"Val split: {len(val_subset)} samples ({len(val_seqs)} sequences)")
    
    import json
    val_seqs_path = os.path.join(args.checkpoint_dir, "val_sequences.json")
    with open(val_seqs_path, "w") as f:
        json.dump(sorted(list(val_seqs)), f, indent=4)
    print(f"Saved validation sequence list to: {val_seqs_path}")
    
    # ---------------------------------------------------------
    # Mix PyBullet Data
    # ---------------------------------------------------------
    try:
        pybullet_dataset = PyBulletDataset(dataset_dir="dataset_pybullet", target_size=(224, 224))
        print(f"Loaded PyBullet Dataset: {len(pybullet_dataset)} samples.")
        train_mixed = torch.utils.data.ConcatDataset([train_subset, pybullet_dataset])
        print(f"Total mixed training samples: {len(train_mixed)}")
        has_pybullet = True
    except Exception as e:
        print(f"Could not load PyBullet dataset ({e}), training only on ODA.")
        train_mixed = train_subset
        has_pybullet = False
    
    num_low = sum(1 for idx in train_indices if dataset.samples[idx]['lux'] <= 3)
    num_high = len(train_indices) - num_low
    
    if num_low == 0:
        total_len = len(train_mixed)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=[1.0] * total_len,
            num_samples=total_len,
            replacement=True
        )
    else:
        w_low = 0.22 / num_low
        w_high = 0.78 / num_high
        weights = [w_low if dataset.samples[idx]['lux'] <= 3 else w_high for idx in train_indices]
        if has_pybullet:
            weights.extend([w_high] * len(pybullet_dataset))
            
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
    
    train_loader = DataLoader(
        train_mixed,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print("Initializing ODANetFast...")
    model = ODANetFast(
        backbone_name=args.backbone,
        pretrained=args.pretrained
    ).to(device)
    
    print("\n--- Model Parameter Summary ---")
    total_params = 0
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  - {name}: {params:,} parameters")
        total_params += params
    print(f"Total Model Parameters: {total_params:,}")
    print("-------------------------------\n")
    
    if args.compile:
        if hasattr(torch, "compile"):
            try:
                print("Compiling ODANetFast sub-modules with torch.compile...")
                model.backbone = torch.compile(model.backbone)
                model.depth_head = torch.compile(model.depth_head)
            except Exception as e:
                print(f"Sub-module compilation skipped/failed: {e}")
        else:
            print("torch.compile is not supported on this PyTorch version.")
    
    is_backbone_frozen = True
    print(f"Freezing visual backbone '{args.backbone}' for first {args.unfreeze_epoch} epochs.")
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    optimizer = get_optimizer(model, args.lr, args.backbone_lr, is_frozen=is_backbone_frozen)
    
    criterion_depth_map = MetricDepthLoss(alpha=0.5, scale=1.0, w_l1=0.1)
    
    # Initialize GPU Data Augmentations
    augmentation = T.Compose([
        T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
        T.RandomGrayscale(p=0.3),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
        T.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
    ])
    
    best_val_loss = float('inf')
    scheduler = None
    
    for epoch in range(1, args.epochs + 1):
        if epoch == args.unfreeze_epoch + 1:
            print(f"\n--- Unfreezing visual backbone at epoch {epoch} with LR = {args.backbone_lr} ---")
            is_backbone_frozen = False
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = get_optimizer(model, args.lr, args.backbone_lr, is_frozen=is_backbone_frozen)
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs - args.unfreeze_epoch,
                eta_min=1e-6
            )
            
        model.train()
        train_loss = 0.0
        train_loss_depth = 0.0
        
        t0 = time.time()
        train_t0 = time.time()
        optimizer.zero_grad()
        
        num_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]", leave=False)
        for batch_idx, batch in enumerate(pbar):
            if args.dry_run and batch_idx >= 5:
                break
            num_batches += 1
            inputs = batch['rgb'].to(device)
            
            # Apply GPU Augmentation to inputs (only during training)
            with torch.no_grad():
                inputs = augmentation(inputs)
                
            pseudo_depth = batch['depth'].unsqueeze(1).to(device)
            
            outputs = model(images=inputs)
            
            loss_depth = criterion_depth_map(outputs["depth"], pseudo_depth)
            loss = args.w_depth * loss_depth
            
            loss = loss / args.grad_accum
            loss.backward()
            
            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * args.grad_accum
            train_loss_depth += loss_depth.item()
            
            pbar.set_postfix({
                "loss": f"{loss.item() * args.grad_accum:.4f}",
                "depth": f"{loss_depth.item():.3f}"
            })
            
        train_loss /= num_batches
        train_loss_depth /= num_batches
        
        train_time = time.time() - train_t0
        train_fps = (num_batches * args.batch_size) / max(train_time, 1e-5)
        
        run_val = (epoch % args.val_interval == 0) or (epoch == args.epochs)
        
        if run_val:
            val_t0 = time.time()
            model.eval()
            val_loss = 0.0
            val_loss_depth = 0.0
            val_depth_abs_rel_total = 0.0
            
            num_val_batches = 0
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Val]", leave=False)
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_pbar):
                    if args.dry_run and batch_idx >= 2:
                        break
                    num_val_batches += 1
                    inputs = batch['rgb'].to(device)
                        
                    poses = batch['pose'].to(device)
                    pseudo_depth = batch['depth'].unsqueeze(1).to(device)
                    
                    outputs = model(images=inputs)
                    
                    loss_depth = criterion_depth_map(outputs["depth"], pseudo_depth)
                    loss = args.w_depth * loss_depth
                    
                    val_loss += loss.item()
                    val_loss_depth += loss_depth.item()
                    
                    gt_mask = (pseudo_depth > 0.0) & (pseudo_depth <= 10.0)
                    if gt_mask.any():
                        pred_raw = outputs["depth"]
                        scale = torch.median(pseudo_depth) / torch.clamp(torch.median(pred_raw), min=1e-3)
                        pred_aligned = pred_raw * scale
                        abs_rel = torch.mean(torch.abs(pred_aligned[gt_mask] - pseudo_depth[gt_mask]) / torch.clamp(pseudo_depth[gt_mask], min=1e-3)).item()
                        val_depth_abs_rel_total += abs_rel
                    
                    if batch_idx == 0 and epoch % 2 == 0:
                        save_prediction_plot(epoch, epoch // 2, inputs, pseudo_depth, outputs["depth"], vis_dir)
                        
                    val_pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "depth": f"{loss_depth.item():.3f}"
                    })
                        
            val_loss /= num_val_batches
            val_loss_depth /= num_val_batches
            val_depth_abs_rel = val_depth_abs_rel_total / num_val_batches
            
            val_time = time.time() - val_t0
            val_fps = (num_val_batches * args.batch_size) / max(val_time, 1e-5)
            val_print = f"Val Loss: {val_loss:.4f} (Depth: {val_loss_depth:.3f}) | Val AbsRel: {val_depth_abs_rel*100:.1f}%"
        else:
            val_fps = float('nan')
            val_depth_abs_rel = float('nan')
            val_print = "Val Loss: skipped"
            
        if device.type == "cuda":
            gpu_max_mem = torch.cuda.max_memory_allocated(device) / (1024**3)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            gpu_max_mem = 0.0
            
        lr_heads = optimizer.param_groups[0]['lr']
        lr_backbone = optimizer.param_groups[1]['lr'] if not is_backbone_frozen else 0.0
        
        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Time: {dt:.1f}s | "
              f"Train Loss: {train_loss:.4f} (Depth: {train_loss_depth:.3f}) | "
              f"{val_print} | Speed: {train_fps:.1f} FPS | GPU Mem: {gpu_max_mem:.2f} GB | "
              f"LR: {lr_heads:.2e}/{lr_backbone:.2e}")
              
        csv_path = os.path.join(args.checkpoint_dir, "training_log.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "epoch", "train_loss", "train_loss_depth",
                    "val_loss", "val_loss_depth",
                    "train_fps", "val_fps", "val_depth_abs_rel", "lr_heads", "lr_backbone", "epoch_time_sec", "gpu_max_mem_gb"
                ])
            writer.writerow([
                epoch, train_loss, train_loss_depth,
                val_loss if run_val else "", 
                val_loss_depth if run_val else "", 
                f"{train_fps:.2f}", 
                f"{val_fps:.2f}" if run_val else "",
                f"{val_depth_abs_rel:.4f}" if run_val else "",
                f"{lr_heads:.2e}",
                f"{lr_backbone:.2e}",
                f"{dt:.2f}", 
                f"{gpu_max_mem:.2f}"
            ])
            f.flush()
            os.fsync(f.fileno())
              
        latest_path = os.path.join(args.checkpoint_dir, "latest_model_fast.pth")
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'is_backbone_frozen': is_backbone_frozen
        }
        if run_val:
            save_dict['val_loss'] = val_loss
        torch.save(save_dict, latest_path)
        
        if run_val and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.checkpoint_dir, "best_model_fast.pth")
            torch.save(save_dict, best_path)
            print(f"-> Saved new best model checkpoint to {best_path} with val_loss={val_loss:.4f}")
            
        if scheduler is not None:
            scheduler.step()

    print("\nTraining Phase 4 (Fast) completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: Fine-tune ODANetFast (No Projection)")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Path to real ODA dataset")
    parser.add_argument("--backbone", type=str, default="fastvit_t12", help="Visual backbone variant")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Sequence batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for non-backbone components")
    parser.add_argument("--backbone_lr", type=float, default=3e-5, help="Learning rate for backbone after unfreezing")
    parser.add_argument("--unfreeze_epoch", type=int, default=1, help="Epoch number to unfreeze the visual backbone")
    parser.add_argument("--pretrained", type=bool, default=True, help="Load pretrained backbone weights from ImageNet")
    parser.add_argument("--cache_data", type=bool, default=False, help="Cache dataset sensor files in VRAM/RAM")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint/phase4", help="Directory to save weights/vis")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader subprocess workers")
    parser.add_argument("--device", type=str, default="cuda", help="Primary execution device (cuda/cpu)")
    parser.add_argument("--dry_run", action="store_true", help="Run a quick 5-batch training / 2-batch validation dry run")
    parser.add_argument("--compile", action="store_true", help="Optimize trainable sub-modules with torch.compile")
    parser.add_argument("--val_interval", type=int, default=1, help="Validation epoch interval")
    
    # Loss weights
    parser.add_argument("--w_depth", type=float, default=1.0, help="Weight for 2D depth map loss")
    
    args = parser.parse_args()
        
    train_phase4(args)
