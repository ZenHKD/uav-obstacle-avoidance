import os
import sys
import torch
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.model.net_fast import ODANetFast

def export_to_onnx(checkpoint_path, output_path, batch_size=1, height=224, width=224):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading weights from {checkpoint_path}...")
    
    # 1. Initialize model
    model = ODANetFast().to(device)
    
    # 2. Load Phase 4 trained weights
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
        
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # 3. Create dummy input matching the expected input shape
    dummy_input = torch.randn(batch_size, 3, height, width, device=device)
    
    print(f"Exporting model to ONNX format: {output_path}...")
    
    # 4. Export to ONNX
    # We specify dynamic axes for batch size so it can be flexible later
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=18,          # ONNX opset version (18 required for modern Resize ops)
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['input_rgb'], # the model's input names
        output_names=['depth', 'confidence', 'distance'] # the model's output names
    )
    
    print(f"=> ONNX Export Successful! Saved at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ODANetFast to ONNX")
    parser.add_argument("--ckpt", type=str, default="checkpoint/phase4/best_model_fast.pth", help="Path to PyTorch checkpoint")
    parser.add_argument("--out", type=str, default="checkpoint/phase4/odanet_fast.onnx", help="Output path for ONNX model")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    export_to_onnx(args.ckpt, args.out)
