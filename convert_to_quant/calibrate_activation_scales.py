"""
Calibrate activation scales for INT8 quantization.

Generates per-channel activation statistics (max, mean) for SmoothQuant and GPTQ.
Supports LoRA-informed calibration for better accuracy.
"""
import argparse
import json
import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from typing import Dict, Optional
from .utils.tensor_utils import load_lora_tensors

def calibrate_model(
    input_file: str,
    output_file: str,
    samples: int = 64,
    seed: int = 42,
    lora_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"Calibrating model: {input_file}")
    if lora_path:
        print(f"Using LoRA for informed calibration: {lora_path}")
    print(f"Samples: {samples}, Seed: {seed}, Device: {device}")

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    lora_tensors = load_lora_tensors(lora_path) if lora_path else {}
    results = {}
    
    with safe_open(input_file, framework="pt", device="cpu") as f:
        keys = [k for k in f.keys() if k.endswith(".weight")]
        
        for key in tqdm(keys, desc="Calibrating layers"):
            weight_shape = f.get_shape(key)
            if len(weight_shape) != 2:
                continue
                
            in_features = weight_shape[1]
            base_name = key.rsplit(".weight", 1)[0]
            
            # Find matching LoRA if available
            lora_match = None
            if lora_tensors:
                # Try exact match or normalized match
                norm_base = base_name.replace("model.diffusion_model.", "").replace("transformer.", "")
                if norm_base in lora_tensors:
                    lora_match = lora_tensors[norm_base]
            
            # Generate calibration inputs
            if lora_match is not None:
                # Use LoRA directions + some noise
                rank = lora_match.shape[0]
                # Repeat or truncate LoRA directions to match requested samples
                if rank >= samples:
                    x = lora_match[:samples].to(device).to(torch.float32)
                else:
                    # Mix LoRA directions with random noise
                    x_lora = lora_match.to(device).to(torch.float32)
                    x_rand = torch.randn(samples - rank, in_features, device=device)
                    x = torch.cat([x_lora, x_rand], dim=0)
            else:
                # Pure random noise
                x = torch.randn(samples, in_features, device=device)
            
            # Compute statistics for SmoothQuant and GPTQ
            channel_max = x.abs().max(dim=0)[0].cpu()
            channel_mean = x.abs().mean(dim=0).cpu()
            global_max = channel_max.max().item()
            
            results[f"{base_name}.channel_max"] = channel_max
            results[f"{base_name}.channel_mean"] = channel_mean
            results[f"{base_name}.input_scale"] = torch.tensor(global_max / 127.0, dtype=torch.float32)

    if output_file.endswith(".json"):
        json_results = {}
        for key, tensor in results.items():
            base, stat = key.rsplit(".", 1)
            if base not in json_results:
                json_results[base] = {}
            json_results[base][stat] = tensor.tolist() if tensor.ndim > 0 else tensor.item()
            
        with open(output_file, "w") as f:
            json.dump(json_results, f, indent=2)
    else:
        save_file(results, output_file)
        
    print(f"Calibration data saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Calibrate activation scales for INT8 quantization.")
    parser.add_argument("input", help="Input safetensors file")
    parser.add_argument("-o", "--output", required=True, help="Output file (.safetensors or .json)")
    parser.add_argument("--samples", type=int, default=64, help="Number of calibration samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lora", help="Optional LoRA file for informed calibration")
    parser.add_argument("--json", action="store_true", help="Save as JSON")

    args = parser.parse_args()
    
    output_path = args.output
    if args.json and not output_path.endswith(".json"):
        output_path += ".json"
        
    calibrate_model(
        args.input,
        output_path,
        samples=args.samples,
        seed=args.seed,
        lora_path=args.lora,
    )

if __name__ == "__main__":
    main()
