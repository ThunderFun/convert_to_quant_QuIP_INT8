#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import glob
import json

def load_lora_mapping(mapping_path: str) -> dict:
    """Load LoRA to model mapping from JSON file."""
    with open(mapping_path, 'r') as f:
        return json.load(f)

def find_lora_for_model(model_path: str, mapping: dict, lora_dir: str = None) -> list:
    """Find the LoRA file(s) for a given model using the mapping.
    
    Returns a list of LoRA paths (supports multiple LoRAs per model).
    """
    model_name = os.path.basename(model_path)
    model_base = os.path.splitext(model_name)[0]
    
    # Direct match by full filename
    if model_name in mapping:
        lora_entry = mapping[model_name]
        return lora_entry if isinstance(lora_entry, list) else [lora_entry]
    
    # Match by base name (without extension)
    if model_base in mapping:
        lora_entry = mapping[model_base]
        return lora_entry if isinstance(lora_entry, list) else [lora_entry]
    
    # Check for pattern-based matches
    for pattern, lora_entry in mapping.items():
        if pattern.startswith("_"):  # Skip metadata keys
            continue
        
        # Use more robust matching: check if pattern is a substring of the base name
        # but avoid partial matches like 'model_1' matching 'model_10'
        if pattern == model_base or pattern == model_name:
            return lora_entry if isinstance(lora_entry, list) else [lora_entry]
            
        # Pattern matching with delimiters
        import re
        escaped_pattern = re.escape(pattern)
        # Match pattern as a whole word or delimited by _ or . or -
        regex = f"(^|[._-]){escaped_pattern}([._-]|$)"
        if re.search(regex, model_base) or re.search(regex, model_name):
            return lora_entry if isinstance(lora_entry, list) else [lora_entry]
    
    # If lora_dir is specified, look for matching LoRA by name
    if lora_dir:
        potential_lora = os.path.join(lora_dir, f"{model_base}.safetensors")
        if os.path.exists(potential_lora):
            return [potential_lora]
        # Try with _lora suffix
        potential_lora = os.path.join(lora_dir, f"{model_base}_lora.safetensors")
        if os.path.exists(potential_lora):
            return [potential_lora]
    
    return []

def main():
    parser = argparse.ArgumentParser(
        description="Batch quantize multiple models using convert_to_quant.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_quantize.py -i model1.safetensors model2.safetensors --comfy_quant
  python batch_quantize.py -d ./models/ --pattern "*.safetensors" --optimizer gptq
  python batch_quantize.py -i models/*.safetensors --simple
  
LoRA Merging Examples:
  python batch_quantize.py -d ./models/ --lora-mapping mapping.json --optimizer quip
  python batch_quantize.py -d ./models/ --lora-dir ./loras/ --optimizer quip
  python batch_quantize.py -i model.safetensors --merge-lora lora.safetensors
"""
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--inputs", nargs="+", help="List of input safetensors files.")
    group.add_argument("-d", "--directory", type=str, help="Directory containing safetensors files.")
    
    parser.add_argument("--pattern", type=str, default="*.safetensors", help="Pattern to match files in directory (default: *.safetensors).")
    parser.add_argument("--dry-run-batch", action="store_true", help="Show what commands would be executed without running them.")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt.")
    
    # LoRA merging options
    lora_group = parser.add_argument_group("LoRA Merging Options")
    lora_group.add_argument("--lora-mapping", type=str, help="JSON file mapping model names to LoRA paths.")
    lora_group.add_argument("--lora-dir", type=str, help="Directory containing LoRA files (will match by model name).")
    lora_group.add_argument("--merge-lora", type=str, help="Single LoRA to merge with all models.")
    lora_group.add_argument("--merge-lora-scale", type=float, default=1.0, help="Scale factor for LoRA merging (default: 1.0).")
    
    # We want to allow passing any other arguments to the underlying script
    args, unknown = parser.parse_known_args()

    if ("-o" in unknown or "--output" in unknown):
        print("Error: You have specified an output path with -o/--output.")
        print("In batch mode, this would cause all models to be saved to the SAME file, overwriting each other.")
        print("Please omit -o/--output to let the script generate unique filenames based on input names.")
        sys.exit(1)

    input_files = []
    if args.inputs:
        for item in args.inputs:
            # Handle cases where shell expansion might not have happened or for manual lists
            expanded = glob.glob(item)
            if expanded:
                input_files.extend(expanded)
            else:
                input_files.append(item)
    elif args.directory:
        search_path = os.path.join(args.directory, args.pattern)
        input_files = glob.glob(search_path)
        if not input_files:
            print(f"No files matching {args.pattern} found in {args.directory}")
            return

    # Remove duplicates and sort
    input_files = sorted(list(set(input_files)))
    
    # Load LoRA mapping if provided
    lora_mapping = {}
    if args.lora_mapping:
        if not os.path.exists(args.lora_mapping):
            print(f"Error: LoRA mapping file not found: {args.lora_mapping}")
            return
        lora_mapping = load_lora_mapping(args.lora_mapping)
        print(f"Loaded LoRA mapping with {len(lora_mapping)} entries")

    # Summarize parameters from unknown args
    print(f"\n{'='*40}")
    print("Batch Configuration Summary")
    print(f"{'='*40}")
    print(f"Input files: {len(input_files)}")
    
    # Check if LoRA merging is configured
    if args.merge_lora:
        print(f"LoRA (all models): {args.merge_lora}")
        print(f"LoRA scale: {args.merge_lora_scale}")
    elif args.lora_mapping or args.lora_dir:
        print(f"LoRA mode: Per-model mapping")
        if args.lora_dir:
            print(f"LoRA directory: {args.lora_dir}")
    
    # Simplified parser to peek at important flags in unknown args
    peek_parser = argparse.ArgumentParser(add_help=False)
    peek_parser.add_argument("--comfy_quant", action="store_true")
    peek_parser.add_argument("--smoothquant", action="store_true")
    peek_parser.add_argument("--optimizer", type=str, default="original")
    peek_parser.add_argument("--block_size", type=int, default=128)
    peek_parser.add_argument("--simple", action="store_true")
    peek_parser.add_argument("--heur", action="store_true")
    peek_parser.add_argument("--fp16", action="store_true")
    peek_parser.add_argument("--streaming-mode", type=str, default="balanced")
    peek_parser.add_argument("--quip-actorder", action="store_true", default=True)
    peek_parser.add_argument("--no-quip-actorder", action="store_false", dest="quip_actorder")
    peek_parser.add_argument("--quip-hadamard", action="store_true", default=True)
    peek_parser.add_argument("--no-quip-hadamard", action="store_false", dest="quip_hadamard")
    
    peek_args, _ = peek_parser.parse_known_args(unknown)
    
    if peek_args.fp16:
        print("Mode: FP16")
    else:
        print(f"Mode: INT8 (Block Size: {peek_args.block_size})")
    
    print(f"Optimizer: {peek_args.optimizer}")
    
    flags = []
    if peek_args.comfy_quant: flags.append("Comfy Quant")
    if peek_args.smoothquant: flags.append("SmoothQuant")
    if peek_args.simple: flags.append("Simple (No SVD)")
    if peek_args.heur: flags.append("Heuristics")
    if peek_args.streaming_mode != "balanced":
        flags.append(f"Streaming ({peek_args.streaming_mode})")
    
    # Check for QuIP/GPTQ specific flags that are enabled by default or explicitly
    if peek_args.optimizer == "quip":
        if peek_args.quip_actorder: flags.append("QuIP ActOrder")
        if peek_args.quip_hadamard: flags.append("QuIP Hadamard")
    elif peek_args.optimizer == "gptq":
        flags.append("GPTQ Fast (Enabled)")
        if "--gptq-actorder" in unknown: flags.append("GPTQ ActOrder")
        if "--gptq-turbo" in unknown: flags.append("GPTQ Turbo")

    if flags:
        print(f"Flags: {', '.join(flags)}")
    
    if unknown:
        print(f"Raw extra args: {' '.join(unknown)}")
    print(f"{'='*40}\n")

    if not args.yes and not args.dry_run_batch:
        confirm = input("Proceed with batch processing? (y/N): ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return

    for i, input_file in enumerate(input_files):
        if not os.path.exists(input_file):
            print(f"[{i+1}/{len(input_files)}] Skipping {input_file}: File not found.")
            continue
        
        # Determine which LoRA(s) to use for this model
        model_loras = []
        if args.merge_lora:
            model_loras = [args.merge_lora]  # Global LoRA applies to all
        elif args.lora_mapping or args.lora_dir:
            model_loras = find_lora_for_model(input_file, lora_mapping, args.lora_dir)
        
        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(input_files)}] Processing: {input_file}")
        if model_loras:
            if len(model_loras) == 1:
                print(f"  LoRA: {model_loras[0]}")
            else:
                print(f"  LoRAs ({len(model_loras)}):")
                for lorapath in model_loras:
                    print(f"    - {lorapath}")
        print(f"{'='*80}")
        
        # Use the same python interpreter and call the module
        # We assume the current directory is in PYTHONPATH or the package is installed
        cmd = [sys.executable, "-m", "convert_to_quant.cli.main", "-i", input_file] + unknown
        
        # Add LoRA merging arguments if specified for this model
        if model_loras:
            if len(model_loras) == 1:
                cmd.extend(["--merge-lora", model_loras[0]])
            else:
                cmd.extend(["--merge-loras"] + model_loras)
            if args.merge_lora_scale != 1.0:
                cmd.extend(["--merge-lora-scale", str(args.merge_lora_scale)])
        
        if args.dry_run_batch:
            print(f"Dry run: {' '.join(cmd)}")
            continue

        try:
            # Run the command and wait for it to finish
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"\n[!] Error: Quantization failed for {input_file} with exit code {result.returncode}")
            else:
                print(f"\n[+] Successfully processed {input_file}")
        except Exception as e:
            print(f"\n[!] Unexpected error processing {input_file}: {e}")

    print(f"\nDone! Processed {len(input_files)} files.")
    
    # Print example mapping file format if LoRA options were used
    if args.lora_mapping or args.lora_dir:
        print("\n" + "="*60)
        print("LoRA Mapping File Format (JSON):")
        print("="*60)
        print("""{
  "model1.safetensors": "/path/to/lora1.safetensors",
  "model2.safetensors": "/path/to/lora2.safetensors",
  "model_base_name": "/path/to/specific_lora.safetensors"
}""")
        print("="*60)

if __name__ == "__main__":
    main()
