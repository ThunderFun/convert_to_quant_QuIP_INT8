#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import glob

def main():
    parser = argparse.ArgumentParser(
        description="Batch quantize multiple models using convert_to_quant.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_quantize.py -i model1.safetensors model2.safetensors --comfy_quant
  python batch_quantize.py -d ./models/ --pattern "*.safetensors" --optimizer gptq
  python batch_quantize.py -i models/*.safetensors --simple
"""
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--inputs", nargs="+", help="List of input safetensors files.")
    group.add_argument("-d", "--directory", type=str, help="Directory containing safetensors files.")
    
    parser.add_argument("--pattern", type=str, default="*.safetensors", help="Pattern to match files in directory (default: *.safetensors).")
    parser.add_argument("--dry-run-batch", action="store_true", help="Show what commands would be executed without running them.")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt.")
    
    # We want to allow passing any other arguments to the underlying script
    args, unknown = parser.parse_known_args()

    if ("-o" in unknown or "--output" in unknown) and not args.dry_run_batch:
        print("Warning: You have specified an output path with -o/--output.")
        print("In batch mode, this will cause all models to be saved to the SAME file, overwriting each other.")
        print("It is recommended to omit -o/--output to let the script generate unique filenames.")
        # In non-interactive environments, we might want to proceed or abort.
        # For safety, let's just print a strong warning.

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

    # Summarize parameters from unknown args
    print(f"\n{'='*40}")
    print("Batch Configuration Summary")
    print(f"{'='*40}")
    print(f"Input files: {len(input_files)}")
    
    # Simplified parser to peek at important flags in unknown args
    peek_parser = argparse.ArgumentParser(add_help=False)
    peek_parser.add_argument("--comfy_quant", action="store_true")
    peek_parser.add_argument("--smoothquant", action="store_true")
    peek_parser.add_argument("--optimizer", type=str, default="original")
    peek_parser.add_argument("--block_size", type=int, default=128)
    peek_parser.add_argument("--simple", action="store_true")
    peek_parser.add_argument("--heur", action="store_true")
    peek_parser.add_argument("--fp16", action="store_true")
    peek_parser.add_argument("--quip-actorder", action="store_false", default=True) # default is True in main.py
    peek_parser.add_argument("--no-quip-actorder", action="store_true") # handle potential negation if it existed, but main.py doesn't have it.
    # Actually main.py has: parser.add_argument("--quip-actorder", action="store_true", default=True, help="Enable activation ordering for QuIP.")
    # Wait, if action="store_true" and default=True, then it's always True unless... wait.
    # In argparse, if default=True and action="store_true", it's always True. That's usually a bug in the parser definition if they wanted a way to disable it.
    # Looking at main.py:
    # parser.add_argument("--quip-actorder", action="store_true", default=True, help="Enable activation ordering for QuIP.")
    # This means it's always True.
    
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
    
    # Check for QuIP/GPTQ specific flags that are enabled by default or explicitly
    if peek_args.optimizer == "quip":
        flags.append("QuIP ActOrder (Enabled)")
        flags.append("QuIP Hadamard (Enabled)")
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
        
        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(input_files)}] Processing: {input_file}")
        print(f"{'='*80}")
        
        # Use the same python interpreter and call the module
        # We assume the current directory is in PYTHONPATH or the package is installed
        cmd = [sys.executable, "-m", "convert_to_quant.cli.main", "-i", input_file] + unknown
        
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

if __name__ == "__main__":
    main()
