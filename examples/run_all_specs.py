#!/usr/bin/env python3
"""
Script to systematically run specifications with different seeds and optimizers.
Supports both Autotrans and CC benchmarks.

For each specification:
1. Try seeds 1-10 with specified optimizer
2. Stop when specification is falsified (robustness < 0)
3. If falsified, also run with LLM optimizer using same spec and seed
"""

import subprocess
import sys
import re
import os
import argparse
from typing import Optional, Tuple, Dict, List

def get_benchmark_config(benchmark: str) -> Dict:
    """
    Get configuration for the specified benchmark.
    
    Args:
        benchmark: Either "autotrans" or "cc"
        
    Returns:
        Dictionary containing benchmark-specific configuration
    """
    if benchmark == "autotrans":
        return {
            "script": "autotrans_all_specs.py",
            "specs": ["AT1", "AT2", "AT51", "AT52", "AT53", "AT54", "AT61", "AT62", "AT63", "AT64"],
            "output_dir": "autotrans_all_specs",
            "summary_file": "autotrans_analysis_summary.txt"
        }
    elif benchmark == "cc":
        return {
            "script": "cc_all_specs.py", 
            "specs": ["CC1", "CC2", "CC3", "CC4", "CC5", "CCx"],
            "output_dir": "cc_all_specs",
            "summary_file": "cc_analysis_summary.txt"
        }
    elif benchmark == "f16":
        return {
            "script": "f16_all_specs.py",
            "specs": ["F16_ALT1", "F16_ALT2", "F16_ALT3", "F16_ROLL1", "F16_ROLL2", "F16_ROLL3", "F16_PITCH1", "F16_PITCH2", "F16_PITCH3", "F16_YAW1", "F16_MODE1", "F16_MODE2", "F16_SAFE1", "F16_SAFE2"],
            "output_dir": "f16_all_specs",
            "summary_file": "f16_analysis_summary.txt"
        }
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

def run_spec(benchmark: str, spec: str, optimizer: str, seed: int, timeout: int) -> Tuple[bool, float]:
    """
    Run specification script with given parameters.
    
    Args:
        benchmark: Benchmark type ("autotrans" or "cc")
        spec: Specification to test
        optimizer: Optimizer to use
        seed: Random seed
        timeout: Timeout in seconds
    
    Returns:
        Tuple of (success, robustness_value)
        success: True if command ran successfully
        robustness_value: The robustness value from the output
    """
    config = get_benchmark_config(benchmark)
    script_name = config["script"]
    
    cmd = [
        sys.executable, script_name,
        "-s", spec,
        "-o", optimizer,
        "--seed", str(seed)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        if timeout is not None:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
        
        if result.returncode != 0:
            print(f"❌ Command failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return False, 0.0
        
        # Parse robustness value from output
        robustness = parse_robustness_from_output(result.stdout)
        
        if robustness is not None:
            print(f"✅ Completed. Robustness: {robustness:.6f}")
            return True, robustness
        else:
            print("⚠️  Could not parse robustness value from output")
            return False, 0.0
            
    except subprocess.TimeoutExpired:
        print(f"❌ Command timed out after {timeout} seconds")
        return False, 0.0
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False, 0.0

def parse_robustness_from_output(output: str) -> Optional[float]:
    """
    Parse robustness value from the script output.
    
    Returns:
        Robustness value if found, None otherwise
    """
    # Look for "Robustness value: X.XXXXXX" pattern
    pattern = r"Robustness value:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    match = re.search(pattern, output)
    
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    return None

def main():
    """Main execution function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Systematically run specifications with different seeds and optimizers"
    )
    parser.add_argument(
        "-b", "--benchmark",
        required=True,
        choices=["autotrans", "cc", "f16"],
        help="Benchmark to run (autotrans or cc or f16)"
    )
    parser.add_argument(
        "-o", "--optimizer",
        default="DA",
        choices=["DA", "DE", "BH", "PSO", "CMAES", "LLM", "LLMGB", "UR"],
        help="Optimizer to use (default: DA)"
    )
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for each run (default: None - no timeout)"
    )
    parser.add_argument(
        "--run-llm",
        action="store_true",
        help="Run LLM optimizer on falsified specifications"
    )
    parser.add_argument(
        "-n", "--n-seeds",
        type=int,
        default=10,
        help="Number of seeds to try (default: 10)"
    )
    parser.add_argument(
        "--brfals",
        action="store_true",
        help="Break the seeds loop when specification is falsified (default: False)"
    )
    
    args = parser.parse_args()
    
    # Get benchmark configuration
    config = get_benchmark_config(args.benchmark)
    specs = config["specs"]
    script_name = config["script"]
    output_dir = config["output_dir"]
    summary_file = config["summary_file"]
    
    # Use provided timeout (None means no timeout)
    timeout = args.timeout
    
    print(f"🚀 Starting systematic {args.benchmark.upper()} specification analysis")
    print(f"🔧 Using optimizer: {args.optimizer}")
    if timeout is not None:
        print(f"⏱️  Timeout per run: {timeout} seconds")
    else:
        print(f"⏱️  No timeout limit")
    print("=" * 60)
    
    # Check if the required script exists
    if not os.path.exists(script_name):
        print(f"❌ Error: {script_name} not found in current directory")
        sys.exit(1)
    
    results_summary = []
    
    for spec in specs:
        print(f"\n📋 Testing specification: {spec}")
        print("-" * 40)
        
        falsified = False
        falsifying_seed = None
        falsifying_robustness = None
        
        N_seeds = args.n_seeds
        # Try seeds 1-N_seeds with specified optimizer
        for seed in range(1, N_seeds+1):
            print(f"\n🎲 Seed {seed} with {args.optimizer} optimizer:")
            
            success, robustness = run_spec(args.benchmark, spec, args.optimizer, seed, timeout)
            
            if not success:
                print(f"⚠️  Failed to run {spec} with seed {seed}, continuing...")
                continue
            
            # Check if specification was falsified
            if robustness < 0:
                print(f"🚨 FALSIFIED! Robustness: {robustness:.6f}")
                falsified = True
                falsifying_seed = seed
                falsifying_robustness = robustness
                if args.brfals:
                    break
            else:
                print(f"✅ Satisfied. Robustness: {robustness:.6f}")
        
        # If falsified, also run with LLM optimizer (if requested)
        if falsified and falsifying_seed is not None and args.run_llm:
            print(f"\n🤖 Running {spec} with LLM optimizer (seed {falsifying_seed}):")
            
            llm_success, llm_robustness = run_spec(args.benchmark, spec, "LLM", falsifying_seed, timeout)
            
            if llm_success:
                results_summary.append({
                    'spec': spec,
                    'falsified': True,
                    'seed': falsifying_seed,
                    'optimizer_robustness': falsifying_robustness,
                    'llm_robustness': llm_robustness,
                    'optimizer_name': args.optimizer
                })
            else:
                results_summary.append({
                    'spec': spec,
                    'falsified': True,
                    'seed': falsifying_seed,
                    'optimizer_robustness': falsifying_robustness,
                    'llm_robustness': None,
                    'optimizer_name': args.optimizer
                })
        else:
            # Record results without LLM optimizer
            llm_robustness = None
            if falsified and falsifying_seed is not None:
                # Falsified but LLM not requested
                results_summary.append({
                    'spec': spec,
                    'falsified': True,
                    'seed': falsifying_seed,
                    'optimizer_robustness': falsifying_robustness,
                    'llm_robustness': llm_robustness,
                    'optimizer_name': args.optimizer
                })
            else:
                # Not falsified
                results_summary.append({
                    'spec': spec,
                    'falsified': False,
                    'seed': None,
                    'optimizer_robustness': None,
                    'llm_robustness': None,
                    'optimizer_name': args.optimizer
                })
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 60)
    
    summary_lines = []
    if args.run_llm:
        summary_lines.append("LLM optimizer requested on falsified specifications")
    else:
        summary_lines.append("LLM optimizer not requested on falsified specifications")
    for result in results_summary:
        spec = result['spec']
        if result['falsified']:
            seed = result['seed']
            opt_rob = result['optimizer_robustness']
            llm_rob = result['llm_robustness']
            opt_name = result['optimizer_name']
            
            print(f"\n{spec}:")
            summary_lines.append(f"\n{spec}:")
            print(f"  🚨 FALSIFIED at seed {seed}")
            summary_lines.append(f"  🚨 FALSIFIED at seed {seed}")
            print(f"  📈 {opt_name} robustness: {opt_rob:.6f}")
            summary_lines.append(f"  📈 {opt_name} robustness: {opt_rob:.6f}")
            if llm_rob is not None:
                print(f"  🤖 LLM robustness: {llm_rob:.6f}")
                summary_lines.append(f"  🤖 LLM robustness: {llm_rob:.6f}")
                if llm_rob < 0:
                    print(f"     → LLM also falsified")
                    summary_lines.append(f"     → LLM also falsified")
                else:
                    print(f"     → LLM satisfied the specification")
                    summary_lines.append(f"     → LLM satisfied the specification")
            else:
                print(f"  🤖 LLM run failed or not requested")
                summary_lines.append(f"  🤖 LLM run failed or not requested")
        else:
            print(f"\n{spec}:")
            summary_lines.append(f"\n{spec}:")
            print(f"  ✅ NOT FALSIFIED (tried seeds 1-{N_seeds})")
            summary_lines.append(f"  ✅ NOT FALSIFIED (tried seeds 1-{N_seeds})")
    
    print(f"\n🏁 Analysis complete! Check ./{output_dir}/ for detailed results.")
    summary_lines.append(f"\n🏁 Analysis complete! Check ./{output_dir}/ for detailed results.")

    # Save summary to a txt file inside the benchmark output folder
    summary_dir = os.path.join(os.path.dirname(__file__), output_dir)
    os.makedirs(summary_dir, exist_ok=True)
    summary_file = summary_file.replace(".txt", f"_{args.optimizer}.txt")
    summary_path = os.path.join(summary_dir, summary_file)
    try:
        with open(summary_path, "w") as f:
            for line in summary_lines:
                f.write(line + "\n")
        print(f"\n📝 Summary saved to {summary_path}")
    except Exception as e:
        print(f"⚠️  Failed to write summary file: {e}")

if __name__ == "__main__":
    main() 