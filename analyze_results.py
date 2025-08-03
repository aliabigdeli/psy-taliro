#!/usr/bin/env python3
"""
Script to analyze optimization results and generate summary statistics.

Usage: python analyze_results.py <optimizer_name> <benchmark_name>
Example: python analyze_results.py LLM autotrans
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path


def analyze_results(optimizer_name, benchmark_name):
    """
    Analyze results CSV file and generate summary statistics.
    
    Args:
        optimizer_name (str): Name of the optimizer (e.g., 'LLM', 'DA', 'UR')
        benchmark_name (str): Name of the benchmark (e.g., 'autotrans', 'cc')
    
    Returns:
        pd.DataFrame: Summary statistics for each specification
    """
    
    # Construct input file path
    input_file = f"examples/{benchmark_name}_all_specs/results_{optimizer_name}.csv"
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found")
    
    # Read the CSV file
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    # Group by specification
    grouped = df.groupby('specification')
    
    results = []
    
    for spec, group in grouped:
        total_runs = len(group)
        falsified_runs = group[group['Falsified'] == True]
        num_falsified = len(falsified_runs)
        
        # Calculate falsification ratio
        falsification_ratio = num_falsified / total_runs
        
        # Calculate robustness statistics
        robustness_values = group['robustness']
        robustness_min = robustness_values.min()
        robustness_max = robustness_values.max()
        robustness_avg = robustness_values.mean()
        
        # Calculate nfev statistics for falsified runs
        if num_falsified > 0:
            falsified_nfev = falsified_runs['nfev']
            nfev_avg_falsified = falsified_nfev.mean()
            nfev_median_falsified = falsified_nfev.median()
        else:
            nfev_avg_falsified = np.nan
            nfev_median_falsified = np.nan
        
        results.append({
            'specification': spec,
            'total_runs': total_runs,
            'falsified_runs': num_falsified,
            'falsification_ratio': falsification_ratio,
            'robustness_min': robustness_min,
            'robustness_max': robustness_max,
            'robustness_avg': robustness_avg,
            'nfev_avg_falsified': nfev_avg_falsified,
            'nfev_median_falsified': nfev_median_falsified
        })
    
    return pd.DataFrame(results)


def main():
    if len(sys.argv) != 3:
        print("Usage: python analyze_results.py <optimizer_name> <benchmark_name>")
        print("Example: python analyze_results.py LLM autotrans")
        sys.exit(1)
    
    optimizer_name = sys.argv[1]
    benchmark_name = sys.argv[2]
    
    try:
        # Analyze results
        summary_df = analyze_results(optimizer_name, benchmark_name)
        
        # Create output file path
        output_dir = Path(f"examples/{benchmark_name}_all_specs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"summary_{optimizer_name}.csv"
        
        # Save results
        summary_df.to_csv(output_file, index=False, float_format='%.6f')
        print(f"Summary saved to {output_file}")
        
        # Display results
        print(f"\nSummary for {optimizer_name} on {benchmark_name}:")
        print("="*80)
        
        # Format display with better column names
        display_df = summary_df.copy()
        display_df.columns = [
            'Spec', 'Total', 'Falsified', 'Fals_Ratio', 'Rob_Min', 
            'Rob_Max', 'Rob_Avg', 'NFev_Avg_Fals', 'NFev_Med_Fals'
        ]
        
        # Round numerical columns for display
        numeric_cols = ['Fals_Ratio', 'Rob_Min', 'Rob_Max', 
                       'Rob_Avg', 'NFev_Avg_Fals', 'NFev_Med_Fals']
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(4)
        
        print(display_df.to_string(index=False))
        
        # Summary statistics
        print(f"\nOverall Statistics:")
        print(f"Total specifications: {len(summary_df)}")
        print(f"Specifications with at least one falsification: {(summary_df['falsification_ratio'] > 0).sum()}")
        print(f"Average falsification ratio: {summary_df['falsification_ratio'].mean():.4f}")
        print(f"Average robustness: {summary_df['robustness_avg'].mean():.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()