import argparse
import logging
import math
import random
import csv
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
import plotly.graph_objects as go
import plotly.subplots as sp

from staliro.core.interval import Interval
from staliro.core.model import Model, ModelInputs, Trace, ExtraResult
from staliro.core.result import best_eval, best_run, worst_eval, worst_run
from staliro.optimizers import DualAnnealing, LLMOptimizer, LLMGrayBoxOpt, DifferentialEvolution, PSO, BasinHopping, CMAES, UniformRandom
from staliro.options import Options
from staliro.specifications import RTAMTDense
from staliro.staliro import simulate_model, staliro
import os

from aerobench.run_f16_sim import run_f16_sim
from aerobench.examples.gcas.gcas_autopilot import GcasAutopilot

from collections import OrderedDict
from math import pi


F16DataT = NDArray[np.float_]
F16ResultT = ExtraResult[F16DataT, F16DataT]


class F16Model(Model[F16ResultT, None]):
    def __init__(self, static_params_map) -> None:
        self.F16_PARAM_MAP = static_params_map


    def get_static_params(self):
        static_params = []
        for param, config in self.F16_PARAM_MAP.items():
            if config['enabled']:
                static_params.append(config['range'])
        return static_params


    def _compute_initial_conditions(self, X):
        conditions = []
        index = 0

        for param, config in self.F16_PARAM_MAP.items():
            if config['enabled']:
                conditions.append(X[index])
                index = index + 1
            else:
                conditions.append(config['default'])

        return conditions

    def simulate(
        self, inputs: ModelInputs, intrvl: Interval
    ) -> F16ResultT:
        
        init_cond = self._compute_initial_conditions(inputs.static)
        
        step = 1 / 30
        autopilot = GcasAutopilot(init_mode="roll", stdout=False, gain_str="old")

        result = run_f16_sim(init_cond, intrvl.upper, autopilot, step, extended_states=True)
        
        # Extract state variables to match specification column expectations:
        # Column 0: mode, Column 1: roll, Column 2: pitch, Column 3: yaw, Column 4: altitude
        states = np.vstack(
            (
                np.array([0 if x == "standby" else 1 for x in result["modes"]]),  # mode: index 0
                result["states"][:, 4],  # roll: index 1
                result["states"][:, 5],  # pitch: index 2  
                result["states"][:, 6],  # yaw: index 3
                result["states"][:, 12],  # altitude: index 4
            )
        )
        
        timestamps = np.array(result["times"], dtype=(np.float32))
        outTrace = Trace(timestamps, states.tolist())
        print(inputs.static)
        inTrace = inputs.static
        return F16ResultT(outTrace, inTrace)


def analyze_violation(specification, trace, robustness_value, output_file=None):
    """
    Generalized violation analysis for any F16 specification.
    
    Args:
        specification: The specification object (RTAMTDense)
        trace: The simulation trace from best_result
        robustness_value: The computed robustness value
        output_file: Optional file path to write analysis to
    """
    times = list(trace.times)
    states = trace.states
    
    def write_output(text, file_handle=None):
        """Helper function to write to both console and file if provided"""
        print(text)
        if file_handle:
            file_handle.write(text + '\n')
    
    # Open file if provided
    file_handle = None
    if output_file:
        file_handle = open(output_file, 'a')
    
    try:
        write_output(f"\n{'='*50}", file_handle)
        write_output(f"ROBUSTNESS ANALYSIS", file_handle)
        write_output(f"{'='*50}", file_handle)
        write_output(f"Robustness value: {robustness_value:.6f}", file_handle)
        
        if robustness_value < 0:
            write_output("🚨 SPECIFICATION VIOLATED! Counterexample found.", file_handle)
            write_output(f"Violation severity: {abs(robustness_value):.6f}", file_handle)
            
            write_output(f"\nViolation analysis:", file_handle)
            write_output(f"Specification: {specification.phi}", file_handle)
            write_output(f"Variable mappings: {specification.column_map}", file_handle)
            
            # Extract variable values from trace based on column mapping
            variable_traces = {}
            for var_name, column_idx in specification.column_map.items():
                variable_traces[var_name] = [state[column_idx] for state in states]
            
            # Sample some time points to show variable values
            write_output(f"\nTrace analysis around potential violation points:", file_handle)
            if len(times) > 1:
                sample_indices = [0, len(times)//4, len(times)//2, 3*len(times)//4, len(times)-1]
                # Remove duplicate indices and ensure they're within bounds
                sample_indices = sorted(list(set([min(i, len(times)-1) for i in sample_indices])))
            else:
                sample_indices = [0]
            
            for i in sample_indices:
                if i < len(times):  # Extra safety check
                    t = times[i]
                    trace_line = f"  t={t:.2f}s:"
                    for var_name, values in variable_traces.items():
                        if i < len(values):  # Safety check for values too
                            if var_name in ["roll", "pitch", "yaw"]:
                                # Convert radians to degrees for better readability
                                trace_line += f" {var_name}={np.rad2deg(values[i]):.2f}°"
                            else:
                                trace_line += f" {var_name}={values[i]:.2f}"
                    write_output(trace_line, file_handle)
            
            # Show min/max values for each variable
            write_output(f"\nVariable ranges during simulation:", file_handle)
            for var_name, values in variable_traces.items():
                min_val, max_val = min(values), max(values)
                min_idx = values.index(min_val)
                max_idx = values.index(max_val)
                if var_name in ["roll", "pitch", "yaw"]:
                    write_output(f"  {var_name}: min={np.rad2deg(min_val):.2f}° at t={times[min_idx]:.2f}s, max={np.rad2deg(max_val):.2f}° at t={times[max_idx]:.2f}s", file_handle)
                else:
                    write_output(f"  {var_name}: min={min_val:.2f} at t={times[min_idx]:.2f}s, max={max_val:.2f} at t={times[max_idx]:.2f}s", file_handle)
        else:
            write_output("✅ Specification satisfied", file_handle)
            write_output(f"Safety margin: {robustness_value:.6f}", file_handle)
    
    finally:
        if file_handle:
            file_handle.close()


def plot_trace_variables(specification, trace, filename="f16.jpeg"):
    """
    Generalized plotting for any variables in the F16 specification.
    
    Args:
        specification: The specification object (RTAMTDense)
        trace: The simulation trace from best_result
        filename: Output filename for the plot
    """
    times = list(trace.times)
    states = trace.states
    
    # Extract variable traces based on specification mapping
    variable_traces = {}
    for var_name, column_idx in specification.column_map.items():
        if var_name in ["roll", "pitch", "yaw"]:
            # Convert radians to degrees for plotting
            variable_traces[f"{var_name} (deg)"] = [np.rad2deg(state[column_idx]) for state in states]
        else:
            variable_traces[var_name] = [state[column_idx] for state in states]
    
    # Create subplots for each variable
    num_vars = len(variable_traces)
    if num_vars > 0:
        figure = sp.make_subplots(rows=num_vars, cols=1, shared_xaxes=True, x_title="Time (s)")
        
        for i, (var_name, values) in enumerate(variable_traces.items(), 1):
            figure.add_trace(go.Scatter(x=times, y=values, name=var_name), row=i, col=1)
            figure.update_yaxes(title_text=var_name.capitalize(), row=i, col=1)
            
            # Add reference lines for key variables
            if "alt" in var_name.lower():
                figure.add_hline(y=0, line_color="red", line_dash="dash", row=i, col=1)
            elif "roll" in var_name.lower() or "pitch" in var_name.lower():
                figure.add_hline(y=0, line_color="gray", line_dash="dot", row=i, col=1)
        
        figure.write_image(filename)
        print(f"\nPlot saved as: {filename}")


#####################################################################################################################
# Define F16 Specifications

# Basic altitude safety - original specification
F16_ALT1_phi = "G[0, 15] (alt > 0)"

# Higher altitude requirements
F16_ALT2_phi = "G[0, 15] (alt > 100)"
F16_ALT3_phi = "G[0, 15] (alt > 500)"

# Roll angle constraints (±60°, ±45°, ±30°)
F16_ROLL1_phi = "G[0, 15] (abs(roll) <= 1.047)"  # ±60° in radians
F16_ROLL2_phi = "G[0, 15] (abs(roll) <= 0.785)"  # ±45° in radians  
F16_ROLL3_phi = "G[0, 15] (abs(roll) <= 0.524)"  # ±30° in radians

# Pitch angle constraints (±30°, ±20°, ±15°)
F16_PITCH1_phi = "G[0, 15] (abs(pitch) <= 0.524)"  # ±30° in radians
F16_PITCH2_phi = "G[0, 15] (abs(pitch) <= 0.349)"  # ±20° in radians
F16_PITCH3_phi = "G[0, 15] (abs(pitch) <= 0.262)"  # ±15° in radians

# Yaw angle constraints
F16_YAW1_phi = "G[0, 15] (abs(yaw) <= 0.785)"  # ±45° in radians

# Autopilot mode constraints
F16_MODE1_phi = "F[5, 15] (mode >= 1)"  # Eventually active mode
F16_MODE2_phi = "G[0, 5] (mode == 0) -> F[5, 15] (mode >= 1)"  # If standby initially, then activate

# Combined specifications
F16_SAFE1_phi = f"({F16_ALT1_phi}) and ({F16_ROLL2_phi})"  # Altitude > 0 and roll ≤ ±45°
F16_SAFE2_phi = f"({F16_ALT2_phi}) and ({F16_ROLL2_phi}) and ({F16_PITCH2_phi})"  # Multi-constraint safety

spec_dict = {
    "F16_ALT1": RTAMTDense(F16_ALT1_phi, {"alt": 4}),
    "F16_ALT2": RTAMTDense(F16_ALT2_phi, {"alt": 4}),
    "F16_ALT3": RTAMTDense(F16_ALT3_phi, {"alt": 4}),
    "F16_ROLL1": RTAMTDense(F16_ROLL1_phi, {"roll": 1}),
    "F16_ROLL2": RTAMTDense(F16_ROLL2_phi, {"roll": 1}),
    "F16_ROLL3": RTAMTDense(F16_ROLL3_phi, {"roll": 1}),
    "F16_PITCH1": RTAMTDense(F16_PITCH1_phi, {"pitch": 2}),
    "F16_PITCH2": RTAMTDense(F16_PITCH2_phi, {"pitch": 2}),
    "F16_PITCH3": RTAMTDense(F16_PITCH3_phi, {"pitch": 2}),
    "F16_YAW1": RTAMTDense(F16_YAW1_phi, {"yaw": 3}),
    "F16_MODE1": RTAMTDense(F16_MODE1_phi, {"mode": 0}),
    "F16_MODE2": RTAMTDense(F16_MODE2_phi, {"mode": 0}),
    "F16_SAFE1": RTAMTDense(F16_SAFE1_phi, {"alt": 4, "roll": 1}),
    "F16_SAFE2": RTAMTDense(F16_SAFE2_phi, {"alt": 4, "roll": 1, "pitch": 2}),
}

F16_PARAM_MAP = OrderedDict({
    'air_speed': {
        'enabled': False,
        'default': 540
    },
    'angle_of_attack': {
        'enabled': False,
        'default': np.deg2rad(2.1215)
    },
    'angle_of_sideslip': {
        'enabled': False,
        'default': 0
    },
    'roll': {
        'enabled': True,
        'default': None,
        'range': (pi / 4) + np.array((-pi / 20, pi / 30)),
    },
    'pitch': {
        'enabled': True,
        'default': None,
        'range': (-pi / 2) * 0.8 + np.array((0, pi / 20)),
    },
    'yaw': {
        'enabled': True,
        'default': None,
        'range': (-pi / 4) + np.array((-pi / 8, pi / 8)),
    },
    'roll_rate': {
        'enabled': False,
        'default': 0
    },
    'pitch_rate': {
        'enabled': False,
        'default': 0
    },
    'yaw_rate': {
        'enabled': False,
        'default': 0
    },
    'northward_displacement': {
        'enabled': False,
        'default': 0
    },
    'eastward_displacement': {
        'enabled': False,
        'default': 0
    },
    'altitude': {
        'enabled': False,
        'default': 2338.0
    },
    'engine_power_lag': {
        'enabled': False,
        'default': 9
    }
})

#####################################################################################################

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run F16 specification analysis")
    parser.add_argument(
        "-s", 
        "--spec", 
        type=str, 
        default="F16_ALT1", 
        choices=list(spec_dict.keys()),
        help="Specification to analyze (default: F16_ALT1). Available options: " + ", ".join(spec_dict.keys())
    )
    parser.add_argument(
        "-o", 
        "--optimizer", 
        type=str, 
        default="DA", 
        choices=["DA", "LLM", "LLMGB", "DE", "PSO", "BH", "CMAES", "UR"],
        help="Optimizer to use (default: DA). Available options: " + ", ".join(["DA", "LLM", "LLMGB", "DE", "PSO", "BH", "CMAES", "UR"])
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible results (default: 42)"
    )
    parser.add_argument(
        "-m",
        "--max-budget",
        type=int,
        default=100,
        help="Maximum number of iterations/budget for optimization (default: 100)"
    )
    args = parser.parse_args()
    
    # Set random seed for reproducible results
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    logging.basicConfig(level=logging.DEBUG)

    # Select specification from command line argument
    spec_name = args.spec
    specification = spec_dict[spec_name]
    
    print(f"Analyzing F16 specification: {spec_name}")
    print(f"Formula: {specification.phi}")
    print(f"Variables: {specification.column_map}")
    print(f"Random seed: {args.seed}")
    print("-" * 50)

    if args.optimizer == "LLM":
        # Create prompts filename based on specification and seed
        prompts_filename = f"./f16_all_specs/{args.optimizer}/prompts_{spec_name}_LLM_seed{args.seed}.txt"
        
        optimizer = LLMOptimizer(
            max_history=25,
            save_prompts=True,
            prompt_file=prompts_filename
        )
    elif args.optimizer == "LLMGB":
        # Define dimension descriptions for the F16 model
        # 3 static parameters: roll (PHI), pitch (THETA), yaw (PSI) initial conditions in radians
        dimension_descriptions = [
            "PHI - Initial roll angle in radians: Controls aircraft banking angle at simulation start (positive = right wing down)",
            "THETA - Initial pitch angle in radians: Controls aircraft nose up/down attitude at simulation start (positive = nose up)", 
            "PSI - Initial yaw angle in radians: Controls aircraft heading direction at simulation start (positive = nose right)"
        ]
        
        # Define output variable descriptions
        output_descriptions = {
            "mode": "Autopilot mode (0=standby, 1=active): Indicates if Ground Collision Avoidance System is engaged",
            "roll": "Aircraft roll angle in radians: Current banking angle during flight (positive = right wing down)",
            "pitch": "Aircraft pitch angle in radians: Current nose up/down attitude during flight (positive = nose up)",
            "yaw": "Aircraft yaw angle in radians: Current heading direction during flight (positive = nose right)",
            "alt": "Aircraft altitude in feet: Current height above ground level during flight"
        }
        
        # Create prompts filename based on specification and seed
        prompts_filename = f"./f16_all_specs/{args.optimizer}/prompts_{spec_name}_LLMGB_seed{args.seed}.txt"
        
        optimizer = LLMGrayBoxOpt(
            dimension_descriptions=dimension_descriptions,
            specification=specification,
            output_descriptions=output_descriptions,
            max_history=25,
            temperature=0.8,
            save_prompts=True,
            prompt_file=prompts_filename,
            include_output_states=True
        )
    elif args.optimizer == "DE":
        # Using enhanced parameters for better exploration/exploitation balance
        optimizer = DifferentialEvolution(
            strategy='best1bin',    # Good balanced strategy
            popsize=20,             # Slightly larger population for better exploration
            mutation=(0.5, 1.2),    # Slightly wider mutation range
            recombination=0.7,      # Standard crossover rate
            polish=True             # Local refinement for better solutions
        )
    elif args.optimizer == "PSO":
        # Particle Swarm Optimization - often very effective for falsification
        optimizer = PSO(
            swarm_size=30,          # Good balance of exploration and computational cost
            inertia=0.9,            # High inertia for exploration
            cognitive=2.0,          # Standard cognitive weight
            social=2.0,             # Standard social weight
            adaptive_inertia=True   # Reduces inertia over time for better convergence
        )
    elif args.optimizer == "BH":
        # Basin Hopping - combines global jumps with local optimization
        optimizer = BasinHopping(
            niter=100,              # Number of basin hopping iterations
            T=1.0,                  # Temperature for accepting jumps
            stepsize=0.5            # Step size for random jumps
        )
    elif args.optimizer == "CMAES":
        # CMA-ES-inspired optimizer using differential evolution
        optimizer = CMAES(
            sigma0=0.3,             # Initial standard deviation
            popsize=None,           # Use default CMA-ES population sizing
            maxiter_factor=50       # Conservative iteration factor
        )
    elif args.optimizer == "UR":
        optimizer = UniformRandom()
    else:
        optimizer = DualAnnealing()

    model = F16Model(F16_PARAM_MAP)
    initial_conditions = model.get_static_params()
    
    options = Options(runs=1, iterations=args.max_budget, interval=(0, 15),  static_parameters = initial_conditions, signals=[], seed=args.seed)
    result = staliro(model, specification, optimizer, options)

    # best sample has the lowest robustness value (in falsification)
    best_sample = worst_eval(worst_run(result)).sample
    best_result = simulate_model(model, options, best_sample)

    # Evaluate robustness
    robustness = specification.evaluate(best_result.trace.states, best_result.trace.times)
    
    # Check if ./f16_all_specs folder exists, if not create it
    if not os.path.exists("./f16_all_specs"):
        os.makedirs("./f16_all_specs")
        print("Created ./f16_all_specs folder")
    # Save results to CSV file
    csv_filename = f"./f16_all_specs/results_{args.optimizer}.csv"
    is_falsified = robustness < 0
    
    # Check if CSV file exists and write header if it doesn't
    file_exists = os.path.exists(csv_filename)
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['specification', 'seed', 'robustness', 'Falsified', 'nfev']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the current result
        writer.writerow({
            'specification': spec_name,
            'seed': args.seed,
            'robustness': robustness,
            'Falsified': is_falsified,
            'nfev': result.runs[0].result.nfev
        })
    
    print(f"Results saved to CSV: {csv_filename}")
    
    
    dir_path = f"./f16_all_specs/{args.optimizer}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = f"./f16_all_specs/{args.optimizer}/f16_{spec_name}_{args.optimizer}_rb{int(robustness)}"
    
    # Write additional information to text file
    txt_filename = filename + ".txt"
    with open(txt_filename, 'w') as f:
        f.write(f"F16 SPECIFICATION ANALYSIS REPORT\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Specification Name: {spec_name}\n")
        f.write(f"Optimizer: {type(optimizer).__name__}\n")
        f.write(f"Random Seed: {args.seed}\n")
        f.write(f"Simulation Interval: {options.interval}\n")
        f.write(f"Number of Runs: {options.runs}\n")
        f.write(f"Number of Iterations (Max Budget): {options.iterations}\n")
        f.write(f"Number of Function Evaluations (equals to # of objective function call & model simulation & Simulink run): {result.runs[0].result.nfev}\n\n")
        
        f.write(f"Initial Conditions (Static Parameters):\n")
        for i, param_range in enumerate(initial_conditions):
            param_names = ["PHI", "THETA", "PSI"]
            f.write(f"  {param_names[i]}: {param_range} (range in radians)\n")
        f.write(f"\n")
        
        # Show counterexample sample if violation occurred
        if robustness < 0:
            f.write(f"COUNTEREXAMPLE INPUT SAMPLE:\n")
            f.write(f"{best_sample}\n\n")
            f.write(f"Initial conditions for counterexample:\n")
            f.write(f"  PHI (roll): {best_sample.values[0]:.6f} rad ({np.rad2deg(best_sample.values[0]):.2f}°)\n")
            f.write(f"  THETA (pitch): {best_sample.values[1]:.6f} rad ({np.rad2deg(best_sample.values[1]):.2f}°)\n")
            f.write(f"  PSI (yaw): {best_sample.values[2]:.6f} rad ({np.rad2deg(best_sample.values[2]):.2f}°)\n\n")
        else:
            f.write(f"BEST SAMPLE FOUND:\n")
            f.write(f"{best_sample}\n\n")
    
    # Use generalized analysis function (appends to the file)
    analyze_violation(specification, best_result.trace, robustness, txt_filename)
    
    # Show counterexample sample if violation occurred (console output)
    if robustness < 0:
        print(f"\nCounterexample input sample: {best_sample}")
        print(f"Initial conditions for counterexample:")
        print(f"  PHI (roll): {best_sample.values[0]:.6f} rad ({np.rad2deg(best_sample.values[0]):.2f}°)")
        print(f"  THETA (pitch): {best_sample.values[1]:.6f} rad ({np.rad2deg(best_sample.values[1]):.2f}°)")
        print(f"  PSI (yaw): {best_sample.values[2]:.6f} rad ({np.rad2deg(best_sample.values[2]):.2f}°)")
    
    print(f"\nAnalysis report saved as: {txt_filename}")
    
    # Print prompts file location if using LLM optimizers
    if isinstance(optimizer, LLMGrayBoxOpt) or isinstance(optimizer, LLMOptimizer):
        print(f"LLM prompts saved as: {prompts_filename}")
        
    # Use generalized plotting function
    plot_trace_variables(specification, best_result.trace, filename+".jpeg") 