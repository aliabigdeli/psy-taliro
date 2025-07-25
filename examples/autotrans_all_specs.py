import argparse
import logging
import random
import csv
from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp

from staliro.core.interval import Interval
from staliro.core.model import BasicResult, Model, ModelInputs, ModelResult, Trace
from staliro.core.result import best_eval, best_run, worst_eval, worst_run
from staliro.core.signal import Signal
from staliro.optimizers import DualAnnealing, LLMOptimizer, LLMGrayBoxOpt, DifferentialEvolution, PSO, BasinHopping, CMAES
from staliro.options import Options, SignalOptions
from staliro.specifications import RTAMTDiscrete, RTAMTDense
from staliro.staliro import simulate_model, staliro
import os

try:
    import matlab
    import matlab.engine
except ImportError:
    _has_matlab = False
else:
    _has_matlab = True


StaticInput = Sequence[float]
Signals = Sequence[Signal]
AutotransResultT = ModelResult[list[float], None]


class AutotransModel(Model[list[float], None]):
    MODEL_NAME = "Autotrans_shift"

    def __init__(self) -> None:
        if not _has_matlab:
            raise RuntimeError(
                "Simulink support requires the MATLAB Engine for Python to be installed"
            )

        engine = matlab.engine.start_matlab()
        engine.addpath(os.path.dirname(os.path.abspath(__file__)))
        model_opts = engine.simget(self.MODEL_NAME)

        self.sampling_step = 0.2
        self.engine = engine
        self.model_opts = engine.simset(model_opts, "SaveFormat", "Array")

    def simulate(self, inputs: ModelInputs, interval: Interval) -> BasicResult[list[float]]:
        sim_t = matlab.double([0, interval.upper])
        n_times = interval.length // self.sampling_step
        signal_times = np.linspace(interval.lower, interval.upper, int(n_times))
        signal_values = np.array(
            [[signal.at_time(t) for t in signal_times] for signal in inputs.signals]
        )
        model_input = matlab.double(np.row_stack((signal_times, signal_values)).T.tolist())

        timestamps, _, data = self.engine.sim(
            self.MODEL_NAME, sim_t, self.model_opts, model_input, nargout=3
        )

        timestamps_list: list[float] = np.array(timestamps).flatten().tolist()
        # data_list: list[list[float]] = list(data)
        # Convert MATLAB data to proper 2D format where each row is a state variable
        data_array = np.array(data)
        if data_array.ndim == 2 and data_array.shape[0] != len(timestamps_list):
            # Transpose if needed so that rows are state variables and columns are time points
            data_array = data_array.T
        data_list: list[list[float]] = data_array.tolist()
        trace = Trace(timestamps_list, data_list)

        return BasicResult(trace)

def generateRobustness(sample, inModel, options: Options, specification):
    result = simulate_model(inModel, options, sample)
    return specification.evaluate(result.trace.states, result.trace.times), result.extra

model = AutotransModel()

def analyze_violation(specification, trace, robustness_value, output_file=None):
    """
    Generalized violation analysis for any specification.
    
    Args:
        specification: The specification object (RTAMTDense/RTAMTDiscrete)
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
            sample_indices = [0, len(times)//4, len(times)//2, 3*len(times)//4, -1]
            for i in sample_indices:
                t = times[i]
                trace_line = f"  t={t:.2f}s:"
                for var_name, values in variable_traces.items():
                    trace_line += f" {var_name}={values[i]:.2f}"
                write_output(trace_line, file_handle)
            
            # Show min/max values for each variable
            write_output(f"\nVariable ranges during simulation:", file_handle)
            for var_name, values in variable_traces.items():
                min_val, max_val = min(values), max(values)
                min_idx = values.index(min_val)
                max_idx = values.index(max_val)
                write_output(f"  {var_name}: min={min_val:.2f} at t={times[min_idx]:.2f}s, max={max_val:.2f} at t={times[max_idx]:.2f}s", file_handle)
        else:
            write_output("✅ Specification satisfied", file_handle)
            write_output(f"Safety margin: {robustness_value:.6f}", file_handle)
    
    finally:
        if file_handle:
            file_handle.close()

def plot_trace_variables(specification, trace, filename="autotrans.jpeg"):
    """
    Generalized plotting for any variables in the specification.
    
    Args:
        specification: The specification object (RTAMTDense/RTAMTDiscrete)
        trace: The simulation trace from best_result
        filename: Output filename for the plot
    """
    times = list(trace.times)
    states = trace.states
    
    # Extract variable traces based on specification mapping
    variable_traces = {}
    for var_name, column_idx in specification.column_map.items():
        variable_traces[var_name] = [state[column_idx] for state in states]
    
    # Create subplots for each variable
    num_vars = len(variable_traces)
    if num_vars > 0:
        figure = sp.make_subplots(rows=num_vars, cols=1, shared_xaxes=True, x_title="Time (s)")
        
        for i, (var_name, values) in enumerate(variable_traces.items(), 1):
            figure.add_trace(go.Scatter(x=times, y=values, name=var_name), row=i, col=1)
            figure.update_yaxes(title_text=var_name.capitalize(), row=i, col=1)
        
        figure.write_image(filename)
        print(f"\nPlot saved as: {filename}")

#####################################################################################################################
# Define Specifications (all specifications)

AT1_phi = "G[0, 20] (speed <= 120)"
# AT0_phi = "G[0, 20] (speed <= 30)"

AT2_phi = "G[0, 10] (rpm <= 4750)"

gear_1_phi = f"(gear <= 1.5 and gear >= 0.5)"
AT51_phi = f"G[0, 30] (((not {gear_1_phi}) and (F[0.001,0.1] {gear_1_phi})) -> (F[0.001, 0.1] (G[0,2.5] {gear_1_phi})))"

gear_2_phi = f"(gear <= 2.5 and gear >= 1.5)"
AT52_phi = f"G[0, 30] (((not {gear_2_phi}) and (F[0.001,0.1] {gear_2_phi})) -> (F[0.001, 0.1] (G[0,2.5] {gear_2_phi})))"

gear_3_phi = f"(gear <= 3.5 and gear >= 2.5)"
AT53_phi = f"G[0, 30] (((not {gear_3_phi}) and (F[0.001,0.1] {gear_3_phi})) -> (F[0.001, 0.1] (G[0,2.5] {gear_3_phi})))"

gear_4_phi = f"(gear <= 4.5 and gear >= 3.5)"
AT54_phi = f"G[0, 30] (((not {gear_4_phi}) and (F[0.001,0.1] {gear_4_phi})) -> (F[0.001, 0.1] (G[0,2.5] {gear_4_phi})))"

# AT6_0_phi = "((G[0, 30] (rpm <= 4500)) -> (G[0,4] (speed <= 20)))"
AT6a_phi = "((G[0, 30] (rpm <= 3000)) -> (G[0,4] (speed <= 35)))"
AT6b_phi = "((G[0, 30] (rpm <= 3000)) -> (G[0,8] (speed <= 50)))"
AT6c_phi = "((G[0, 30] (rpm <= 3000)) -> (G[0,20] (speed <= 65)))"
AT6abc_phi = f"{AT6a_phi} and {AT6b_phi} and {AT6c_phi}"

spec_dict = {
    # "AT0": RTAMTDense(AT0_phi, {"speed": 0}),
    "AT1": RTAMTDense(AT1_phi, {"speed": 0}),
    "AT2": RTAMTDense(AT2_phi, {"rpm": 1}),
    "AT51": RTAMTDense(AT51_phi, {"gear": 2}),    
    "AT52": RTAMTDense(AT52_phi, {"gear": 2}),    
    "AT53": RTAMTDense(AT53_phi, {"gear": 2}),    
    "AT54": RTAMTDense(AT54_phi, {"gear": 2}),    
    # "AT61-0": RTAMTDense(AT6_0_phi, {"speed": 0, "rpm":1}),
    "AT61": RTAMTDense(AT6a_phi, {"speed": 0, "rpm":1}),
    "AT62": RTAMTDense(AT6b_phi, {"speed": 0, "rpm":1}),
    "AT63": RTAMTDense(AT6c_phi, {"speed": 0, "rpm":1}),
    "AT64": RTAMTDense(AT6abc_phi, {"speed": 0, "rpm":1}),
    }

#####################################################################################################
# Define Signals
signals = [
    SignalOptions(control_points = [(0, 100)]*7, signal_times=np.linspace(0.,50.,7)),
    SignalOptions(control_points = [(0, 325)]*3, signal_times=np.linspace(0.,50.,3)),
]

#####################################################################################################

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Autotrans specification analysis")
    parser.add_argument(
        "-s", 
        "--spec", 
        type=str, 
        default="AT61", 
        choices=list(spec_dict.keys()),
        help="Specification to analyze (default: AT61). Available options: " + ", ".join(spec_dict.keys())
    )
    parser.add_argument(
        "-o", 
        "--optimizer", 
        type=str, 
        default="DA", 
        choices=["DA", "LLM", "LLMGB", "DE", "PSO", "BH", "CMAES"],
        help="Optimizer to use (default: DA). Available options: " + ", ".join(["DA", "LLM", "LLMGB", "DE", "PSO", "BH", "CMAES"])
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible results (default: 42)"
    )
    args = parser.parse_args()
    
    # Set random seed for reproducible results
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    logging.basicConfig(level=logging.DEBUG)

    # Check if ./autotrans_all_specs folder exists, if not create it
    if not os.path.exists("./autotrans_all_specs"):
        os.makedirs("./autotrans_all_specs")
        print("Created ./autotrans_all_specs folder")

    # Select specification from command line argument
    spec_name = args.spec
    specification = spec_dict[spec_name]
    
    print(f"Analyzing specification: {spec_name}")
    print(f"Formula: {specification.phi}")
    print(f"Variables: {specification.column_map}")
    print(f"Random seed: {args.seed}")
    print("-" * 50)

    if args.optimizer == "LLM":
        # Create prompts filename based on specification and seed
        prompts_filename = f"./autotrans_all_specs/prompts_{spec_name}_LLM_seed{args.seed}.txt"
        
        optimizer = LLMOptimizer(
            max_history=100,
            save_prompts=True,
            prompt_file=prompts_filename
        )
    elif args.optimizer == "LLMGB":
        # Define dimension descriptions for the autotrans model
        # Signal 1: 7 throttle control points at different time intervals (0-100%)
        # Signal 2: 3 brake control points at different time intervals (0-325 units)
        dimension_descriptions = [
            "Throttle level at t=0.0s (0-100%): Initial acceleration input",
            "Throttle level at t=8.33s (0-100%): Early phase acceleration control", 
            "Throttle level at t=16.67s (0-100%): Mid-early phase acceleration control",
            "Throttle level at t=25.0s (0-100%): Mid phase acceleration control",
            "Throttle level at t=33.33s (0-100%): Mid-late phase acceleration control",
            "Throttle level at t=41.67s (0-100%): Late phase acceleration control",
            "Throttle level at t=50.0s (0-100%): Final acceleration input",
            "Brake pressure at t=0.0s (0-325 units): Initial braking force",
            "Brake pressure at t=25.0s (0-325 units): Mid-simulation braking force", 
            "Brake pressure at t=50.0s (0-325 units): Final braking force"
        ]
        
        # Define output variable descriptions
        output_descriptions = {
            "speed": "Vehicle speed in mph - how fast the car is traveling",
            "rpm": "Engine RPM (revolutions per minute) - engine rotational speed", 
            "gear": "Current transmission gear (1=first, 2=second, 3=third, 4=fourth gear)"
        }
        
        # Create prompts filename based on specification and seed
        prompts_filename = f"./autotrans_all_specs/prompts_{spec_name}_LLMGB_seed{args.seed}.txt"
        
        optimizer = LLMGrayBoxOpt(
            dimension_descriptions=dimension_descriptions,
            specification=specification,
            output_descriptions=output_descriptions,
            max_history=50,
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
    else:
        optimizer = DualAnnealing()
    
    options = Options(runs=1, iterations=100, interval=(0, 50), signals=signals, seed=args.seed)
    result = staliro(model, specification, optimizer, options)

    # best sample has the lowest robustness value (in falsification)
    best_sample = worst_eval(worst_run(result)).sample
    best_result = simulate_model(model, options, best_sample)

    # Evaluate robustness
    robustness = specification.evaluate(best_result.trace.states, best_result.trace.times)
    
    # Save results to CSV file
    csv_filename = f"./autotrans_all_specs/results_{args.optimizer}.csv"
    is_falsified = robustness < 0
    
    # Check if CSV file exists and write header if it doesn't
    file_exists = os.path.exists(csv_filename)
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['specification', 'seed', 'robustness', 'Falsified']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the current result
        writer.writerow({
            'specification': spec_name,
            'seed': args.seed,
            'robustness': robustness,
            'Falsified': is_falsified
        })
    
    print(f"Results saved to CSV: {csv_filename}")
    
    if isinstance(optimizer, DualAnnealing):
        filename = f"./autotrans_all_specs/autotrans_{spec_name}_DA_rb{int(robustness)}"
    elif isinstance(optimizer, LLMGrayBoxOpt):
        filename = f"./autotrans_all_specs/autotrans_{spec_name}_LLMGB_rb{int(robustness)}"
    elif isinstance(optimizer, LLMOptimizer):
        filename = f"./autotrans_all_specs/autotrans_{spec_name}_LLM_rb{int(robustness)}"
    elif isinstance(optimizer, DifferentialEvolution):
        filename = f"./autotrans_all_specs/autotrans_{spec_name}_DE_rb{int(robustness)}"
    elif isinstance(optimizer, PSO):
        filename = f"./autotrans_all_specs/autotrans_{spec_name}_PSO_rb{int(robustness)}"
    elif isinstance(optimizer, BasinHopping):
        filename = f"./autotrans_all_specs/autotrans_{spec_name}_BH_rb{int(robustness)}"
    elif isinstance(optimizer, CMAES):
        filename = f"./autotrans_all_specs/autotrans_{spec_name}_CMAES_rb{int(robustness)}"
    else:
        filename = f"./autotrans_all_specs/autotrans_{spec_name}_rb{int(robustness)}"
    
    # Write additional information to text file
    txt_filename = filename + ".txt"
    with open(txt_filename, 'w') as f:
        f.write(f"AUTOTRANS SPECIFICATION ANALYSIS REPORT\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Specification Name: {spec_name}\n")
        f.write(f"Optimizer: {type(optimizer).__name__}\n")
        f.write(f"Random Seed: {args.seed}\n")
        f.write(f"Simulation Interval: {options.interval}\n")
        f.write(f"Number of Runs: {options.runs}\n")
        f.write(f"Number of Iterations: {options.iterations}\n")
        f.write(f"Total Optimizer Calls: {options.runs * options.iterations}\n\n")
        
        # Show counterexample sample if violation occurred
        if robustness < 0:
            f.write(f"COUNTEREXAMPLE INPUT SAMPLE:\n")
            f.write(f"{best_sample}\n\n")
        else:
            f.write(f"BEST SAMPLE FOUND:\n")
            f.write(f"{best_sample}\n\n")
    
    # Use generalized analysis function (appends to the file)
    analyze_violation(specification, best_result.trace, robustness, txt_filename)
    
    # Show counterexample sample if violation occurred (console output)
    if robustness < 0:
        print(f"\nCounterexample input sample: {best_sample}")
    
    print(f"\nAnalysis report saved as: {txt_filename}")
    
    # Print prompts file location if using LLM optimizers
    if isinstance(optimizer, LLMGrayBoxOpt):
        prompts_filename = f"./autotrans_all_specs/prompts_{spec_name}_LLMGB_seed{args.seed}.txt"
        print(f"LLM prompts saved as: {prompts_filename}")
    elif isinstance(optimizer, LLMOptimizer):
        prompts_filename = f"./autotrans_all_specs/prompts_{spec_name}_LLM_seed{args.seed}.txt"
        print(f"LLM prompts saved as: {prompts_filename}")
        
    # Use generalized plotting function
    plot_trace_variables(specification, best_result.trace, filename+".jpeg")
