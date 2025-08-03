import argparse
import logging
import random
import csv
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
import plotly.graph_objects as go
import plotly.subplots as sp

from staliro.core.interval import Interval
from staliro.core.model import BasicResult, Model, ModelInputs, ModelResult, Trace, ExtraResult
from staliro.core.result import best_eval, best_run, worst_eval, worst_run
from staliro.core.signal import Signal
from staliro.optimizers import DualAnnealing, LLMOptimizer, LLMGrayBoxOpt, DifferentialEvolution, PSO, BasinHopping, CMAES, UniformRandom
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


CCDataT = NDArray[np.float_]
CCResultT = ExtraResult[CCDataT, CCDataT]

class CCModel(Model[CCResultT, None]):
    MODEL_NAME = "cars"

    def __init__(self) -> None:
        if not _has_matlab:
            raise RuntimeError(
                "Simulink support requires the MATLAB Engine for Python to be installed"
            )

        engine = matlab.engine.start_matlab()
        # engine.addpath("examples")
        model_opts = engine.simget(self.MODEL_NAME)

        self.sampling_step = 0.05
        self.engine = engine
        self.model_opts = engine.simset(model_opts, "SaveFormat", "Array")

    def simulate(self, inputs:ModelInputs, intrvl: Interval) -> CCResultT:
        sim_t = matlab.double([0, intrvl.upper])
        n_times = (intrvl.length // self.sampling_step) + 2
        signal_times = np.linspace(intrvl.lower, intrvl.upper, int(n_times))
        signal_values = np.array([[signal.at_time(t) for t in signal_times] for signal in inputs.signals])
        
        model_input = matlab.double(np.row_stack((signal_times, signal_values)).T.tolist())

        timestamps, _, data = self.engine.sim(
            self.MODEL_NAME, sim_t, self.model_opts, model_input, nargout=3
        )

        
        data_array = np.array(data)
        y54 = (data_array[:,4]-data_array[:,3]).reshape((-1,1))
        y43 = (data_array[:,3]-data_array[:,2]).reshape((-1,1))
        y32 = (data_array[:,2]-data_array[:,1]).reshape((-1,1))
        y21 = (data_array[:,1]-data_array[:,0]).reshape((-1,1))
        diff_array = np.hstack((y21, y32, y43, y54))
        timestamps_list = np.array(timestamps).flatten()
        data_list = np.array(diff_array)
        trace = Trace(timestamps_list, data_list)

        inTrace = Trace(signal_times, signal_values)
        return CCResultT(trace, inTrace)


def generateRobustness(sample, inModel, options: Options, specification):
    result = simulate_model(inModel, options, sample)
    return specification.evaluate(result.trace.states, result.trace.times), result.extra

model = CCModel()

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

def plot_trace_variables(specification, trace, filename="cc.jpeg"):
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

CC1_phi = "G[0, 100] (y54 <= 40)"
CC2_phi = "G[0, 70] (F[0,30] (y54 >= 15))"
CC3_phi = "G[0, 80] ((G[0, 20] (y21 <= 20)) or (F[0,20] (y54 >= 40)))"
CC4_phi = "G[0,65] (F[0,30] (G[0,5] (y54 >= 8)))"
CC5_phi = "G[0,72] (F[0,8] ((G[0,5] (y21 >= 9)) -> (G[5,20] (y54 >= 9))))"

phi_1 = "(G[0, 50] (y21 >= 7.5))"
phi_2 = "(G[0, 50] (y32 >= 7.5))"
phi_3 = "(G[0, 50] (y43 >= 7.5))"
phi_4 = "(G[0, 50] (y54 >= 7.5))"
CCx_phi = phi_1 + " and " + phi_2 + " and " + phi_3 + " and " + phi_4


spec_dict = {
    "CC1": RTAMTDense(CC1_phi, {"y54": 3}),
    "CC2": RTAMTDense(CC2_phi, {"y54": 3}),
    "CC3": RTAMTDense(CC3_phi, {"y21": 0, "y54":3}),
    "CC4": RTAMTDense(CC4_phi, {"y54": 3}),    
    "CC5": RTAMTDense(CC5_phi, {"y21": 0, "y54":3}),    
    "CCx": RTAMTDense(CCx_phi,{"y21":0, "y32":1, "y43":2, "y54":3}),
    }



#####################################################################################################
# Define Signals
signals = [
    SignalOptions(control_points = [(0., 1.)] * 10, signal_times=np.linspace(0.0, 100.0, 10)),
    SignalOptions(control_points = [(0., 1.)] * 10, signal_times=np.linspace(0.0, 100.0, 10))
]

#####################################################################################################

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run CC specification analysis")
    parser.add_argument(
        "-s", 
        "--spec", 
        type=str, 
        default="CC1", 
        choices=list(spec_dict.keys()),
        help="Specification to analyze (default: CC1). Available options: " + ", ".join(spec_dict.keys())
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
    
    print(f"Analyzing specification: {spec_name}")
    print(f"Formula: {specification.phi}")
    print(f"Variables: {specification.column_map}")
    print(f"Random seed: {args.seed}")
    print("-" * 50)

    if args.optimizer == "LLM":
        # Create prompts filename based on specification and seed
        prompts_filename = f"./cc_all_specs/{args.optimizer}/prompts_{spec_name}_LLM_seed{args.seed}.txt"
        
        optimizer = LLMOptimizer(
            max_history=50,
            save_prompts=True,
            prompt_file=prompts_filename
        )
    elif args.optimizer == "LLMGB":
        # Define dimension descriptions for the CC model
        # Signal 1: 10 control points for lead car behavior (0-1 normalized throttle/brake)
        # Signal 2: 10 control points for second car behavior (0-1 normalized throttle/brake)
        dimension_descriptions = [
            "Lead car control at t=0.0s (0-1): Initial lead car throttle/brake input",
            "Lead car control at t=11.11s (0-1): Early phase lead car behavior control",
            "Lead car control at t=22.22s (0-1): Early-mid phase lead car behavior control",
            "Lead car control at t=33.33s (0-1): Mid phase lead car behavior control",
            "Lead car control at t=44.44s (0-1): Mid-late phase lead car behavior control",
            "Lead car control at t=55.56s (0-1): Late phase lead car behavior control",
            "Lead car control at t=66.67s (0-1): Very late phase lead car behavior control",
            "Lead car control at t=77.78s (0-1): Near-final phase lead car behavior control",
            "Lead car control at t=88.89s (0-1): Pre-final phase lead car behavior control",
            "Lead car control at t=100.0s (0-1): Final lead car throttle/brake input",
            "Following car control at t=0.0s (0-1): Initial following car throttle/brake input",
            "Following car control at t=11.11s (0-1): Early phase following car behavior control",
            "Following car control at t=22.22s (0-1): Early-mid phase following car behavior control",
            "Following car control at t=33.33s (0-1): Mid phase following car behavior control",
            "Following car control at t=44.44s (0-1): Mid-late phase following car behavior control",
            "Following car control at t=55.56s (0-1): Late phase following car behavior control",
            "Following car control at t=66.67s (0-1): Very late phase following car behavior control",
            "Following car control at t=77.78s (0-1): Near-final phase following car behavior control",
            "Following car control at t=88.89s (0-1): Pre-final phase following car behavior control",
            "Following car control at t=100.0s (0-1): Final following car throttle/brake input"
        ]
        
        # Define output variable descriptions
        output_descriptions = {
            "y21": "Distance between car 2 and car 1 (following distance)",
            "y32": "Distance between car 3 and car 2 (second following distance)",
            "y43": "Distance between car 4 and car 3 (third following distance)",
            "y54": "Distance between car 5 and car 4 (fourth following distance)"
        }
        
        # Create prompts filename based on specification and seed
        prompts_filename = f"./cc_all_specs/{args.optimizer}/prompts_{spec_name}_LLMGB_seed{args.seed}.txt"
        
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
    elif args.optimizer == "UR":
        optimizer = UniformRandom()
    else:
        optimizer = DualAnnealing()
    
    options = Options(runs=1, iterations=args.max_budget, interval=(0, 100), signals=signals, seed=args.seed)
    result = staliro(model, specification, optimizer, options)

    # best sample has the lowest robustness value (in falsification)
    best_sample = worst_eval(worst_run(result)).sample
    best_result = simulate_model(model, options, best_sample)

    # Evaluate robustness
    robustness = specification.evaluate(best_result.trace.states, best_result.trace.times)
    
    # Check if ./cc_all_specs folder exists, if not create it
    if not os.path.exists("./cc_all_specs"):
        os.makedirs("./cc_all_specs")
        print("Created ./cc_all_specs folder")
    # Save results to CSV file
    csv_filename = f"./cc_all_specs/results_{args.optimizer}.csv"
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
    
    
    dir_path = f"./cc_all_specs/{args.optimizer}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = f"./cc_all_specs/{args.optimizer}/cc_{spec_name}_{args.optimizer}_rb{int(robustness)}"
    
    # Write additional information to text file
    txt_filename = filename + ".txt"
    with open(txt_filename, 'w') as f:
        f.write(f"CC SPECIFICATION ANALYSIS REPORT\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Specification Name: {spec_name}\n")
        f.write(f"Optimizer: {type(optimizer).__name__}\n")
        f.write(f"Random Seed: {args.seed}\n")
        f.write(f"Simulation Interval: {options.interval}\n")
        f.write(f"Number of Runs: {options.runs}\n")
        f.write(f"Number of Iterations (Max Budget): {options.iterations}\n")
        f.write(f"Number of Function Evaluations (equals to # of objective function call & model simulation & Simulink run): {result.runs[0].result.nfev}\n\n")
        
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
    if isinstance(optimizer, LLMGrayBoxOpt) or isinstance(optimizer, LLMOptimizer):
        print(f"LLM prompts saved as: {prompts_filename}")
        
    # Use generalized plotting function
    plot_trace_variables(specification, best_result.trace, filename+".jpeg")
