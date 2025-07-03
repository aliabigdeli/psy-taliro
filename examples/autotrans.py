import logging
from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp

from staliro.core.interval import Interval
from staliro.core.model import BasicResult, Model, ModelInputs, ModelResult, Trace
from staliro.core.result import best_eval, best_run
from staliro.core.signal import Signal
from staliro.optimizers import DualAnnealing, LLMOptimizer
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


model = AutotransModel()

# phi = "always[0,30] (rpm >= 3000) -> (always[0,4] speed >= 35)"
phi = "(always[0, 30] (rpm <= 3000)) -> (always[0,4] (speed <= 35))"
specification = RTAMTDiscrete(phi, {"rpm": 1, "speed": 0})

# optimizer = LLMOptimizer()  # DualAnnealing()
optimizer = DualAnnealing()

signals = [
    SignalOptions(control_points=[(0, 100)] * 7),
    SignalOptions(control_points=[(0, 350)] * 3),
]
options = Options(runs=1, iterations=100, interval=(0, 30), signals=signals)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    result = staliro(model, specification, optimizer, options)

    best_sample = best_eval(best_run(result)).sample
    best_result = simulate_model(model, options, best_sample)
    
    # Evaluate robustness
    robustness = specification.evaluate(best_result.trace.states, best_result.trace.times)
    print(f"\n{'='*50}")
    print(f"ROBUSTNESS ANALYSIS")
    print(f"{'='*50}")
    print(f"Robustness value: {robustness:.6f}")
    
    if robustness < 0:
        print("🚨 SPECIFICATION VIOLATED! Counterexample found.")
        print(f"Violation severity: {abs(robustness):.6f}")
        
        print(f"\nCounterexample input sample: {best_sample}")
        
        # Analyze the violation in detail
        times = list(best_result.trace.times)
        rpm = [state[1] for state in best_result.trace.states]
        speed = [state[0] for state in best_result.trace.states]
        
        print(f"\nViolation analysis:")
        print(f"Specification: (always[0,30] (rpm <= 3000)) -> (always[0,4] (speed <= 35))")
        
        violation_found = False
        for i, (t, r, s) in enumerate(zip(times, rpm, speed)):
            if t <= 30 and r <= 3000 and t <= 4 and s > 35:
                if not violation_found:
                    print(f"First violation point:")
                    violation_found = True
                print(f"  t={t:.2f}s: RPM={r:.1f} ≤ 3000, Speed={s:.1f} > 35")
                break
    else:
        print("✅ Specification satisfied")
        print(f"Safety margin: {robustness:.6f}")

    # Continue with plotting...
    times = list(best_result.trace.times)
    rpm = [state[1] for state in best_result.trace.states]
    speed = [state[0] for state in best_result.trace.states]

    figure = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, x_title="Time (s)")
    figure.add_trace(go.Scatter(x=times, y=rpm), row=1, col=1)
    figure.add_trace(go.Scatter(x=times, y=speed), row=2, col=1)
    figure.update_yaxes(title_text="RPM", row=1, col=1)
    figure.update_yaxes(title_text="Speed", row=2, col=1)
    figure.write_image("autotrans.jpeg")

    # Get all evaluations from the best run
    best_run_data = best_run(result)
    print(f"\nOptimization history: {len(best_run_data.history)} evaluations")

    # Find all violations (if any)
    violations = []
    for eval_data in best_run_data.history:
        if eval_data.cost < 0:  # cost is the robustness value
            violations.append((eval_data.sample, eval_data.cost))

    print(f"Total violations found: {len(violations)}")
    if violations:
        # Sort by severity (most negative first)
        violations.sort(key=lambda x: x[1])
        print(f"Most severe violation: robustness = {violations[0][1]:.6f}")
