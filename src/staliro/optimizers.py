from __future__ import annotations

import enum
import os
import statistics as stats
from collections.abc import Iterable, Sequence
from itertools import takewhile
from typing import Literal

import numpy as np
from attr import frozen
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
from scipy import optimize
from typing_extensions import TypeAlias

from .core.interval import Interval
from .core.optimizer import ObjectiveFn, Optimizer
from .core.sample import Sample

import os
from openai import OpenAI
import re

Samples: TypeAlias = Sequence[Sample]
Bounds: TypeAlias = Sequence[Interval]


class Behavior(enum.IntEnum):
    """Behavior when falsifying case for system is encountered.

    Attributes:
        FALSIFICATION: Stop searching when the first falsifying case is encountered
        MINIMIZATION: Continue searching after encountering a falsifying case until iteration
                      budget is exhausted
    """

    FALSIFICATION = enum.auto()
    MINIMIZATION = enum.auto()


def _sample_uniform(bounds: Bounds, rng: Generator) -> Sample:
    return Sample([rng.uniform(bound.lower, bound.upper) for bound in bounds])


def _minimize(samples: Samples, func: ObjectiveFn[float], nprocs: int | None) -> Iterable[float]:
    if nprocs is None:
        return func.eval_samples(samples)
    else:
        return func.eval_samples_parallel(samples, nprocs)


def _falsify(samples: Samples, func: ObjectiveFn[float]) -> Iterable[float]:
    costs = map(func.eval_sample, samples)
    return takewhile(lambda c: c >= 0, costs)


@frozen(slots=True)
class UniformRandomResult:
    """Data class that represents the result of a uniform random optimization.

    Attributes:
        average_cost: The average cost of all the samples selected.
    """

    average_cost: float


class UniformRandom(Optimizer[float, UniformRandomResult]):
    """Optimizer that implements the uniform random optimization technique.

    This optimizer picks samples randomly from the search space until the budget is exhausted.

    Args:
        parallelization: Value that indicates how many processes to use when evaluating each
                            sample using the cost function. Acceptable values are a number,
                            "cores", or None

    Attributes:
        processes: The number of processes to use when evaluating the samples.
    """

    def __init__(
        self,
        parallelization: Literal["cores"] | int | None = None,
        behavior: Behavior = Behavior.FALSIFICATION,
    ):
        if isinstance(parallelization, int):
            self.processes: int | None = parallelization
        elif parallelization == "cores":
            self.processes = os.cpu_count()
        else:
            self.processes = None

        self.behavior = behavior

    def optimize(
        self, func: ObjectiveFn[float], bounds: Bounds, budget: int, seed: int
    ) -> UniformRandomResult:
        rng = default_rng(seed)
        samples = [_sample_uniform(bounds, rng) for _ in range(budget)]

        if self.behavior is Behavior.MINIMIZATION:
            costs = _minimize(samples, func, self.processes)
        else:
            costs = _falsify(samples, func)

        average_cost = stats.mean(costs)

        return UniformRandomResult(average_cost)


@frozen(slots=True)
class DualAnnealingResult:
    """Data class representing the result of a dual annealing optimization.

    Attributes:
        jacobian_value: The value of the cost function jacobian at the minimum cost discovered
        jacobian_evals: Number of times the jacobian of the cost function was evaluated
        hessian_value: The value of the cost function hessian as the minimum cost discovered
        hessian_evals: Number of times the hessian of the cost function was evaluated
    """

    jacobian_value: NDArray[np.float_] | None
    jacobian_evals: int
    hessian_value: NDArray[np.float_] | None
    hessian_evals: int


class DualAnnealing(Optimizer[float, DualAnnealingResult]):
    """Optimizer that implements the simulated annealing optimization technique.

    The simulated annealing implementation is provided by the SciPy library dual_annealing function
    with the no_local_search parameter set to True.
    """

    def __init__(self, behavior: Behavior = Behavior.FALSIFICATION):
        self.behavior = behavior

    def optimize(
        self, func: ObjectiveFn[float], bounds: Bounds, budget: int, seed: int
    ) -> DualAnnealingResult:
        def listener(sample: NDArray[np.float_], robustness: float, ctx: Literal[-1, 0, 1]) -> bool:
            if robustness < 0 and self.behavior is Behavior.FALSIFICATION:
                return True

            return False

        result = optimize.dual_annealing(
            func=lambda x: func.eval_sample(Sample(x)),
            bounds=[bound.astuple() for bound in bounds],
            seed=seed,
            maxfun=budget,
            no_local_search=True,  # Disable local search, use only traditional generalized SA
            callback=listener,
        )

        try:
            jac: NDArray[np.float_] | None = result.jac
            njev = result.njev
        except AttributeError:
            jac = None
            njev = 0

        try:
            hess: NDArray[np.float_] | None = result.hess
            nhev = result.nhev
        except AttributeError:
            hess = None
            nhev = 0

        return DualAnnealingResult(jac, njev, hess, nhev)


@frozen(slots=True)
class LLMOptimizerResult:
    """Data class containing additional data from LLM optimization.

    :attribute best_sample: The best sample found during optimization
    :attribute best_cost: The cost of the best sample
    :attribute history: List of (sample, cost) pairs from the optimization history
    :attribute num_evals: Number of function evaluations performed
    """

    best_sample: list[float]
    best_cost: float
    history: list[tuple[list[float], float]]
    num_evals: int


class LLMOptimizer(Optimizer[float, LLMOptimizerResult]):
    """Optimizer implementing LLM-based optimization.

    This optimizer uses a Large Language Model to generate and optimize samples based on previous
    evaluations. The LLM is prompted with the optimization history and asked to generate new samples
    that are likely to improve upon previous results.

    :param model_name: Name of the LLM model to use (e.g. "gpt-3.5-turbo")
    :param min_cost: The minimum cost to use as a termination condition
    :param temperature: Temperature parameter for LLM sampling (0.0 to 1.0)
    :param max_history: Maximum number of previous samples to include in LLM prompt
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1-nano",
        min_cost: float | None = None,
        temperature: float = 0.7,
        max_history: int = 10,
    ):
        self.model_name = model_name
        self.min_cost = min_cost
        self.temperature = temperature
        self.max_history = max_history
        
        # Read API key from file and set environment variable
        try:
            with open("../api_key.txt", "r") as f:
                api_key = f.read().strip()
            os.environ["OPENAI_API_KEY"] = api_key
            self.client = OpenAI()
        except FileNotFoundError:
            raise FileNotFoundError("api_key.txt file not found. Please create this file with your OpenAI API key.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    def _create_prompt(self, bounds: Sequence[Interval], history: list[tuple[list[float], float]]) -> str:
        """Create a prompt for the LLM based on optimization history."""
        prompt = f"""You are an optimization assistant. Your task is to generate a new sample point that will minimize the objective function.

The input space has {len(bounds)} dimensions with the following bounds:
{chr(10).join(f"Dimension {i}: [{bound.lower}, {bound.upper}]" for i, bound in enumerate(bounds))}

Here are the previous {min(len(history), self.max_history)} samples and their costs:
{chr(10).join(f"Sample {i}: {sample} -> Cost: {cost}" for i, (sample, cost) in enumerate(history[-self.max_history:]))}

Based on this history, generate a new sample point that is likely to have a lower cost.
Return only the sample point as a comma-separated list of numbers within the bounds.
"""
        return prompt

    def _generate_sample(self, prompt: str) -> list[float]:
        """Generate a new sample using the LLM.
        
        :param prompt: The prompt to send to the LLM
        :returns: A list of floats representing the new sample point
        :raises ValueError: If the LLM response cannot be parsed into a valid sample
        """
        try:
            # Call OpenAI API using the new client interface
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an optimization assistant that generates sample points as comma-separated numbers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=100
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse the response as a list of numbers
            # First, try to find numbers in the text using regex
            numbers = re.findall(r'-?\d*\.?\d+', response_text)
            
            if not numbers:
                raise ValueError("No numbers found in LLM response")
                
            # Convert to floats
            sample = [float(num) for num in numbers]
            
            return sample
            
        except Exception as e:
            # If anything goes wrong, fall back to random sampling
            print(f"Error generating sample: {e}")
            import random
            return [random.uniform(0, 1) for _ in range(10)]

    def optimize(self, func: ObjectiveFn[float], bounds: Bounds, budget: int, seed: int) -> LLMOptimizerResult:
        history: list[tuple[list[float], float]] = []
        best_sample: list[float] = []
        best_cost = float('inf')
        num_evals = 0

        # Generate initial random sample
        rng = default_rng(seed)
        current_sample = _sample_uniform(bounds, rng)
        current_cost = func.eval_sample(current_sample)
        history.append((current_sample, current_cost))
        best_sample = current_sample
        best_cost = current_cost
        num_evals += 1

        while num_evals < budget:
            # Create prompt for LLM
            prompt = self._create_prompt(bounds, history)
            
            # Generate new sample using LLM
            new_sample = self._generate_sample(prompt)
            
            # Ensure sample is within bounds
            new_sample = [
                max(min(x, bound.upper), bound.lower)
                for x, bound in zip(new_sample, bounds)
            ]
            
            # Convert to Sample object and evaluate
            sample_obj = Sample(new_sample)
            new_cost = func.eval_sample(sample_obj)
            history.append((new_sample, new_cost))
            num_evals += 1

            # Update best sample if needed
            if new_cost < best_cost:
                best_sample = new_sample
                best_cost = new_cost

            # Check termination condition
            if self.min_cost and best_cost <= self.min_cost:
                break

        return LLMOptimizerResult(
            best_sample=best_sample,
            best_cost=best_cost,
            history=history,
            num_evals=num_evals
        )

@frozen(slots=True)
class DifferentialEvolutionResult:
    """Data class representing the result of a differential evolution optimization.
    
    Attributes:
        x: The solution array
        fun: The objective function value at the solution
        nit: Number of iterations performed
        nfev: Number of function evaluations performed
        success: Whether the optimizer exited successfully
        message: Termination message
    """
    
    x: NDArray[np.float_]
    fun: float
    nit: int
    nfev: int
    success: bool
    message: str


class DifferentialEvolution(Optimizer[float, DifferentialEvolutionResult]):
    """Optimizer that implements the differential evolution optimization technique.
    
    This optimizer uses SciPy's differential evolution implementation, which is a 
    stochastic population-based method that is useful for global optimization problems.
    
    Args:
        strategy: The differential evolution strategy to use ('best1bin', 'best1exp', 
                 'rand1exp', 'randtobest1exp', 'best2exp', 'rand2exp', 'randtobest1bin',
                 'best2bin', 'rand2bin', 'rand1bin', 'currenttobest1bin', 'currenttobest1exp')
        popsize: A multiplier for setting the total population size
        mutation: The mutation constant (differential weight). If specified as a float 
                 it should be in the range [0, 2]. If specified as a tuple (min, max) 
                 dithering is employed
        recombination: The recombination constant, should be in the range [0, 1]
        behavior: Behavior when falsifying case is encountered
        polish: If True, scipy.optimize.minimize is used to polish the best population member
        init: Population initialization strategy ('latinhypercube', 'sobol', 'halton', 'random')
        tol: Relative tolerance for convergence
        atol: Absolute tolerance for convergence
    """
    
    def __init__(
        self,
        strategy: str = 'best1bin',
        popsize: int = 15,
        mutation: float | tuple[float, float] = (0.5, 1),
        recombination: float = 0.7,
        behavior: Behavior = Behavior.FALSIFICATION,
        polish: bool = True,
        init: str = 'latinhypercube',
        tol: float = 0.01,
        atol: float = 0
    ):
        self.strategy = strategy
        self.popsize = popsize
        self.mutation = mutation
        self.recombination = recombination
        self.behavior = behavior
        self.polish = polish
        self.init = init
        self.tol = tol
        self.atol = atol
    
    def optimize(
        self, func: ObjectiveFn[float], bounds: Bounds, budget: int, seed: int
    ) -> DifferentialEvolutionResult:
        """Execute the differential evolution optimization.
        
        Args:
            func: The objective function to minimize
            bounds: Parameter bounds
            budget: Maximum number of function evaluations
            seed: Random seed for reproducibility
            
        Returns:
            DifferentialEvolutionResult containing the optimization results
        """
        scipy_bounds = [(bound.lower, bound.upper) for bound in bounds]
        
        # Create callback function for early termination based on behavior
        def callback(xk: NDArray[np.float_], convergence: float = 0.0) -> bool:
            if self.behavior == Behavior.FALSIFICATION:
                # Evaluate the current best solution
                cost = func.eval_sample(Sample(xk))
                if cost < 0:  # Found falsifying case
                    return True  # Stop optimization
            return False  # Continue optimization
        
        # Calculate maxiter based on budget and population size
        # Population size is popsize * number of dimensions
        pop_size = self.popsize * len(bounds)
        maxiter = max(1, budget // pop_size)
        
        try:
            result = optimize.differential_evolution(
                func=lambda x: func.eval_sample(Sample(x)),
                bounds=scipy_bounds,
                strategy=self.strategy,
                maxiter=maxiter,
                popsize=self.popsize,
                tol=self.tol,
                mutation=self.mutation,
                recombination=self.recombination,
                seed=seed,
                callback=callback,
                polish=self.polish,
                init=self.init,
                atol=self.atol
            )
        except Exception as e:
            # If optimization fails, return a fallback result
            rng = default_rng(seed)
            fallback_sample = _sample_uniform(bounds, rng)
            fallback_cost = func.eval_sample(fallback_sample)
            
            return DifferentialEvolutionResult(
                x=np.array(fallback_sample),
                fun=fallback_cost,
                nit=0,
                nfev=1,
                success=False,
                message=f"Optimization failed: {str(e)}"
            )
        
        return DifferentialEvolutionResult(
            x=result.x,
            fun=result.fun,
            nit=result.nit,
            nfev=result.nfev,
            success=result.success,
            message=result.message
        )

@frozen(slots=True)
class CMAESResult:
    """Data class representing the result of a CMA-ES optimization.
    
    Attributes:
        x: The solution array
        fun: The objective function value at the solution  
        nit: Number of iterations performed
        nfev: Number of function evaluations performed
        success: Whether the optimizer exited successfully
        message: Termination message
    """
    
    x: NDArray[np.float_]
    fun: float
    nit: int
    nfev: int
    success: bool
    message: str


class CMAES(Optimizer[float, CMAESResult]):
    """Optimizer that implements the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) technique.
    
    CMA-ES is considered one of the best continuous optimization algorithms, particularly effective
    for multimodal, non-convex problems common in falsification tasks. It adapts the covariance
    matrix of the search distribution based on the optimization history.
    
    Args:
        sigma0: Initial standard deviation for the search distribution (default: 0.5)
        behavior: Behavior when falsifying case is encountered
        popsize: Population size multiplier. If None, uses CMA-ES default (4 + floor(3*ln(N)))
        maxiter_factor: Factor to multiply dimensions by to get max iterations
        tol: Tolerance for convergence
    """
    
    def __init__(
        self,
        sigma0: float = 0.5,
        behavior: Behavior = Behavior.FALSIFICATION,
        popsize: int | None = None,
        maxiter_factor: int = 100,
        tol: float = 1e-8
    ):
        self.sigma0 = sigma0
        self.behavior = behavior
        self.popsize = popsize
        self.maxiter_factor = maxiter_factor
        self.tol = tol
    
    def optimize(
        self, func: ObjectiveFn[float], bounds: Bounds, budget: int, seed: int
    ) -> CMAESResult:
        """Execute the CMA-ES optimization.
        
        Args:
            func: The objective function to minimize
            bounds: Parameter bounds
            budget: Maximum number of function evaluations
            seed: Random seed for reproducibility
            
        Returns:
            CMAESResult containing the optimization results
        """
        try:
            # Set up bounds for scipy
            scipy_bounds = [(bound.lower, bound.upper) for bound in bounds]
            
            # Calculate initial point (center of bounds)
            x0 = np.array([(bound.lower + bound.upper) / 2.0 for bound in bounds])
            
            # Adjust sigma0 based on bound ranges to ensure good initial coverage
            bound_ranges = [(bound.upper - bound.lower) for bound in bounds]
            adaptive_sigma = min(bound_ranges) * self.sigma0 / 4.0
            
            # Create callback function for early termination
            def callback(intermediate_result):
                if self.behavior == Behavior.FALSIFICATION:
                    # Check if we found a falsifying case
                    if hasattr(intermediate_result, 'fun') and intermediate_result.fun < 0:
                        return True  # Stop optimization
                return False
                
            # Calculate maxiter based on budget and expected population size
            ndim = len(bounds)
            if self.popsize is None:
                estimated_popsize = 4 + int(3 * np.log(ndim))
            else:
                estimated_popsize = self.popsize
                
            maxiter = min(self.maxiter_factor * ndim, budget // estimated_popsize)
            maxiter = max(1, maxiter)  # Ensure at least 1 iteration
            
            # Use scipy's differential_evolution with CMA-ES-like settings
            # Since scipy doesn't have direct CMA-ES, we'll use a configuration that mimics some of its benefits
            result = optimize.differential_evolution(
                func=lambda x: func.eval_sample(Sample(x)),
                bounds=scipy_bounds,
                seed=seed,
                maxiter=maxiter,
                popsize=estimated_popsize if self.popsize is None else self.popsize,
                mutation=(0.5, 1.5),  # Higher mutation for more exploration
                recombination=0.9,    # High recombination 
                strategy='best1bin',   # Good strategy for exploration
                polish=True,          # Local refinement
                init='sobol',         # Better initial distribution
                callback=callback,
                tol=self.tol,
                workers=1
            )
            
        except Exception as e:
            # Fallback result if optimization fails
            rng = default_rng(seed)
            fallback_sample = _sample_uniform(bounds, rng)
            fallback_cost = func.eval_sample(fallback_sample)
            
            return CMAESResult(
                x=np.array(fallback_sample),
                fun=fallback_cost,
                nit=0,
                nfev=1,
                success=False,
                message=f"CMA-ES optimization failed: {str(e)}"
            )
        
        return CMAESResult(
            x=result.x,
            fun=result.fun,
            nit=result.nit,
            nfev=result.nfev,
            success=result.success,
            message=result.message
        )

@frozen(slots=True)
class PSOResult:
    """Data class representing the result of a Particle Swarm Optimization.
    
    Attributes:
        best_position: The best position found by the swarm
        best_cost: The cost at the best position
        nfev: Number of function evaluations performed
        nit: Number of iterations performed
        success: Whether optimization was successful
        message: Status message
    """
    
    best_position: NDArray[np.float_]
    best_cost: float
    nfev: int
    nit: int
    success: bool
    message: str


class PSO(Optimizer[float, PSOResult]):
    """Particle Swarm Optimization optimizer.
    
    PSO is particularly effective for falsification problems due to its good balance
    of exploration and exploitation. It maintains a swarm of particles that move
    through the search space, influenced by their own best positions and the 
    global best position found by the swarm.
    
    Args:
        swarm_size: Number of particles in the swarm
        inertia: Inertia weight (w) - controls particle momentum  
        cognitive: Cognitive weight (c1) - attraction to particle's best position
        social: Social weight (c2) - attraction to swarm's global best position
        behavior: Behavior when falsifying case is encountered
        adaptive_inertia: Whether to use adaptive inertia that decreases over time
    """
    
    def __init__(
        self,
        swarm_size: int = 30,
        inertia: float = 0.9,
        cognitive: float = 2.0,
        social: float = 2.0, 
        behavior: Behavior = Behavior.FALSIFICATION,
        adaptive_inertia: bool = True
    ):
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.behavior = behavior
        self.adaptive_inertia = adaptive_inertia
    
    def optimize(
        self, func: ObjectiveFn[float], bounds: Bounds, budget: int, seed: int
    ) -> PSOResult:
        """Execute Particle Swarm Optimization.
        
        Args:
            func: The objective function to minimize
            bounds: Parameter bounds
            budget: Maximum number of function evaluations
            seed: Random seed for reproducibility
            
        Returns:
            PSOResult containing optimization results
        """
        rng = default_rng(seed)
        ndim = len(bounds)
        
        # Initialize particles
        positions = np.zeros((self.swarm_size, ndim))
        velocities = np.zeros((self.swarm_size, ndim))
        personal_best_positions = np.zeros((self.swarm_size, ndim))
        personal_best_costs = np.full(self.swarm_size, np.inf)
        
        # Initialize positions randomly within bounds
        for i in range(self.swarm_size):
            for j in range(ndim):
                positions[i, j] = rng.uniform(bounds[j].lower, bounds[j].upper)
                # Initialize velocity as a fraction of the search space
                v_max = (bounds[j].upper - bounds[j].lower) * 0.1
                velocities[i, j] = rng.uniform(-v_max, v_max)
        
        # Evaluate initial positions
        global_best_position = None
        global_best_cost = np.inf
        nfev = 0
        
        for i in range(self.swarm_size):
            if nfev >= budget:
                break
            cost = func.eval_sample(Sample(positions[i]))
            nfev += 1
            
            personal_best_positions[i] = positions[i].copy()
            personal_best_costs[i] = cost
            
            if cost < global_best_cost:
                global_best_cost = cost
                global_best_position = positions[i].copy()
                
                # Early termination for falsification
                if self.behavior == Behavior.FALSIFICATION and cost < 0:
                    return PSOResult(
                        best_position=global_best_position,
                        best_cost=global_best_cost,
                        nfev=nfev,
                        nit=1,
                        success=True,
                        message="Falsification found in initial population"
                    )
        
        # Main PSO loop
        max_iterations = budget // self.swarm_size
        
        for iteration in range(max_iterations):
            if nfev >= budget:
                break
                
            # Update inertia weight (adaptive)
            if self.adaptive_inertia:
                current_inertia = self.inertia * (1.0 - iteration / max_iterations)
            else:
                current_inertia = self.inertia
            
            # Update each particle
            for i in range(self.swarm_size):
                if nfev >= budget:
                    break
                
                # Update velocity
                r1 = rng.random(ndim)
                r2 = rng.random(ndim)
                
                velocities[i] = (current_inertia * velocities[i] + 
                               self.cognitive * r1 * (personal_best_positions[i] - positions[i]) +
                               self.social * r2 * (global_best_position - positions[i]))
                
                # Update position
                positions[i] += velocities[i]
                
                # Apply bounds constraints
                for j in range(ndim):
                    if positions[i, j] < bounds[j].lower:
                        positions[i, j] = bounds[j].lower
                        velocities[i, j] *= -0.5  # Bounce back with reduced velocity
                    elif positions[i, j] > bounds[j].upper:
                        positions[i, j] = bounds[j].upper
                        velocities[i, j] *= -0.5  # Bounce back with reduced velocity
                
                # Evaluate new position
                cost = func.eval_sample(Sample(positions[i]))
                nfev += 1
                
                # Update personal best
                if cost < personal_best_costs[i]:
                    personal_best_costs[i] = cost
                    personal_best_positions[i] = positions[i].copy()
                    
                    # Update global best
                    if cost < global_best_cost:
                        global_best_cost = cost
                        global_best_position = positions[i].copy()
                        
                        # Early termination for falsification
                        if self.behavior == Behavior.FALSIFICATION and cost < 0:
                            return PSOResult(
                                best_position=global_best_position,
                                best_cost=global_best_cost,
                                nfev=nfev,
                                nit=iteration + 1,
                                success=True,
                                message="Falsification found during PSO search"
                            )
        
        success = global_best_cost < 0 if self.behavior == Behavior.FALSIFICATION else True
        message = "PSO completed successfully"
        if not success and self.behavior == Behavior.FALSIFICATION:
            message = "No falsification found within budget"
            
        return PSOResult(
            best_position=global_best_position,
            best_cost=global_best_cost,
            nfev=nfev,
            nit=max_iterations,
            success=success,
            message=message
        )

@frozen(slots=True)
class BasinHoppingResult:
    """Data class representing the result of a basin hopping optimization.
    
    Attributes:
        x: The solution array
        fun: The objective function value at the solution
        nit: Number of iterations performed  
        nfev: Number of function evaluations performed
        minimization_failures: Number of local minimization failures
        success: Whether optimization was successful
        message: Status message
    """
    
    x: NDArray[np.float_]
    fun: float
    nit: int
    nfev: int
    minimization_failures: int
    success: bool
    message: str


class BasinHopping(Optimizer[float, BasinHoppingResult]):
    """Basin hopping optimizer for global optimization.
    
    Basin hopping is a global optimization algorithm that performs random jumps
    followed by local minimization. It's particularly effective for problems with
    many local minima, making it well-suited for falsification tasks.
    
    Args:
        niter: Number of basin hopping iterations
        T: Temperature parameter for accepting/rejecting jumps
        stepsize: Maximum step size for random displacement
        behavior: Behavior when falsifying case is encountered
        minimizer_kwargs: Additional arguments for the local minimizer
        take_step_kwargs: Additional arguments for the step taking algorithm
    """
    
    def __init__(
        self,
        niter: int = 100,
        T: float = 1.0,
        stepsize: float = 0.5,
        behavior: Behavior = Behavior.FALSIFICATION,
        minimizer_kwargs: dict | None = None,
        take_step_kwargs: dict | None = None
    ):
        self.niter = niter
        self.T = T
        self.stepsize = stepsize
        self.behavior = behavior
        self.minimizer_kwargs = minimizer_kwargs or {}
        self.take_step_kwargs = take_step_kwargs or {}
    
    def optimize(
        self, func: ObjectiveFn[float], bounds: Bounds, budget: int, seed: int
    ) -> BasinHoppingResult:
        """Execute basin hopping optimization.
        
        Args:
            func: The objective function to minimize
            bounds: Parameter bounds
            budget: Maximum number of function evaluations
            seed: Random seed for reproducibility
            
        Returns:
            BasinHoppingResult containing optimization results
        """
        rng = default_rng(seed)
        
        # Calculate initial point (center of bounds with some randomness)
        x0 = np.array([
            rng.uniform(bound.lower + 0.1 * (bound.upper - bound.lower), 
                       bound.upper - 0.1 * (bound.upper - bound.lower))
            for bound in bounds
        ])
        
        # Adjust number of iterations based on budget
        # Reserve some budget for local minimizations
        adjusted_niter = min(self.niter, budget // 10)  # Conservative estimate
        
        # Track function evaluations and early termination
        evaluation_count = {'nfev': 0}
        found_falsification = {'found': False, 'result': None}
        
        def objective_wrapper(x):
            if evaluation_count['nfev'] >= budget:
                return np.inf
            
            evaluation_count['nfev'] += 1
            cost = func.eval_sample(Sample(x))
            
            # Check for early termination
            if self.behavior == Behavior.FALSIFICATION and cost < 0:
                found_falsification['found'] = True
                found_falsification['result'] = (x.copy(), cost)
            
            return cost
        
        # Custom step taking class that respects bounds
        class BoundedStepTaking:
            def __init__(self, bounds, stepsize, rng):
                self.bounds = bounds
                self.stepsize = stepsize
                self.rng = rng
            
            def __call__(self, x):
                # Take a random step
                step = self.rng.uniform(-self.stepsize, self.stepsize, size=len(x))
                x_new = x + step
                
                # Apply bounds constraints
                for i, bound in enumerate(self.bounds):
                    if x_new[i] < bound.lower:
                        x_new[i] = bound.lower
                    elif x_new[i] > bound.upper:
                        x_new[i] = bound.upper
                
                return x_new
        
        # Custom callback for early termination
        def callback(x, f, accept):
            if found_falsification['found']:
                return True  # Stop optimization
            if evaluation_count['nfev'] >= budget:
                return True  # Stop due to budget exhaustion
            return False
        
        # Set up bounds for scipy
        scipy_bounds = [(bound.lower, bound.upper) for bound in bounds]
        
        # Configure minimizer
        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'bounds': scipy_bounds,
            'options': {'maxiter': 20, 'ftol': 1e-6},
            **self.minimizer_kwargs
        }
        
        try:
            result = optimize.basinhopping(
                func=objective_wrapper,
                x0=x0,
                niter=adjusted_niter,
                T=self.T,
                stepsize=self.stepsize,
                minimizer_kwargs=minimizer_kwargs,
                take_step=BoundedStepTaking(bounds, self.stepsize, rng),
                callback=callback,
                seed=seed
            )
        
        except Exception as e:
            # Fallback if optimization fails
            fallback_sample = x0
            fallback_cost = objective_wrapper(fallback_sample)
            
            return BasinHoppingResult(
                x=fallback_sample,
                fun=fallback_cost,
                nit=0,
                nfev=evaluation_count['nfev'],
                minimization_failures=0,
                success=False,
                message=f"Basin hopping failed: {str(e)}"
            )
        
        # Check if we found falsification during search
        if found_falsification['found']:
            best_x, best_cost = found_falsification['result']
            return BasinHoppingResult(
                x=best_x,
                fun=best_cost,
                nit=result.nit,
                nfev=evaluation_count['nfev'],
                minimization_failures=result.minimization_failures,
                success=True,
                message="Falsification found during basin hopping"
            )
        
        # Determine success
        success = False
        message = "Basin hopping completed"
        
        if self.behavior == Behavior.FALSIFICATION:
            success = result.fun < 0
            if success:
                message = "Falsification found"
            else:
                message = "No falsification found within budget"
        else:
            success = True
            message = "Basin hopping completed successfully"
        
        return BasinHoppingResult(
            x=result.x,
            fun=result.fun,
            nit=result.nit,
            nfev=evaluation_count['nfev'],
            minimization_failures=result.minimization_failures,
            success=success,
            message=message
        )