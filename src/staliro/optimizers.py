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