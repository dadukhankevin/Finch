from typing import Union
from Finch.tools.individualselectors import RankBasedSelection


class Fitness:
    def __init__(self):
        """Initialize a generic Fitness object."""
        pass

    def fit(self, individual):
        """Placeholder method for applying fitness to an individual."""
        pass


class FitnessChain(Fitness):
    def __init__(self, functions: list[tuple[callable, Union[float, int]]]):
        """
        Initialize a FitnessChain object.

        Parameters:
        - functions: List of tuples with a callable fitness function and its associated threshold.
        """
        super().__init__()
        self.functions = functions
        self.index = 0

    def fit(self, individual):
        """Apply the current fitness function to an individual."""
        self.functions[self.index][0](individual)

    def callback(self, environment):
        """Update the index if the current individual's fitness exceeds the threshold."""
        if environment.individuals[-1].fitness > self.functions[self.index][1]:
            self.index += 1


class MixFitnessByFactor(Fitness):
    """
    Combine fitness functions with weights (summing up to one) to calculate individual fitness.
    """

    def __init__(self, functions, weights, track_fitness=False):
        """
        Initialize a MixFitness object.

        Parameters:
        - functions: List of fitness functions to be combined.
        - weights: List of weights corresponding to each fitness function (must add up to one).
        - track_fitness: Bool, if true will track each functions fitness histories
        """
        super().__init__()
        self.functions = functions
        self.weights = weights
        self.track_fitness = track_fitness
        self.history = [[0] * len(functions)]

    def fit(self, individual):
        """
        Calculate the combined fitness of an individual using provided functions and weights.

        Parameters:
        - individual: The individual for which combined fitness is calculated.

        Returns:
        - float: The combined fitness value.
        """
        if len(self.functions) != len(self.weights):
            raise ValueError("Number of functions must be equal to the number of weights.")

        total_fitness = 0.0
        ind = 0
        for func, weight in zip(self.functions, self.weights):
            new_fitness = weight * func(individual)
            if self.track_fitness:
                self.history[ind].append(new_fitness)
            total_fitness += new_fitness
            ind += 1

        return total_fitness


class MixFitness(Fitness):
    def __init__(self, functions, criteria=min, track_fitness=False):
        """
        Initialize a MixFitness object.

        Parameters:
        - functions: List of fitness functions to be combined.
        - criteria: The aggregation criteria to combine fitness values, either `min` or `max`.
                    Defaults to `min`.
        - track_fitness: Bool, if true will track each function's fitness histories.
        """
        super().__init__()
        self.functions = functions
        self.criteria = criteria
        self.track_fitness = track_fitness
        self.history = [[] for _ in range(len(functions))] if track_fitness else None

    def fit(self, individual):
        """
        Calculate the combined fitness of an individual using provided functions and aggregation criteria.

        Parameters:
        - individual: The individual for which combined fitness is calculated.

        Returns:
        - float: The combined fitness value.
        """
        scores = [function(individual) for function in self.functions]

        if self.track_fitness:
            for i, score in enumerate(scores):
                self.history[i].append(score)

        return self.criteria(scores)


