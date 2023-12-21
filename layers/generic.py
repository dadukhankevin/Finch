import math

from Finch.genetics import Individual
from Finch.layers.layer import Layer
from Finch.tools.rates import make_callable
from Finch.genepools import GenePool
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


class Populate(Layer):
    def __init__(self, gene_pool: GenePool, population: int):
        """
        :param gene_pool: any GenePool
        :param population: amount of individuals to ensure the population has
        """
        super().__init__(individual_selection=None)
        self.gene_pool = gene_pool
        self.population = make_callable(population)

    def execute(self, individuals):
        individuals = list(individuals)  # TODO fix the need for this... Is this fixed???? I have no idea
        while len(individuals) < self.population():
            new = self.gene_pool.generate_individual()
            new.fitness = self.environment.fitness_function(new)
            individuals += [new]
        self.environment.individuals = individuals


class KillBySelection(Layer):
    def __init__(self, individual_selection):
        """
        :param individual_selection: the method by witch to select the individuals for death
        """
        super().__init__(individual_selection=individual_selection)

    def execute(self, individuals):
        # select some individuals to kill using the selection function
        # remove the selected individuals from the population
        for ind in individuals:
            ind.genes = []
            self.environment.dead_individuals.append(ind)
        individuals = [ind for ind in self.environment.individuals if ind not in individuals]
        self.environment.individuals = individuals


class DuplicateSelection(Layer):
    def __init__(self, individual_selection):
        """
        :param individual_selection:
        """
        super().__init__(individual_selection=individual_selection)

    def execute(self, individuals):
        # select some individuals to duplicate using the selection function

        duplicates = [d.copy() for d in individuals]  # make them different (:
        individuals += duplicates


class RemoveDuplicatesFromTop(Layer):
    def __init__(self, top_n: int):
        super().__init__(individual_selection=None)
        self.top_n = make_callable(top_n)

    def execute(self, individuals):
        # remove any duplicates from the top n individuals
        unique_individuals = individuals[:self.top_n()]
        if self.environment.device == 'cpu':
            npcp = np
        else:
            npcp = cp

        for i in range(self.top_n, len(individuals)):
            if not any(npcp.array_equal(ind.genes, individuals[i].genes) for ind in unique_individuals):
                unique_individuals.append(individuals[i])
        return unique_individuals


class SortByFitness(Layer):
    def __init__(self):
        super().__init__(individual_selection=None)

    def execute(self, individuals):
        sorted_individuals = sorted(individuals, key=lambda x: -x.fitness)
        self.environment.individuals = list(sorted_individuals)


class CapPopulation(Layer):
    def __init__(self, max_population: int):
        super().__init__(individual_selection=None)
        self.max_population = make_callable(max_population)

    def execute(self, individuals):
        self.environment.individuals = individuals[
                                       0:self.max_population()]  # kills only the worst ones assuming they are sorted

        for i in individuals[self.max_population():-1]:
            i.genes = []
            self.environment.dead_individuals.append(i)


class BatchFitness(Layer):
    """
    Layer applying a batch fitness function to selected individuals.

    Parameters:
    - batch_fitness_function: Callable, batch fitness function to be applied.

    Attributes:
    - batch_fitness_function: Callable, the batch fitness function.

    Methods:
    - execute(individuals: List[Individual]) -> None:
        Apply batch fitness function to selected individuals.
    """

    def __init__(self, batch_fitness_function):
        super().__init__()
        self.batch_fitness_function = batch_fitness_function

    def execute(self, individuals: list) -> None:
        """
        Apply batch fitness function to selected individuals.

        Parameters:
        - individuals: List[Individual], individuals to process.

        Returns:
        - None
        """
        selected = [individual for individual in individuals if individual.check_fitness]
        self.batch_fitness_function(selected)
        for i in selected:
            i.check_fitness = False


class Function(Layer):
    def __init__(self, function, individual_selection=None):
        super().__init__(individual_selection=individual_selection)
        self.function = function

    def execute(self, individuals):
        return self.function(individuals)


class Controller(Layer):
    def __init__(self, layer: Layer, execute_every=1, repeat=1, delay=0, stop_at=math.inf):
        super().__init__(individual_selection=None)
        self.layer = layer
        self.every = make_callable(execute_every)
        self.delay = make_callable(delay)
        self.end = stop_at
        self.repeat = make_callable(repeat)
        self.n = 0

    def execute(self, individuals):
        self.n += 1
        if self.delay() <= self.n <= self.end:
            if self.every() % self.n == 0:
                for i in range(self.repeat() + 1):
                    individuals = self.layer.execute(individuals)
        return individuals


class RemoveAllButBest(Layer):
    def __init__(self):
        super().__init__(individual_selection=None)

    def execute(self, individuals):
        for individual in individuals[1:-1]:
            individual.genes = [] # saves on memory, but retains everything else important about individuals
            self.environment.dead_individuals.append(individual)
        self.environment.individuals = [individuals[0]]


#
# class FreezeRandom(Layer):
#     def __init__(self, amount_genes: int, selection_function: callable = randomSelect):
#         super().__init__(individual_selection=individual_selection)
#         self.amount_genes = make_callable(amount_genes)
#
#     def execute(self, individuals):
#         selected_individuals = self.selection_function.select(individuals)
#         if environment.device == 'cpu':
#             npcp = np
#         else:
#             npcp = cp
#         for individual in selected_individuals:
#             random_indices = npcp.random.choice(individual.genes.size, self.amount_genes(), replace=False)
#             individual.freeze(random_indices)
#         return individuals


