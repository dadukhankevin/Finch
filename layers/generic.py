import math
from Finch3.layers.layer import Layer
from Finch3.tools.rates import make_callable
from Finch3.genepools import GenePool
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


class KillByFitnessPercentile(Layer):
    def __init__(self, percentile: float):
        super().__init__(individual_selection=None)
        self.percentile = make_callable(percentile)

    def execute(self, individuals):
        num_to_kill = int(len(individuals) * self.percentile())
        sorted_individuals = sorted(individuals, key=lambda x: x.fitness)
        remaining_individuals = sorted_individuals[num_to_kill:]
        self.environment.individuals = remaining_individuals