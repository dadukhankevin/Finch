from Finch.universal import ARRAY_MANAGER
from Finch.generic import Layer
from Finch.generic import GenePool
from Finch.generic import make_callable
from typing import Union, Callable


class Populate(Layer):
    def __init__(self, population: Union[Callable, int], gene_pool: GenePool):
        super().__init__(application_function=self.populate, selection_function=lambda x: x, repeat=1, refit=False)
        self.population = make_callable(population)
        self.gene_pool = gene_pool

    def populate(self, individuals):
        new_individuals = []
        needed = self.population() - len(individuals)
        if needed > 0:
            new_individuals.extend(self.gene_pool.generate_individuals(needed))
            self.environment.add_individuals(new_individuals)


class SortByFitness(Layer):
    def __init__(self):
        super().__init__(application_function=self.sort, selection_function=lambda x: x, repeat=1)

    def sort(self, individuals):
        self.environment.individuals = list(sorted(individuals, key=lambda individual: -individual.fitness))


class CapPopulation(Layer):
    def __init__(self, max_population: Union[Callable, int]):
        super().__init__(application_function=self.cap_population, selection_function=lambda x: x, repeat=1)
        self.max_population = make_callable(max_population)

    def cap_population(self, individuals):
        self.environment.individuals = individuals[:self.max_population()]
