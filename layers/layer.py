import random

from Finch.genetics import Individual
from Finch.tools.fitness_functions import layer_fitness
from typing import Callable, Union
from Finch.tools.individualselectors import RandomSelection
from Finch.tools.geneselectors import PercentSelector, AmountSelector
from Finch.tools.rates import make_callable

cp = None
try:
    import cupy as cp
except ImportError:
    pass


class Layer(Individual):
    """
        A Layer is a building block for an Environment and modifies or sorts the individuals in some way
    """

    def __init__(self, individual_selection: Union[float, int, Callable, None] = None,
                 gene_selection: [float, int, Callable, None] = None,
                 refit=False, fitness: int = 0, device: str = 'cpu'):
        """
        :param execution_function: retrieved from a higher class
        :param gene_selection: gene selection method: float, int, Callable
        :param individual_selection: individual selection method: float, int, Callable
        :param fitness: set to 0
        :param device: 'cpu' or 'gpu'
        """
        assert device in ('cpu', 'gpu'), f"Invalid device: {self.device}. Must be either 'cpu' or 'gpu'."
        super().__init__(genes=None, fitness=fitness, device=device, gene_pool=None)  # TODO: do layers need gene pools?
        if type(individual_selection) == float:
            assert 1 >= individual_selection >= 0, f"Selection percent mut be in the range 0-1, got: {individual_selection}"
            self.individual_selection = RandomSelection(percent_to_select=individual_selection).select
        elif type(individual_selection) == int:
            self.individual_selection = RandomSelection(amount_to_select=individual_selection).select
        elif callable(individual_selection):
            self.individual_selection = individual_selection
        else:
            self.individual_selection = lambda x: x  # make it do nothing

        if type(gene_selection) == float:
            assert 1 >= gene_selection >= 0, f"Selection percent mut be in the range 0-1, got: {gene_selection}"
            self.gene_selection = PercentSelector(gene_selection).select
        elif type(gene_selection) == int:
            self.gene_selection = AmountSelector(gene_selection).select
        elif callable(gene_selection):
            self.gene_selection = gene_selection
        else:
            self.gene_selection = None

        self.execution_function = self.execute  # TODO: remove this
        self.total_fitness_contribution = 0
        self.refit = refit
        self.environment = None

    def set_environment(self, environment):
        self.environment = environment

    def run(self, individuals: list, environment):
        """
        :param individuals: The individuals in the environment
        :param environment: The environment containing the individuals
        :return: individuals
        """
        # overall fitness of the individuals before layer execution
        before = environment.get_fitness_metric()
        # run layer on selected individuals
        selected = self.individual_selection(individuals)
        self.execution_function(individuals=selected)
        # fitness on every selected individual
        if self.refit:
            for individual in selected:
                individual.fitness = environment.fitness_function(individual)
        # overall fitness after the layer executes
        after = environment.get_fitness_metric()

        # assign the layer a fitness
        self.fitness = layer_fitness(before, after)
        self.total_fitness_contribution += before - after
        return individuals

    def execute(self, individuals):
        pass


class Mutate(Layer):
    """
    Base mutation layer, not to be used directly.
    """

    def __init__(self, gene_selection, individual_selection, device='cpu', refit=True):
        """
        :param gene_selection: gene selection method
        :param individual_selection: individual selection method
        :param execution_function: execution_function (from higher layer)
        :param device: 'cpu' or 'gpu'
        """
        super().__init__(device=device, fitness=0, individual_selection=individual_selection,
                         gene_selection=gene_selection, refit=refit)

    def mutate_one(self, individual, environment):
        pass

    def execute(self, individuals):
        """
        :param individuals: individuals to execute the mutation on
        :param environment: the environment our individuals live in (:
        :return: None
        """
        for individual in individuals:
            self.mutate_one(individual, self.environment)


class Parent(Layer):
    """
    Base parenting/crossover layer, not to be used directly.
    """

    def __init__(self, families, children=1, device='cpu', refit=True, track_genealogies=False):
        """
        :param individual_selection: individual selection method
        :param execution_function:
        :param device: 'cpu' or 'gpu'
        :param track_genealogies: bool
        """
        self.families = families
        self.children = make_callable(children)
        self.track_genealogies = track_genealogies
        super().__init__(device=device, fitness=0, individual_selection=families,
                         refit=refit)

    def parent(self, parent1, parent2, environment) -> list:
        pass

    def execute(self, individuals):  # TODO: fix this all up to be good
        assert len(individuals) % 2 == 0, f"Must be an even number parents! Got: '{len(individuals)}'"
        parents = list(zip(individuals[::2], individuals[1::2]))
        for parent1, parent2 in parents:
            children = self.parent(parent1=parent1, parent2=parent2, environment=self.environment)
            if self.track_genealogies:
                for child in children:
                    child.parents.append(parent1)
                    child.parents.append(parent2)
            self.environment.individuals += children
