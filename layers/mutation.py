import numpy as np
import Finch.layers.layer as layer
from typing import Union, Callable
from Finch.tools.rates import make_callable

cp = None
try:
    import cupy as cp
except ImportError:
    pass


class Mutate(layer.Mutate):
    def __init__(self, individual_selection: Union[float, int, Callable], gene_selection: Union[float, int, Callable],
                 refit=True):
        super().__init__(individual_selection=individual_selection, gene_selection=gene_selection,
                         refit=refit)

    def mutate_one(self, individual, environment):
        """
        Mutates a single individual
        :param environment: the environment the individuals live in
        :param individual: the individual to mutate
        :return: None
        """
        idx = self.gene_selection(individual)
        individual.genes[idx] = individual.gene_pool.generate_genes(idx.size)


class FloatMutateRange(layer.Mutate):
    """
    A float based layer that mutates genes by changing their values rather than replacing them.
    This can allow for more fine-grained mutations.
    """

    def __init__(self, individual_selection: Union[float, int, Callable], gene_selection: Union[float, int, Callable],
                 max_mutation: Union[float, int] = 1.0, min_mutation: Union[float, int] = -1.0,
                 refit=True, keep_within_genepool_bounds=False):
        """
        :param max_mutation: maximum mutation
        :param min_mutation: minimum mutation
        :param individual_selection: individual selection method
        :param gene_selection: gene selection method
        :param keep_within_genepool_bounds: keeps the values of the floats within the max
        and min set in their gene pools
        """
        super().__init__(individual_selection=individual_selection, gene_selection=gene_selection,
                         refit=refit)
        self.max_mutation = make_callable(max_mutation)
        self.min_mutation = make_callable(min_mutation)
        self.keep_within_genepool_bounds = keep_within_genepool_bounds

    def mutate_one(self, individual, environment):
        """
        Mutates a single individual
        :param environment: the environment the individuals live in
        :param individual: the individual to mutate
        :return: None
        """
        genes_to_change = self.gene_selection(individual)
        if individual.device == 'gpu':
            npcp = cp
        else:
            npcp = np

        individual.genes[genes_to_change] += npcp.random.uniform(self.max_mutation(), self.min_mutation(),
                                                                 size=genes_to_change.size)
        if self.keep_within_genepool_bounds:
            individual.genes[individual.genes > individual.gene_pool.maximum] = individual.gene_pool.maximum
            individual.genes[individual.genes < individual.gene_pool.minimum] = individual.gene_pool.minimum


class IntMutateRange(FloatMutateRange):
    """
    An integer-based layer that mutates genes by changing their values rather than replacing them.
    This can allow for more fine-grained mutations.
    """

    def __init__(self, gene_selection: Union[float, int, Callable], individual_selection: Union[float, int, Callable],
                 max_mutation: Union[int, Callable] = 1, min_mutation: Union[int, Callable] = -1,
                 refit=True, keep_within_genepool_bounds=False):
        """
        :param max_mutation: maximum mutation
        :param min_mutation: minimum mutation
        :param individual_selection: individual selection method
        :param gene_selection: gene selection method
        :param keep_within_genepool_bounds: keep within bounds of possible genes
        """
        super().__init__(max_mutation=max_mutation, min_mutation=min_mutation,
                         individual_selection=individual_selection,
                         gene_selection=gene_selection,
                         refit=refit, keep_within_genepool_bounds=keep_within_genepool_bounds)

    def mutate_one(self, individual, environment):
        """
        Mutates a single individual
        :param environment: the environment the individuals live in
        :param individual: the individual to mutate
        :return: None
        """
        if individual.device == 'gpu':
            npcp = cp
        else:
            npcp = np

        genes_to_change = self.gene_selection(individual)

        individual.genes[genes_to_change] += npcp.random.randint(self.min_mutation(), self.max_mutation() + 1,
                                                           size=genes_to_change.size)

        if self.keep_within_genepool_bounds:
            individual.genes[individual.genes > individual.gene_pool.maximum] = individual.gene_pool.maximum
            individual.genes[individual.genes < individual.gene_pool.minimum] = individual.gene_pool.minimum


class BinaryMutate(layer.Mutate):
    """
    A binary-based layer that mutates genes by changing their values to 0 or 1.
    """

    def __init__(self, individual_selection: Union[float, int, Callable], gene_selection: Union[float, int, Callable],
                 refit=True):
        super().__init__(individual_selection=individual_selection, gene_selection=gene_selection,
                         refit=refit)

    def mutate_one(self, individual, environment):
        """
        Mutates a single individual
        :param environment: the environment the individuals live in
        :param individual: the individual to mutate
        :return: None
        """
        genes_to_change = self.gene_selection(individual)

        if individual.device == 'cpu':
            individual.genes[genes_to_change] = np.random.randint(2, size=genes_to_change.size)
        if individual.device == 'gpu':
            individual.genes[genes_to_change] = cp.random.randint(2, size=genes_to_change.size)
