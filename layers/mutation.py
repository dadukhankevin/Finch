import numpy as np
import Finch3.layers.layer as layer
from typing import Union, Callable
from Finch3.tools.rates import make_callable

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
        genes_to_change = self.gene_selection(individual)
        genes_to_change[:] = individual.gene_pool.generate_genes(len(genes_to_change))


class FloatMutateRange(layer.Mutate):
    """
    A float based layer that mutates genes by changing their values rather than replacing them.
    This can allow for more fine-grained mutations.
    """

    def __init__(self, max_mutation: Union[float, int], min_mutation: Union[float, int],
                 individual_selection: Union[float, int, Callable], gene_selection: Union[float, int, Callable],
                 refit=True):
        """
        :param max_mutation: maximum mutation
        :param min_mutation: minimum mutation
        :param individual_selection: individual selection method
        :param gene_selection: gene selection method
        """
        super().__init__(individual_selection=individual_selection, gene_selection=gene_selection,
                         refit=refit)
        self.max_mutation = make_callable(max_mutation)
        self.min_mutation = make_callable(min_mutation)

    def mutate_one(self, individual, environment):
        """
        Mutates a single individual
        :param environment: the environment the individuals live in
        :param individual: the individual to mutate
        :return: None
        """
        genes_to_change = self.gene_selection(individual)

        if individual.device == 'cpu':
            genes_to_change += np.random.uniform(self.max_mutation(), self.min_mutation(),
                                                 size=genes_to_change.size)
        if individual.device == 'gpu':
            genes_to_change += cp.random.uniform(self.max_mutation(), self.min_mutation(),
                                                 size=genes_to_change.size)


class IntMutateRange(FloatMutateRange):
    """
    An integer-based layer that mutates genes by changing their values rather than replacing them.
    This can allow for more fine-grained mutations.
    """

    def __init__(self, gene_selection: Union[float, int, Callable], individual_selection: Union[float, int, Callable],
                 max_mutation: Union[int, Callable] = 1, min_mutation: Union[int, Callable] = -1,
                 refit=True):
        """
        :param max_mutation: maximum mutation
        :param min_mutation: minimum mutation
        :param individual_selection: individual selection method
        :param gene_selection: gene selection method
        """
        super().__init__(max_mutation=max_mutation, min_mutation=min_mutation,
                         individual_selection=individual_selection,
                         gene_selection=gene_selection,
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
            genes_to_change += np.random.randint(self.min_mutation(), self.max_mutation() + 1,
                                                 size=genes_to_change.size)
        if individual.device == 'gpu':
            genes_to_change += cp.random.randint(self.min_mutation(), self.max_mutation() + 1,
                                                 size=genes_to_change.size)


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
            genes_to_change[:] = np.random.randint(2, size=len(genes_to_change))
        if individual.device == 'gpu':
            genes_to_change[:] = cp.random.randint(2, size=len(genes_to_change))
