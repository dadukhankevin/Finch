"""
Something here
"""
import numpy as np

cp = None
try:
    import cupy as cp
except ImportError:
    pass


class Individual:
    """
    Represents an individual in a population.

    This class can be used to model various types of individuals, such as images, books, or vectors.

    Attributes:
        genes (Any): The genetic information or characteristics of the individual.
        fitness (float): The default fitness value for this individual.
        parents (list): The parents of the individual
        children (list): Any children the parents have had.
    Example:
        >>> individual = Individual(genes=[0, 1, 1, 0], fitness=0.75)
    """

    def __init__(self, gene_pool, genes: np.array, fitness: float = 0, device="cpu"):
        self.genes = genes
        self.fitness = fitness
        self.device = device
        self.gene_pool = gene_pool
        self.check_fitness = False

        self.children = []
        self.parents = []
        self.age = 0

    def copy(self):
        """
        :return: A new copy of the individual
        """
        if self.device == "gpu":
            return Individual(genes=cp.copy(self.genes), fitness=self.fitness, device=self.device,
                              gene_pool=self.gene_pool)
        else:
            return Individual(genes=np.copy(self.genes), fitness=self.fitness, device=self.device,
                              gene_pool=self.gene_pool)

    def to_cpu(self):
        """
        self.genes will move to cpu and use numpy
        :return: self
        """
        if self.device == "gpu":
            self.genes = cp.asnumpy(self.genes)
            self.device = "cpu"
        return self

    def to_gpu(self):
        """
        self.genes will move to gpu and use cupy
        :return: self
        """
        if self.device == "cpu":
            self.genes = cp.asarray(self.genes)
            self.device = "gpu"
        return self
