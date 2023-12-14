import numpy as np
cp = None
try:
    import cupy as cp
except ImportError:
    pass


class Individual:
    def __init__(self, gene_pool, genes: np.array, fitness: float = 0, device="cpu",):
        """
        Defines an individual in a population. This can be anything from an image to a book to a vector.
        :param genes: individual genes
        :param fitness: the default fitness for this individual
        """
        self.genes = genes
        self.fitness = fitness
        self.device = device
        self.gene_pool = gene_pool

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
