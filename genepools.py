"""
GenePools

This script defines various gene pool classes for the creation of individuals with different types of genes.
Each gene pool class inherits from the base class GenePool and implements methods for generating individuals
with specific gene types.

Classes:
- GenePool: Base class for gene pools. (not usable)
- FloatPool: Gene pool for individuals containing floating-point genes.
- IntPool: Gene pool for individuals containing integer genes.
- BinaryPool: Gene pool for individuals containing binary genes (0 or 1).
- ArrayPool: Gene pool for individuals by picking items from an array.

Author: Daniel Losey
Year: 2023
"""
from Finch import genetics
import numpy as np

cp = None
try:
    import cupy as cp
except ImportError:
    pass


class GenePool:
    def __init__(self, length: int, device='cpu'):
        """
        Base GenePool class
        :param length: Amount of genes to place in an individual
        :param device: 'cpu' or 'gpu'
        """
        self.device = device
        self.length = length
        assert self.device in ('cpu', 'gpu'), f"Invalid device: {self.device}. Must be either 'cpu' or 'gpu'."

    def generate_individual(self):
        """
        Generates a new individual.
        :return: genetics.Individual with genes of type int
        """
        genes = self.generate_genes(self.length)
        return genetics.Individual(genes=genes, fitness=0, device=self.device, gene_pool=self)

    def generate_genes(self, amount: int):
        pass


class FloatPool(GenePool):
    def __init__(self, length: int, minimum: float = 0, maximum: float = 1, default: float = None, device="cpu"):
        """
        A GenePool meant specifically for the creation of individuals containing floats.
        :param length: Amount of genes in each individual
        :param minimum: Minimum value for a gene
        :param maximum: Maximum value for a gene
        :param default: If set will initialize all genes to the same value
        :param device: Where genes in Individuals should be kept 'gpu' or 'cpu'
        """
        super().__init__(length=length, device=device)
        self.minimum = minimum
        self.maximum = maximum
        self.default = default

    def generate_genes(self, amount: int):
        """
        :param amount: Amount of genes to generate
        :return: numpy or cupy array containing genes
        """
        if self.device == "cpu":  # generate genes using numpy
            if self.default is not None:
                genes = np.full(amount, self.default)
            else:
                genes = np.random.uniform(self.minimum, self.maximum, size=amount)

        elif self.device == "gpu":  # generate genes using cupy
            if self.default is not None:
                genes = cp.full(amount, self.default)
            else:
                genes = cp.random.uniform(self.minimum, self.maximum, size=amount)

        return genes


class IntPool(GenePool):
    def __init__(self, length: int, minimum: int = 0, maximum: int = 1, default: int = None, device="cpu"):
        """
        A GenePool meant specifically for the creation of individuals containing floats.
        :param length: Amount of genes in each individual
        :param minimum: Minimum value for a gene
        :param maximum: Maximum value for a gene
        :param default: If set will initialize all genes to the same value
        :param device: Where genes in Individuals should be kept 'gpu' or 'cpu'
        """
        super().__init__(length=length, device=device)
        self.minimum = minimum
        self.maximum = maximum
        self.default = default

    def generate_genes(self, amount: int):
        """
        :param amount: Amount of genes to generate
        :return: numpy or cupy array containing genes
        """
        if self.device == "cpu":  # generate genes using numpy
            if self.default is not None:
                genes = np.full(amount, self.default)
            else:
                genes = np.random.randint(self.minimum, self.maximum + 1, size=amount)

        elif self.device == "gpu":  # generate genes using cupy
            if self.default is not None:
                genes = cp.full(amount, self.default)
            else:
                genes = cp.random.randint(self.minimum, self.maximum + 1, size=amount)
        return genes


class BinaryPool(GenePool):
    def __init__(self, length: int, default: int = None, device="cpu"):
        """
        A GenePool meant specifically for the creation of individuals containing binary genes (0 or 1).
        :param length: Amount of genes in each individual
        :param default: If set will initialize all genes to the same value
        :param device: Where genes in Individuals should be kept 'gpu' or 'cpu'
        """
        super().__init__(length=length, device=device)
        self.default = default

    def generate_genes(self, amount: int):
        """
        :param amount: Amount of genes to generate
        :return: numpy or cupy array containing binary genes (0 or 1)
        """
        if self.device == "cpu":  # generate genes using numpy
            if self.default is not None:
                genes = np.full(amount, self.default, dtype=int)
            else:
                genes = np.random.randint(2, size=amount, dtype=int)

        elif self.device == "gpu":  # generate genes using cupy
            if self.default is not None:
                genes = cp.full(amount, self.default, dtype=int)
            else:
                genes = cp.random.randint(2, size=amount, dtype=int)

        return genes


class ArrayPool(GenePool):
    def __init__(self, gene_array: np.ndarray, length: int, device="cpu", unique=False):
        """
        A GenePool meant for the creation of individuals by picking items from an array.
        :param gene_array: The array from which genes will be picked.
        :param length: Amount of genes in each individual
        :param device: Where genes in Individuals should be kept 'gpu' or 'cpu'
        :param unique: Whether the generated genes should be unique
        """
        super().__init__(length=length, device=device)
        self.gene_array = gene_array
        self.unique = unique

    def generate_genes(self, amount: int):
        """
        :param amount: Amount of genes to generate
        :return: numpy or cupy array containing genes picked from the array
        """
        if self.device == "cpu":
            if self.unique:
                genes = np.random.choice(self.gene_array, size=amount, replace=False)
            else:
                genes = np.random.choice(self.gene_array, size=amount)

        elif self.device == "gpu":
            if self.unique:
                genes = cp.random.choice(self.gene_array, size=amount, replace=False)
            else:
                genes = cp.random.choice(self.gene_array, size=amount)
        return genes
