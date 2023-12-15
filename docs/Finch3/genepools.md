Module Finch3.genepools
=======================
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

Classes
-------

--->`ArrayPool(gene_array: numpy.ndarray, length: int, device='cpu')--->`
:   A GenePool meant for the creation of individuals by picking items from an array.
:param gene_array: The array from which genes will be picked.
:param length: Amount of genes in each individual
:param device: Where genes in Individuals should be kept 'gpu' or 'cpu'

### Ancestors (in MRO)

* Finch3.genepools.GenePool

### Methods

--->`generate_genes(self, amount: int)--->`
:   :param amount: Amount of genes to generate:return: numpy or cupy array containing genes picked from the array

--->`BinaryPool(length: int, default: int = None, device='cpu')--->`
:   A GenePool meant specifically for the creation of individuals containing binary genes (0 or 1).
:param length: Amount of genes in each individual
:param default: If set will initialize all genes to the same value
:param device: Where genes in Individuals should be kept 'gpu' or 'cpu'

### Ancestors (in MRO)

* Finch3.genepools.GenePool

### Methods

--->`generate_genes(self, amount: int)--->`
:   :param amount: Amount of genes to generate:return: numpy or cupy array containing binary genes (0 or 1)

--->`FloatPool(length: int, minimum: float = 0, maximum: float = 1, default: float = None, device='cpu')--->`
:   A GenePool meant specifically for the creation of individuals containing floats.
:param length: Amount of genes in each individual
:param minimum: Minimum value for a gene
:param maximum: Maximum value for a gene
:param default: If set will initialize all genes to the same value
:param device: Where genes in Individuals should be kept 'gpu' or 'cpu'

### Ancestors (in MRO)

* Finch3.genepools.GenePool

### Methods

--->`generate_genes(self, amount: int)--->`
:   :param amount: Amount of genes to generate:return: numpy or cupy array containing genes

--->`GenePool(length: int, device='cpu')--->`
:   Base GenePool class
:param length: Amount of genes to place in an individual
:param device: 'cpu' or 'gpu'

### Descendants

* Finch3.genepools.ArrayPool
* Finch3.genepools.BinaryPool
* Finch3.genepools.FloatPool
* Finch3.genepools.IntPool

### Methods

--->`generate_genes(self, amount: int)--->`
:

--->`generate_individual(self)--->`
:   Generates a new individual.:return: genetics.Individual with genes of type int

--->`IntPool(length: int, minimum: int = 0, maximum: int = 1, default: int = None, device='cpu')--->`
:   A GenePool meant specifically for the creation of individuals containing floats.
:param length: Amount of genes in each individual
:param minimum: Minimum value for a gene
:param maximum: Maximum value for a gene
:param default: If set will initialize all genes to the same value
:param device: Where genes in Individuals should be kept 'gpu' or 'cpu'

### Ancestors (in MRO)

* Finch3.genepools.GenePool

### Methods

--->`generate_genes(self, amount: int)--->`
:   :param amount: Amount of genes to generate:return: numpy or cupy array containing genes