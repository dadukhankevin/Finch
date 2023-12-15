Module Finch3.genetics
======================
Something here

Classes
-------

`Individual(gene_pool, genes: <built-in function array>, fitness: float = 0, device='cpu')`:


Represents an individual in a population.

This class can be used to model various types of individuals, such as images, books, or vectors.

Attributes:

    genes (Any): The genetic information or characteristics of the individual.
    fitness (float): The default fitness value for this individual.
Example:

    >>> individual = Individual(genes=[0, 1, 1, 0], fitness=0.75)

### Descendants

* Finch3.layers.layer.Layer

### Methods

`copy(self)`
:   :return: A new copy of the individual

`to_cpu(self)`
:   self.genes will move to cpu and use numpy
    :return: self

`to_gpu(self)`
:   self.genes will move to gpu and use cupy
    :return: self