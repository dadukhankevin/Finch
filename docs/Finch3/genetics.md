Module Finch3.genetics
======================

Classes
-------

`Individual(gene_pool, genes: <built-in function array>, fitness: float = 0, device='cpu')`
:   Defines an individual in a population. This can be anything from an image to a book to a vector.
    :param genes: individual genes
    :param fitness: the default fitness for this individual

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