

layers.generic
============================

Classes
-------

``CapPopulation(max_population: int)``
:   A Layer is a building block for an Environment and modifies or sorts the individuals in some way:param execution_function: retrieved from a higher class:param gene_selection: gene selection method: float, int, Callable:param individual_selection: individual selection method: float, int, Callable:param fitness: set to 0:param device: 'cpu' or 'gpu'
### Ancestors (in MRO)
* Finch3.layers.layer.Layer* Finch3.genetics.Individual
### Methods
``execute(self, individuals)``:



``Controller(layer: Finch3.layers.layer.Layer, execute_every=1, repeat=1, delay=0, stop_at=inf)``
:   A Layer is a building block for an Environment and modifies or sorts the individuals in some way:param execution_function: retrieved from a higher class:param gene_selection: gene selection method: float, int, Callable:param individual_selection: individual selection method: float, int, Callable:param fitness: set to 0:param device: 'cpu' or 'gpu'
### Ancestors (in MRO)
* Finch3.layers.layer.Layer* Finch3.genetics.Individual
### Methods
``execute(self, individuals)``:



``DuplicateSelection(individual_selection)``
:   A Layer is a building block for an Environment and modifies or sorts the individuals in some way:param individual_selection:
### Ancestors (in MRO)
* Finch3.layers.layer.Layer* Finch3.genetics.Individual
### Methods
``execute(self, individuals)``:



``Function(function, individual_selection=None)``
:   A Layer is a building block for an Environment and modifies or sorts the individuals in some way:param execution_function: retrieved from a higher class:param gene_selection: gene selection method: float, int, Callable:param individual_selection: individual selection method: float, int, Callable:param fitness: set to 0:param device: 'cpu' or 'gpu'
### Ancestors (in MRO)
* Finch3.layers.layer.Layer* Finch3.genetics.Individual
### Methods
``execute(self, individuals)``:



``KillByFitnessPercentile(percentile: float)``
:   A Layer is a building block for an Environment and modifies or sorts the individuals in some way:param execution_function: retrieved from a higher class:param gene_selection: gene selection method: float, int, Callable:param individual_selection: individual selection method: float, int, Callable:param fitness: set to 0:param device: 'cpu' or 'gpu'
### Ancestors (in MRO)
* Finch3.layers.layer.Layer* Finch3.genetics.Individual
### Methods
``execute(self, individuals)``:



``KillBySelection(individual_selection)``
:   A Layer is a building block for an Environment and modifies or sorts the individuals in some way:param individual_selection: the method by witch to select the individuals for death
### Ancestors (in MRO)
* Finch3.layers.layer.Layer* Finch3.genetics.Individual
### Methods
``execute(self, individuals)``:



``Populate(gene_pool: Finch3.genepools.GenePool, population: int)``
:   A Layer is a building block for an Environment and modifies or sorts the individuals in some way:param gene_pool: any GenePool:param population: amount of individuals to ensure the population has
### Ancestors (in MRO)
* Finch3.layers.layer.Layer* Finch3.genetics.Individual
### Methods
``execute(self, individuals)``:



``RemoveAllButBest()``
:   A Layer is a building block for an Environment and modifies or sorts the individuals in some way:param execution_function: retrieved from a higher class:param gene_selection: gene selection method: float, int, Callable:param individual_selection: individual selection method: float, int, Callable:param fitness: set to 0:param device: 'cpu' or 'gpu'
### Ancestors (in MRO)
* Finch3.layers.layer.Layer* Finch3.genetics.Individual
### Methods
``execute(self, individuals)``:



``RemoveDuplicatesFromTop(top_n: int)``
:   A Layer is a building block for an Environment and modifies or sorts the individuals in some way:param execution_function: retrieved from a higher class:param gene_selection: gene selection method: float, int, Callable:param individual_selection: individual selection method: float, int, Callable:param fitness: set to 0:param device: 'cpu' or 'gpu'
### Ancestors (in MRO)
* Finch3.layers.layer.Layer* Finch3.genetics.Individual
### Methods
``execute(self, individuals)``:



``SortByFitness()``
:   A Layer is a building block for an Environment and modifies or sorts the individuals in some way:param execution_function: retrieved from a higher class:param gene_selection: gene selection method: float, int, Callable:param individual_selection: individual selection method: float, int, Callable:param fitness: set to 0:param device: 'cpu' or 'gpu'
### Ancestors (in MRO)
* Finch3.layers.layer.Layer* Finch3.genetics.Individual
### Methods
``execute(self, individuals)``:

