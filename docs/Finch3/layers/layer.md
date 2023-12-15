Module Finch3.layers.layer
==========================

Classes
-------

`Layer(individual_selection: Union[float, int, Callable, ForwardRef(None)] = None, gene_selection: [<class 'float'>, <class 'int'>, typing.Callable, None] = None, refit=True, fitness: int = 0, device: str = 'cpu')`
:   A Layer is a building block for an Environment and modifies or sorts the individuals in some way

:param execution_function: retrieved from a higher class
:param gene_selection: gene selection method: float, int, Callable
:param individual_selection: individual selection method: float, int, Callable
:param fitness: set to 0
:param device: 'cpu' or 'gpu'

### Ancestors (in MRO)

* Finch3.genetics.Individual

### Descendants

* Finch3.environments.Environment
* Finch3.layers.generic.CapPopulation
* Finch3.layers.generic.Controller
* Finch3.layers.generic.DuplicateSelection
* Finch3.layers.generic.Function
* Finch3.layers.generic.KillByFitnessPercentile
* Finch3.layers.generic.KillBySelection
* Finch3.layers.generic.Populate
* Finch3.layers.generic.RemoveAllButBest
* Finch3.layers.generic.RemoveDuplicatesFromTop
* Finch3.layers.generic.SortByFitness
* Finch3.layers.layer.Mutate
* Finch3.layers.layer.Parent

### Methods

`execute(self, individuals)`
:


`run(self, individuals: list, environment)`
:   :param individuals: The individuals in the environment
    :param environment: The environment containing the individuals
    :return: individuals

`set_environment(self, environment)`
:


`Mutate(gene_selection, individual_selection, device='cpu', refit=True)`
:   Base mutation layer, not to be used directly.

:param gene_selection: gene selection method
:param individual_selection: individual selection method
:param execution_function: execution_function (from higher layer)
:param device: 'cpu' or 'gpu'

### Ancestors (in MRO)

* Finch3.layers.layer.Layer
* Finch3.genetics.Individual

### Descendants

* Finch3.layers.mutation.BinaryMutate
* Finch3.layers.mutation.FloatMutateRange
* Finch3.layers.mutation.Mutate

### Methods

`execute(self, individuals)`
:   :param individuals: individuals to execute the mutation on
    :param environment: the environment our individuals live in (:

    :return: None

`mutate_one(self, individual, environment)`
:


`Parent(families, children=1, device='cpu', refit=True)`
:   Base parenting/crossover layer, not to be used directly.

:param individual_selection: individual selection method
:param execution_function:

:param device: 'cpu' or 'gpu'

### Ancestors (in MRO)

* Finch3.layers.layer.Layer
* Finch3.genetics.Individual

### Descendants

* Finch3.layers.parenting.ParentNPoint
* Finch3.layers.parenting.ParentSimple

### Methods

`execute(self, individuals)`
:


`parent(self, parent1, parent2, environment) ‑> list`
: