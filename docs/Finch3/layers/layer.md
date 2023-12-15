Module Finch.layers.layer
=========================

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

    * Finch.genetics.Individual

    ### Descendants

    * Finch.environments.Environment
    * Finch.layers.generic.CapPopulation
    * Finch.layers.generic.Controller
    * Finch.layers.generic.DuplicateSelection
    * Finch.layers.generic.Function
    * Finch.layers.generic.KillByFitnessPercentile
    * Finch.layers.generic.KillBySelection
    * Finch.layers.generic.Populate
    * Finch.layers.generic.RemoveAllButBest
    * Finch.layers.generic.RemoveDuplicatesFromTop
    * Finch.layers.generic.SortByFitness
    * Finch.layers.layer.Mutate
    * Finch.layers.layer.Parent

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

    * Finch.layers.layer.Layer
    * Finch.genetics.Individual

    ### Descendants

    * Finch.layers.mutation.BinaryMutate
    * Finch.layers.mutation.FloatMutateRange
    * Finch.layers.mutation.Mutate

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

    * Finch.layers.layer.Layer
    * Finch.genetics.Individual

    ### Descendants

    * Finch.layers.parenting.ParentNPoint
    * Finch.layers.parenting.ParentSimple

    ### Methods

    `execute(self, individuals)`
    :

    `parent(self, parent1, parent2, environment) ‑> list`
    :