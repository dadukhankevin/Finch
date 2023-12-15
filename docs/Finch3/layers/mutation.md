

layers.mutation
=============================

Classes
-------

``BinaryMutate(individual_selection: Union[float, int, Callable], gene_selection: Union[float, int, Callable], refit=True)``
:   A binary-based layer that mutates genes by changing their values to 0 or 1.:param gene_selection: gene selection method:param individual_selection: individual selection method:param execution_function: execution_function (from higher layer):param device: 'cpu' or 'gpu'
### Ancestors (in MRO)
* Finch3.layers.layer.Mutate* Finch3.layers.layer.Layer* Finch3.genetics.Individual
### Methods
``mutate_one(self, individual, environment)``:

   Mutates a single individual    :param environment: the environment the individuals live in    :param individual: the individual to mutate    :return: None

``FloatMutateRange(individual_selection: Union[float, int, Callable], gene_selection: Union[float, int, Callable], max_mutation: Union[float, int] = 1.0, min_mutation: Union[float, int] = -1.0, refit=True, keep_within_genepool_bounds=False)``
:   A float based layer that mutates genes by changing their values rather than replacing them.This can allow for more fine-grained mutations.:param max_mutation: maximum mutation:param min_mutation: minimum mutation:param individual_selection: individual selection method:param gene_selection: gene selection method:param keep_within_genepool_bounds: keeps the values of the floats within the maxand min set in their gene pools
### Ancestors (in MRO)
* Finch3.layers.layer.Mutate* Finch3.layers.layer.Layer* Finch3.genetics.Individual
### Descendants
* Finch3.layers.mutation.IntMutateRange
### Methods
``mutate_one(self, individual, environment)``:

   Mutates a single individual    :param environment: the environment the individuals live in    :param individual: the individual to mutate    :return: None

``IntMutateRange(gene_selection: Union[float, int, Callable], individual_selection: Union[float, int, Callable], max_mutation: Union[int, Callable] = 1, min_mutation: Union[int, Callable] = -1, refit=True)``
:   An integer-based layer that mutates genes by changing their values rather than replacing them.This can allow for more fine-grained mutations.:param max_mutation: maximum mutation:param min_mutation: minimum mutation:param individual_selection: individual selection method:param gene_selection: gene selection method
### Ancestors (in MRO)
* Finch3.layers.mutation.FloatMutateRange* Finch3.layers.layer.Mutate* Finch3.layers.layer.Layer* Finch3.genetics.Individual

``Mutate(individual_selection: Union[float, int, Callable], gene_selection: Union[float, int, Callable], refit=True)``
:   Base mutation layer, not to be used directly.:param gene_selection: gene selection method:param individual_selection: individual selection method:param execution_function: execution_function (from higher layer):param device: 'cpu' or 'gpu'
### Ancestors (in MRO)
* Finch3.layers.layer.Mutate* Finch3.layers.layer.Layer* Finch3.genetics.Individual
### Methods
``mutate_one(self, individual, environment)``:

   Mutates a single individual    :param environment: the environment the individuals live in    :param individual: the individual to mutate    :return: None