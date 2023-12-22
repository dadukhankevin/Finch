---
description: overview of generic Finch layers
---

# generic layers

### Layer:

Base class, not to be used independently. All other Layers inherit it.

**params/attributes: (some are optional)**

* `execution_function` retrieved from a higher class.
* `gene_selection` gene selection method: float (percent), int (amount), or callable.
* `individual_selection` individual selection method: float, int, Callable.
* `fitness` set to 0 (layers count as Individuals, this will matter only in later versions).
* `device` 'cpu' or 'gpu'.

**methods:**

* `set_environment(environment: Environment)` Allows any Layer to access the Environment it is in. This is automatically completed in the host Environment.
* `run(self, individuals: list, environment)` This is what an Environment calls, which in turn calls `self.execution_function`. This allows individuals to be mutated, added, killed, or any number of other things.

### Populate:

`from Finch.layers.generic import Populate`

Populates your environment until it reaches a population.

**params**:

* `gene_pool: GenePool.` Whichever GenePool type you want to have to generate individuals
* `population: int.` The minimum population you would like your environment to maintain.

**methods:**

* `execute(individuals: list[Individual]).` Populates the self.environment with new individuals.

### KillBySelection:

Give it a selection function, and it will terminate those individuals that are selected. Terminated individuals will have their genes erased however all other associated data will be transferred to `environment.dead_individuals`.

**params:**

* `individual_selection`  : A function with which to select individuals. (see more on the[selection-functions.md](../selection-functions.md "mention") page)

**methods:**

* `execute...` (identical to every other Layer) Kills selected individuals





### DuplicateSelection:

Give it a selection function, and it will duplicate those individuals that are selected.&#x20;

**params:**

* `individual_selection`  : A function with which to select individuals. (see more on the[selection-functions.md](../selection-functions.md "mention") page)

**Methods:**

* `execute...` (identical to every other Layer) Duplicate selected individuals



### SortByFitness:

This is one of the most important layers. It sorts the `environment.individuals` list so that individuals with higher fitness scores are close to index 0, and lower fitness scores -1. This allows other layers like `CapPopulation` to kill individuals lower on the ranking list, without checking fitness individually. **This layer is almost a requirement in every environment you create, and should generally be placed second to last.**&#x20;

**params:**

* None

**methods:**

* `execute...`&#x20;

### CapPopulation:

Keeps only the top `n` ranked individuals in the environment, killing the rest and placing them in `environment.dead_individuals`. **This layer (or another Kill layer) is almost a requirement in every environment you create, and should generally be placed last (after a fitness sort).**

**params:**

* `max_population: int`  The maximum number of individuals you specify in your environment, the rest will be killed. This may also be a function that returns an int.

**methods:**

* &#x20;`execute...`

### BatchFitness:

If you are using an ML model as a fitness function, you may want to run multiple fitness functions concurrently on the GPU. This can speed things up a lot!

**params:**

* `batch_fitness_function` The fitness function, which must take a list of individuals, rather than a singular individual, and pass them to the GPU however you specify.&#x20;

**methods:**

* `execute...` executes the batch\_fitness\_function and realigns the `fitness` score if each individual. It will only work on individuals that have `check_fitness=True`. The environment must also specify `fitness_function='batch'` &#x20;



### Function

Given a selection function, and a custom function, it will supply the custom function with the individuals selected. This is useful if you are building custom mutation functions or anything of that sort.

**params:**

* `function: callable`. The function that you define.
* `individual_selection: callable`. The selection function with which to select individuals from an environment.

**methods:**

* `execute...` Supplies your function with selected individuals



### Controller

Allows for fine-grained control over the execution of a specific layer. You can:

* repeat the layer x times
* delay the use of the layer until after x epochs
* stop the layer after x epochs
* or only execute the layer every x epochs

**params:**

* `layer: Layer` Place any other layer here.
* `execute_every: int` Executes the layer only every x times.
* `repeat: int` Repeat the layer x times per epoch.
* `delay: int` Delays the execution of this layer until after x epochs.
* `stop_at: int` Stops the layer from executing after x epochs.

**methods:**

* `execute...` Executes the `layer` that you specified in the manner defined above.

**example:**

```
layer = Controller(layer=DuplicateSelection(...), execute_every=10)
```

