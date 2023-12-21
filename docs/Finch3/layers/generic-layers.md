---
description: overview of generic Finch layers
---

# generic layers

### Layer:

Base class, not to be used independently. All other Layers inherit it.

**params/attributes:**

* `execution_function` retrieved from a higher class.
* `gene_selection` gene selection method: float (percent), int (amount), or callable.
* `individual_selection` individual selection method: float, int, Callable.
* `fitness` set to 0 (layers count as Individuals, this will matter only in later versions).
* `device` 'cpu' or 'gpu'.

**Methods:**

* `set_environment(environment: Environment)` Allows any Layer to access the Environment it is in. This is automatically completed in the host Environment.
* `run(self, individuals: list, environment)` This is what an Environment calls, which in turn calls `self.execution_function`. This allows individuals to be mutated, added, killed, or any number of other things.

### Populate:

`from Finch.layers.generic import Populate`

Populates your environment until it reaches a population.

**params**:

* `gene_pool: GenePool.` Whichever GenePool type you want to have to generate individuals
* `population: int.` The minimum population you would like your environment to maintain.

**Methods:**

* `execute(individuals: list[Individual]).` Populates the self.environment with new individuals.
