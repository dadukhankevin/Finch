---
description: overview of generic Finch layers
---

# generic layers

### Populate:

`from Finch.layers.generic import Populate`

Populates your environment until it reaches a population:

* `gene_pool: GenePool.` Whichever GenePool type you want to have to generate individuals
* `population: int.` The minimum population you would like your environment to maintain.
