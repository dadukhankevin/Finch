---
description: A simple example of an ArrayPool.
---

# Arrays

In this example, we will evolve a set of individuals. Each individual will contain a long list of characters. Our goal is to simply evolve our individuals such that the individual contains many occurrences of the letter 'a'

### Import Finch:

```python
from Finch.environments import Sequential
from Finch.genepools import ArrayPool
from Finch.layers import *
import numpy as np
```

### Define a Fitness Function:

This function determines how "fit" any given individual is. The higher the count of the letter 'a' that appears, the more fit it its, and the more likely to pass on its genes and live.

```python
def fit(individual):
    return "".join(individual.genes).count('a')
```

### Configure helpful variables.

<pre class="language-python"><code class="lang-python"># Information for our GenePool
length = 100 # each individual.genes array will have 100 items
pool_minimum = 0
pool_maximum = 10
<strong># environment variables
</strong>population_size = 100
evolution_steps = 500
max_population = 99
# parenting/crossover
parent_count = 20
children_count = 2
# mutation
amount_to_mutate = 10
gene_selection = 5
</code></pre>

### Create a ArrayPool:

Here, our `GenePool` is a `ArrayPool`, which simply means it is meant to generate individuals containing any elements from a source array.

* genes = the possible genes to choose from when it creates individuals
* length = amount of genes to give each individual

```python
genes = list("qwertyuiopasdfghjklzxcvbnm") # this could be a list of anything
pool = ArrayPool(genes=genes, length=length)
```

### Create an Environment:

Since we are creating a pretty simple genetic algorithm, the only `Environment` we need to use is the simple `Sequential` environment. This allows us to put the basic elements of our genetic algorithm together using layers:

* `Populate` Generates individuals for the environment until the `population_size` is met.
* `Mutate`Mutates several individuals, changing some of their genes by replacing them with new ones.
* `SortByFitness` Sorts our individuals.
* `CapPopulation` Using the sorting in the previous layer, removes the worst individuals and keeps the best.

```python
environment = Sequential(layers=[
    Populate(pool, population=population_size),
    Mutate(individual_selection=amount_to_mutate, gene_selection=gene_selection),
    ParentSimple(parent_count, children=children_count),
    SortByFitness(),
    CapPopulation(max_population=max_population),
])
```



### Evolve our Individuals:

Now we can compile and run our environment!

```python
environment.compile(fitness_function=fit)
environment.evolve(evolution_steps)
print("Here is the best individual:\n", environment.best_ever.genes)
```

This should contain many 'a's.&#x20;

Congrats! now you can change the fitness function to be anything!

### View Environment History:

Lastly, let's plot the history of the fitness of our best individuals from each generation:

`environment.plot()`
