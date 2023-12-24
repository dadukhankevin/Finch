---
description: >-
  A simple example of how to use genetic algorithms to trick AI image
  recognition systems
---

# Trick Clip

port libraries

```python
from Finch.fitness.ml import image
from Finch.fitness.fitness_tools import MixFitness
from Finch.environments import Sequential
from Finch.genepools import BinaryPool
from Finch.layers import Mutate, ParentNPoint, Populate, CapPopulation, SortByFitness, BinaryMutate, BatchFitness, ParentSimple
from Finch.tools.individualselectors import RankBasedSelection
```

Now we need a way for our GenePool to know the amount of genes to give each individual

```python
size = (300, 300)
length = size[0] * size[1] * 3
shape = (size[1], size[0], 3)
print(length)
```

Now we need an ML model, we will attempt to trick Clip that our image of random noise contains a dog:

```python
batch_size = 10 # How many concurrent images to test on the GPU at a time
target = ["dog"]
other_labels = ["random noise", "white", "black"] # Our goal is for 'dog' to outscore any of these true labels
clip1 = image.ZeroShotImage(target_labels=target, other_labels=other_labels,
                            shape=shape, denormalize=True, batch_size=batch_size) # Default Clip model
```

Now we need a fitness function. Since we are running ML models, we probably want to use a GPU. To do this most efficiently, we need a fitness function that can calculate the fitness of many individuals at once.

```python
fit = clip1.batch_enhance
```

Now we can configure some helpful variables for our environment to use later:

```python
max_population = 100
start_pop = 2

parents = 20
factor = 10
parents = RankBasedSelection(factor=factor, amount_to_select=parents).select
children = 2
parent_points = 1

individual_selection = 50
gene_selection = 200

generations = 600
```

Next, we can create our gene pool and environment. Even though our Individuals will be on the CPU, the fitness functions will take place on the GPU.\


```python
pool = BinaryPool(length=length, device='cpu', default = 1)

environment = Sequential([
    Populate(pool, start_pop),
    BinaryMutate(individual_selection=individual_selection,
             gene_selection=gene_selection, refit=True),
    BatchFitness(batch_fitness_function=fit), # Calls the fitness function on every individual that has been modified
    SortByFitness(),
    ParentNPoint(families=parents, points=parent_points, children=children,
                 refit=True),
    BatchFitness(batch_fitness_function=fit),# Calls the fitness function on every individual that has been modified
    SortByFitness(),
    CapPopulation(max_population=max_population)
])
```

Next, we simply compile and evolve our environment:

```python
environment.compile(fitness_function='batch') # marks individuals for later batch_fitness
environment.evolve(generations=generations)
```

To view are best image, we simply call:

```python
clip1.show(environment.best_ever)
```

<figure><img src="../../.gitbook/assets/image (1).png" alt="caption"><figcaption><p>This image tricks clip into thinking it is a dog</p></figcaption></figure>

We can also plot our fitness over time, to see the progress our environment has made:\


```python
environment.plot()
```
