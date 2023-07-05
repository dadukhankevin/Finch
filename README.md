# Finch 2.0
![](resources/img.png)        

## What is Finch?
Finch is a genetic algorithm framework. 
Genetic algorithms are types of algorithms that mimic natural evolution to evolve 
solutions to problems. This is not to be confused with machine learning, they are not the same!
## Why Finch?
- Simple, easy to use
- All major functionalities are in seperate layers
- Easy to make custom layers
- GPU support can speed up GAs drastically
- Specify any crossover function, fitness function, or selection function
- Very modular
- Support for most types of genes:
  - floats
  - ints
  - strings
  - Keras weights * 
  - Pytorch weights *
  - Tensorflow weights *
  - Objects
## What is a genetic algorithm?
There are very few genetic algorithm frameworks out there. Finch aims to fill the void. 
Finch is simple, fast, and very customizable. It is modeled after Keras (the ML library) for its eas of use.
Before we get into examples, lets go over what a genetic algorithms consist of.
1. A fitness function
   - A fitness function tells the algorithm how fit an individual is, based on this the algorithm can remove the individual, or select if for further evolving.

2. Mutation
   - Mutation allows the individuals to change over time.

3. Crossover/parenting
    - The best individuals can pass their genes on to their children, ensuring beneficial mutations can survive and even creating better individuals through the mixing of genes.

These 3 concepts are universal to genetic algorithms, but in Finch we will introduce several more.
These include:
- Gene Pools
- Environments
- Mutation sharing (coming soon)

Now lets get started!
### Installation
```git clone https://github.com/dadukhankevin/Finch.git```

pip will be added later.
### Usage

```python

from Finch.environmental import layers, environments
from Finch.genetics import genepools
from Finch.functions import selection
```

Lets say we want to evolve a string to include only the letter "a". This is a pointless problem, but lets do it anyway!
We need a fitness function that returns a higher score the higher the amount of 'a's occor in text.

```python
def fit(individual):
    return "".join(individual).count("a") # individuals are lists
```

Next lets set up our environment!

```python
gene_pool = genepools.StringPool("qwertyuiopasdfghjklzxcvbnm", length=20, fitness_function=fit)

environment = environments.Sequential(layers=[
    layers.Populate(gene_pool=gene_pool, population=4),
    layers.MutateAmount(amount_individuals=4, amount_genes=3, gene_pool=gene_pool),
    layers.Parent(num_children=2, num_families=4),
    layers.SortByFitness(),
    layers.Kill(percent=.3),
])

environment.evolve(100)
```

It will output something like this after each generation:

```Generation 1/100. Max fitness: 0. Population: 0```

After 100 generations (less than a second):

```Generation 100/100. Max fitness: 10. Population: 16```

Now lets see the results:

```python
print(environment.individuals[0].genes)
```

Which prints a list of mostly the letter "a".
```
['o' 'a' 'p' 'v' 't' 'a' 'a' 'b' 'a' 'o' 'a' 'a' 'a' 'a' 'a' 'e' 'q' 'c'
 'a' 'a']
 ```

Congratulations! Your first genetic algorithm.

### Roadmap
- Set up a socket server (or something) to allow Finch to control elements of non-python environments like Unity or Unreal
- Better support of LLMs
- Mutation of Prompts (MoP) - specific to Finch
- Mutation Sharing (more on that later)
- Q# library (eventually) for quantum computing
- Better parallelization 
- Adversarial environments (soon)
- lists, tuples, and dictionaries as genes (possible if you make your own class) (soon)
- Consistent selection, crossover, and parenting functions (soon)
- AutoGA (similar to AutoML) (soonish)
- LLM pruning (long term)
- Better support for problems like traveling salesperson (soon) possible already
- Genetic Prompt Injection (just an idea 🤔)
- lots, lots more...
