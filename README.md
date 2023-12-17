## Finch3
### About
Finch3 is a genetic algorithm framework. It aims to be:
- understandable
- fast
- and capable

This is a very early version but here are some things you can expect:
- The ability to run genetic algorithms on a GPU
- Much better documentation than Finch2 (although this is a work in progress)
- Overall much faster, more consistent, and far less bugs.
### Docs
Here are the docs:

https://finch-1.gitbook.io/finch3/

Note: They are accurate but horribly formatted (will fix this soon)
### Colab Notebooks
1: [Simple Finch  Colab Demo](https://colab.research.google.com/drive/1QK7zYTBSkud4V6QQQRCi0ofkJ9bFp9B9?usp=sharing)
- Very simple genetic algorithm

2: [Trick Clip](https://colab.research.google.com/drive/1q_MDZAqofAbj-wkZHoWoWSP_6osymLGK?usp=sharing)
- A genetic algorithm built to create 'adversarial images' against the Clip image recognition model.

Many Google Colab notebooks will be added here soon!
### Installation 
run `git clone https://github.com/dadukhankevin/Finch`

### Example
This example demonstrates the usage of Finch3, a genetic programming library, to evolve individuals in a sequential environment.

1. Import Finch:
    ```python
    from Finch.environments import Sequential
    from Finch.genepools import FloatPool
    from Finch.layers import *
    ```

2. Define a fitness function `fit` that evaluates the performance of an individual. This is a very simple fitness function, and will essentially help us evolve a list floats to slowly become higher and higher numbers. The fitness function *could* also simply turn the floats into weights in a neural network and then evaluate their performance, this concept is called neuroevolution.
   ```python
   def fit(individual):
       return sum(individual.genes)  # You can modify the fitness function to make it interesting
   ```

3. Configure the genetic algorithm parameters in the Config section.
   ```python
   # Config section
   length = 100
   pool_minimum = 0
   pool_maximum = 10
   population_size = 100
   amount_to_mutate = 10
   gene_selection = 5
   parent_count = 20
   children_count = 2
   max_population = 99
   evolution_steps = 1000
   min_mutation = -2
   max_mutation = 2
   ```

4. Create a FloatPool to define the gene pool for individuals.
   ```python
   # Creating the FloatPool
   pool = FloatPool(length=length, minimum=pool_minimum, maximum=pool_maximum)
   ```

5. Create a Sequential environment with various layers representing different genetic operations.
   ```python
   # Creating the Sequential environment
   environment = Sequential(layers=[
       Populate(pool, population=population_size),
       FloatMutateRange(individual_selection=amount_to_mutate, gene_selection=gene_selection,
                        min_mutation=min_mutation, max_mutation=max_mutation, keep_within_genepool_bounds=True),
       ParentSimple(parent_count, children=children_count),
       SortByFitness(),
       CapPopulation(max_population=max_population),
   ])
   ```

6. Run the genetic algorithm.
   ```python
   if __name__ == "__main__":
       # Compiling the environment with the fitness function
       environment.compile(fitness_function=fit)
       environment.evolve(evolution_steps)

       # Printing the best individual
       print("Here is the best individual:\n", environment.best_ever.genes)

       # Plotting the environment
       environment.plot()
   ```
Another example can be found in the /examples directory!