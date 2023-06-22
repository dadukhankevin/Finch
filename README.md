# Finch 2.0
![](img.png)        

## What is Finch?
Finch is a genetic algorithm framework. 
Genetic algorithms are types of algorithms that mimic natural evolution to evolve 
solutions to problems. This is not to be confused with machine learning, they are not the same!

## Why Finch?
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

TODO: Finish the rest of the readme (:
