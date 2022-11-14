# Finch
### A Keras style genetic algorithm framework.
#### Finch aims to be as simple and expandable to use as possible
#### What are genetic algorithms (GAs)?
GAs are a type of algorithm that simulate evolution in order to find answers to problems 
much faster than simply randomly guessing. They do not find "perfect" answers only "good" answers (at least usually). 
They do this by creating random "individuals". GAs then use a customizable fitness
function to determine how fit each individual is. After this mutation, parenting, and survival of the fittest do their magic to create
optimal solutions. GAs can do many of the same things that reinforcement learning algorithms can do but 
are more efficient because they entirely avoid the computationally expensive process of "backpropagation"
to learn more about this check out this excellent paper by OpenAI: https://openai.com/blog/evolution-strategies/
##  Disclaimer!
This framework is very new and many of the features are only "half baked". It is also very suseptable to change so use at your own risk.
## Installation 
```git clone https://github.com/dadukhankevin/Finch.git```

I hope to add it to PyPi (pip) soon.
## Colab notebooks (See these for example usage)
#### 1. [Sentiment based fitness function, for word generation GA:](https://colab.research.google.com/drive/1iknzsYyYYH66AOucfWznLlSTXFcDXP2P#scrollTo=LuYrxVC0N7kD)
#### 2. [Solve for all the missing variables, math based fitness function GA:](https://colab.research.google.com/drive/1MH5W08Jp4yUAv3Fx2qTO5Ds17XPfPFw4?usp=sharing)
#### 3. [Make the best backpack, value based fitness function GA:](https://colab.research.google.com/drive/1vpKZgWXK8fDN1xfm1x_cR8kJu1xbIiU1?usp=sharing)
#### 4. [Recreate an image (This one is so cool!)](https://colab.research.google.com/drive/1LCZSRed7n4ZMet1S6SsVTAvHXj3qeVKP?usp=sharing#scrollTo=lAuCsS-CBiDr)

## What makes Finch different than other libraries?
### Performance:
Finch does a lot of things to increase performance:
1. Finch only calculates fitness when individuals (or chromosomes) are generated, or changed.
2. Finch uses novel mutation functions that always improve the fitness of an individual see OverPowerdMutation().
This mutation function incentives mutating genes that have been modified the least amount of times and only keeps mutations
that benefit you. If you have a fitness function that requires more time, use FastMutate which creates a weight for each gene
and learns which ones generally improve the fitness.
3. Finch is highly customizable. Since it is modeled after Keras you can use the a Sequential environment to do any combinations and amounts of layers.
4. Each individual can be assigned its own fitness function.
5. You can take individuals from one environment and put them in another one.
6. You can change the fitness function when a past fitness function is maximized in its effect.
7. Built-in fitness functions
8. Built in environments (AutoGA)