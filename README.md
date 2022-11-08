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
### Disclaimer!
This framework is very new and many of the features are only "half baked". It is also very suseptable to change so use at your own risk.
## Installation 
```git clone https://github.com/dadukhankevin/Finch.git```

I hope to add it to PyPi (pip) soon.
# Usage:
Wow, lets move on to the interesting stuff already!
### The "backpack" problem
The backpack problem is sort the "hello world" of genetic algorithms.
The problem is as such:

1. You have to go on a trip
2. You can't carry more than, say, 15 pounds in your backpack
3. find the best possible things to pack given this constraint.

To solve this we must:
1. Create a list of items where each element looks like this: [item name, weight, importance]
2. Create a fitness function that returns the added up "importance" of each item in your backpack or, if the backpack weighs too much, returns 0.
3. Create many backpacks (individuals)
4. Determine the most "fit"
5. Parent the best ones together.
6. Mutate some of them
7. Determine fitness
8. Repeat x amount of times, also known as epochs.
9. Print the best result

Voilà! This is intimidating but once you get this process down you'll be able to solve more interesting problems like find the best 100 variables to make 1 equation equal 7 (and other interesting things).

### The code (finally!)
First lets import the relevant modules
```
A Keras style GA genetic algorithm library
from Finch.FinchGA.GenePools import GenePool
from Finch.FinchGA import Layers
import matplotlib.pyplot as plt
from Finch.FinchGA.Environments import *
import numpy as np
```
Layers make up "environments" that help evolve your individuals. GenePools help you define what items you want in your backpack and increase the odds of better genes being generated as the environment is simulated.


#### The fitness function
This function determines the fitness of an individual, in our case, a backpack.
```

def fit(backpack):
    weight = 0 
    value = 0
    for item in backpack: 
        weight += float(item[2]) # Adds to the weight of the backpack
        value += float(item[1]) # Adds to the overall value of the backpack
    if weight > 15: # If weight is above 15 fitness is 0
        return 0
    else:
        return value # otherwise return the sum of the importance of each item in teh backpack
    
```
That wasn't so bad! Now lets define the items that _can_ be in our backpack.
```
# In the format [name, weight, value] all of these have little bearing on reality.
backpack = np.array(
    [["apple", 1, .1], ["phone", 8, 6], ["lighter", 1, .1], ["Book", .1, 2], ["compass", 2, .4], ["flashlight", 1, 6],
     ["water", 5, 9], ["passport", 7, .5]])
```
Now we need to put all these items into a GenePool class:
```
pool = GenePool(backpack, fit, replacement=False)  # TO avoid duplicates "replacement" must be false
```
The gene pool helps produce better genes as the environment learns what genes perform best in individuals. 
Now we can create our environment. Tinker around with it to see how each thing effects the performance.
```
env = SequentialEnvironment(layers=[
    Layers.GenerateData(pool, population=10, array_length=4, delay=0), # Generates data
    Layers.SortFitness(), # Sorts individuals by fitness
    Layers.NarrowGRN(pool, delay=1, method="best", amount=1, reward=.05, penalty=.05, mn=.1, mx=100, every=1), # Calculates new weights
    Layers.UpdateWeights(pool, every=1, end=200), # Updates likelihood of specific 
    Layers.Parents(pool, gene_size=1, family_size=1, delay=0, every=4, method="random"), #Parents random individuals together
    Layers.Mutate(pool, delay=0, select_percent=100, likelihood=40), #mutates 40% ish of all of the individuals 
    Layers.SortFitness(), # Sorts individuals by fitness
    Layers.KeepLength(100), # Keeps the population under 100

])
```
As you may be able to tell, this looks a lot like the AI library Keras's Sequential class.

now lets run our environment.
```
env.compile(epochs=40, fitness=fit, every=1, stop_threshold=18) #stop when value > 18
_, hist = env.simulate_env()
print(pool.weights) # relative weights of each gene
plt.plot(hist)
plt.show() # Graph our progress
```

If you want to help out just submit a PR! It would be much appreciated.

# Roadmap
#### 1. Game engine integration with Unity!
#### 2. Create example Google Colab notebooks.
#### 3. Allow for more types of evolutionary algorithms.
#### 4. Predefined fitness functions.
#### 5. AutoGA principles inspired by AutoML where there are many predefined environments.
#### 6. Potential integration with OpenAI Gym.
#### 7. Continue to work on scalability.