# Finch
A Keras style GA genetic algorithm library
Example:
import modules
```
from FinchGA.finchstring import *
```
Genetic algorithms simulate evolution in order to find solutions to complex problems.
They work by 
```
max = 100
def fit(generation):  # A basic function that insensitive generations with the most occurrences of the letter a.
    points = generation.count("a")
    if max:
        points = (points/max)*100
    return points, generation
```
