# Finch
A Keras style GA genetic algorithm library
Example:
import modules
```
from FinchGA.finchstring import *
```
Genetic algorithms simulate evolution in order to find solutions to complex problems.
They work by generating a random "generation" and then testing how "fit" each individual in the population is.
To do this, we need to create a fitness function which will take an individual from the population and determine how fit it is.
```
max = 100
def fit(individual):  # A basic function that insensitive generations with the most occurrences of the letter a.
    points = individual.count("a")
    if max:
        points = (points/max)*100
    return points, individual
```
Now we need to declare an "Environment" think of this as similar to a keras.Sequential. 
```
environment = Environment()  # Similar to keras.Sequential
```
This allows us to use enviornment.Add() to add diffrent attributes to our environment.
```
environment.add(
    Kill(.2)  # Kills the bottom 90% of the population
)
```
This function, will kill 20% of our population in each epoch (repitition or loop).

We still don't have any actuall data so we can add a generation using the String() class. 
With these parameters it will generate an array of size 100. Within each element will be a random string of letters from the "letters" variable. This can be any string. We will also supply it with our fittness function.
```
environment.add(
    String(100, 100, fit, letters="qwertyuiopasdfghjklzxcvbnm")
)
```
What is evolution without mutation?
Not that great.
Here we can mutate the generation above. There are various diffrent ways to mutate data but this but here is the simple version.
```
environment.add(
    StringMutate(percentage=99, small_percent=1, letters="qwertyuiopasdfghjklzxcvbnm")  #mutates 99 percent of the generation and within those it mutates it changes 1% of the letters.
)
```
Now lets take the top 2 individuals in the population and create 5 "children" variations of them. 
```
environment.add(
    Parent(5, gene_size=1) #Later updates will have more robust ways of parenting more thna just two individuals
)
```
Lets see how this goes as we simulate evolution on 1,000 generations. 
```
data, history = environment.compile(1000, func=fit, verbose=True)
plt.plot(history)
plt.show()
```
Here is our best individual:
(70.0, 'zxyvzayoaabaaaaatajavawavaihaaaaaaatrxaaabaaawraaaaaaaaaajaaaaaaafazaaaaawaaaaavavalaaaaasaaaaaaauaa')
It was 70% the letter "a". There are lots of ways to improve this. 
Here is the plot of how well our population became more fit over time:
[ag](https://user-images.githubusercontent.com/61605641/198392970-3f6be40f-5bc7-4e05-a598-84b018859c5c.png)
And for those of you who want to copy the whole thing:
```
from FinchGA.finchstring import *

max = 100
def fit(generation):  # A basic function that insensitive generations with the most occurrences of the letter a.
    points = generation.count("a")
    if max:
        points = (points/max)*100
    return points, generation


environment = Environment()  # Similar to keras.Sequential
environment.add(
    Kill(.2)  # Kills the bottom 90% of the population
)
environment.add(
    String(100, 100, fit, letters="qwertyuiopasdfghjklzxcvbnm")  # Creates a new generation of
)
environment.add(
    StringMutate(percentage=99, small_percent=1, letters="qwertyuiopasdfghjklzxcvbnm")  #mutates 99 percent of the generation and within those it mutates it changes 1% of the letters.
)

environment.add(
    Parent(5, gene_size=1)
)
data, history = environment.compile(1000, func=fit, verbose=True)
plt.plot(history)
plt.show()
```
