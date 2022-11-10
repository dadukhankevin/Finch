"""
This is an example of how to find good *possible* values for all variables in a singular equation with many variables.
"""
from Finch.FinchGA import generic
from Finch.FinchGA import FitnessFunctions as ff
from Finch.FinchGA import Environments, GenePools
from Finch.FinchGA import Layers as l

desired = 10 #what we want our equation to equal
expression = """((x**y+z)*h)/j+t+l*k/x""" #the equation with unkown variables

eq = generic.Equation(["x", "y", "z", "h", "j","t","l","k"], expression, desired=desired) #Declare the equation in the equation class give it the variable names.
fitness = ff.EquationFitness(desired_result=desired, equation = eq) # Use the built in fitness function for an equation

gp = range(-200, 200)
gp = list(gp)
gp[gp.index(0)] = 1 # to prevent division by zero, this will be fixed soon
pool = GenePools.GenePool(gp, fitness.func, mx=100, mn=0.001) # define the gene pool as well as the max and min gene weight. Supply the fitness function.

env = Environments.SequentialEnvironment(layers=[ # Define the environment
    l.GenerateData(pool, population=30, array_length=8), # Generates data until len(data) == population
    l.SortFitness(), #Sort individuals by fitness. Fitness is computed when individuals are changed or created.
    l.Mutate(pool, select_percent=.5, likelihood=10), #Mutates 10% of 50% of the individuals
    l.NarrowGRN(pool, delay=1, method="best", amount=1, reward=.3, penalty=.05, mn=.1, mx=200, every=1), # Add weight to our favorite genes
    l.UpdateWeights(pool),
    l.Parents(pool, gene_size=1, family_size=2, percent=100, every=4, method="best", amount=4), # parents the best ones.
    l.KeepLength(10), #keeps a low population
])

#Run for 100 epochs or until the prediction is 99% true
env.compile(epochs=10,every=1, fitness=fitness.func, stop_threshold=.999) #prints every=20 epochs
hist, data = env.simulate_env()
info = env.display_history()
print("best percent: "+str(env.best)) # this is still slightly buggy
print("best individual: "+str(env.best_ind.genes))
print(info)
#This should be fairly close to 10
a=eq.evaluate(env.best_ind.genes)
print(a)
#graph the fitness over time (fitness * epochs)
env.history = [max(0, i) for i in env.history] # Keeps only the valid percents
env.plot()