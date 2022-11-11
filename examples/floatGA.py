"""
This is an example of how to find good *possible* values for all variables in a singular equation with many variables.
"""
from Finch.FinchGA import generic
from Finch.FinchGA import FitnessFunctions as ff
from Finch.FinchGA import Environments, GenePools
from Finch.FinchGA import Layers as l

desired = 33 #what we want our equation to equal
expression = """(x+y+z+o)*100""" #the equation with unkown variables
print("in")
eq = generic.Equation(["x", "y", "z", "o"], expression, desired=desired) #Declare the equation in the equation class give it the variable names.

fitness = ff.EquationFitness(desired_result=desired, equation = eq) # Use the built in fitness function for an equation

pool = GenePools.FloatPool(-100000, 100000, fitfunc=fitness.func) # define the gene pool as well as the max and min gene weight. Supply the fitness function.
env = Environments.SequentialEnvironment(layers=[ # Define the environment
    l.GenerateData(pool, population=50, array_length=4), # Generates data until len(data) == population
    l.SortFitness(), #Sort individuals by fitness. Fitness is computed when individuals are changed or created.
    l.FastMutateTop(pool, amount=10, every=1, fitness_mix_factor=1, individual_mutation_amount=5),
    l.Parents(pool, gene_size=1, family_size=2, percent=100, every=4, method="best", amount=4), # parents the best ones.
    l.KeepLength(50), #keeps a low population
])

#Run for 100 epochs or until the prediction is 99% true
env.compile(epochs=5000,every=100, fitness=fitness.func, stop_threshold=.9999) #prints every=20 epochs
hist, data = env.simulate_env()
info = env.display_history()
print("best percent: "+str(env.best)) # this is still slightly buggy
print("best individual: "+str(env.best_ind.genes))
print(info)
#This should be fairly close to 10
a=eq.evaluate(env.best_ind.genes)
print(a)
print(pool.weights)
#graph the fitness over time (fitness * epochs)
env.history = [max(0, i) for i in env.history] # Keeps only the valid percents
env.plot()
