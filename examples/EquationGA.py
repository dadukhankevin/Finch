"""
This is an example of how to find good *possible* values for all variables in a singular equation with many variables.
"""
from Finch.FinchGA import generic
from Finch.FinchGenetics import FitnessFunctions as ff
from Finch.FinchGA import Environments, GenePools
from Finch.FinchGA import Layers as l

desired = 33 #what we want our equation to equal
expression = """(x**x)+(y**y)+(z**z)""" #the equation with unkown variables
print("in")
eq = generic.Equation(["x", "y", "z"], expression, desired=desired) #Declare the equation in the equation class give it the variable names.

fitness = ff.EquationFitness(desired_result=desired, equation = eq) # Use the built in fitness function for an equation

gp = range(-100, 100)
gp = list(gp)
gp[gp.index(0)] = 1 # to prevent division by zero, this will be fixed soon
pool = GenePools.GenePool(gp, fitness.func, mx=1, mn=.01) # define the gene pool as well as the max and min gene weight. Supply the fitness function.
a = 0
def call():
    global a
    print(a)

    a+=1
env = Environments.SequentialEnvironment(layers=[ # Define the environment
    l.GenerateData(pool, population=30, array_length=3), # Generates data until len(data) == population
    l.Function(call),
    l.SortFitness(), #Sort individuals by fitness. Fitness is computed when individuals are changed or created.
    l.RecalculateDirectionalMutation(pool, fitness.func, amount=1, change_constant=1, every=5),
    l.MutateByDirectionalWeights(pool, fitness.func, amount=1),
    l.Function(call),
    l.KeepLength(0), #keeps a low population
])

#Run for 100 epochs or until the prediction is 99% true
env.compile(epochs=1,every=1, fitness=fitness.func, stop_threshold=.96) #prints every=20 epochs
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
