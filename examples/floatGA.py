"""
This is an example of how to find good *possible* values for all variables in a singular equation with many variables.
"""
from Finch.FinchGA import generic, GenePools
from Finch.FinchGA import FitnessFunctions as ff
from Finch.FinchGA import Environments, EvolveRates
from Finch.FinchGA import Layers as l

desired = 0 # what we want our equation to equal
expression = """-(x**x)+3"""  # the equation with unkown variables
print("in")
rate = EvolveRates.Rate(5, .1, 1000)
eq = generic.Equation(["x"], expression,
                      desired=desired)  # Declare the equation in the equation class give it the variable names.

fitness = ff.EquationFitness(desired_result=desired, equation=eq)  # Use the built in fitness function for an equation

pool = GenePools.FloatPool(-2000, 2000, fitfunc=fitness.func)  # define the gene pool as well as the max and min
# gene weight. Supply the fitness function.
env = Environments.SequentialEnvironment(layers=[  # Define the environment
    l.GenerateData(pool, population=5, array_length=1),  # Generates data until len(data) == population
    l.SortFitness(),  # Sort individuals by fitness. Fitness is computed when individuals are changed or created.
    l.OverPoweredMutation(pool=pool, iterations=1, index=-1, fitness_function=fitness, range_rate=rate.get, method="smart"), #Parenting is not needed in this case
    l.KeepLength(20),  # keeps a low population
])

input("stop")
# Run for 100 epochs or until the prediction is 99% true
env.compile(epochs=1000, every=10, fitness=fitness.func, stop_threshold=.99, callbacks=[rate.next])  # prints every=20 epochs
hist, data = env.simulate_env()
info = env.display_history()
print("best percent: " + str(env.best))  # this is still slightly buggy
print("best individual: " + str(env.best_ind.genes))
print(info)
# This should be fairly close to 10
a = eq.evaluate(env.best_ind.genes)
print(a)
print(pool.weights)
# graph the fitness over time (fitness * epochs)
env.history = [max(0, i) for i in env.history]  # Keeps only the valid percents
env.plot()
