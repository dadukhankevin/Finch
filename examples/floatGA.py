from Finch.FinchGA import Environments, EvolveRates, GenePools
from Finch.FinchGenetics import FitnessFunctions as ff
from Finch.FinchGA import Layers as l
from Finch.FinchGA import generic
desired = 500  # what we want our equation to equal
expression = """(x+y+z)/(x+.01)"""  # the equation with unkown variables
print("in")
rate = EvolveRates.Rate(5, .1, 1000)
eq = generic.Equation(["x", "y", "z"], expression,
                      desired=desired)  # Declare the equation in the equation class give it the variable names.

fitness = ff.EquationFitness(desired_result=desired, equation=eq)  # Use the built in fitness function for an equation

pool = GenePools.FloatPool(-.1, .1, fitfunc=fitness.func)  # define the gene pool as well as the max and min
# gene weight. Supply the fitness function.
env = Environments.SequentialEnvironment(layers=[  # Define the environment
    l.GenerateData(pool, population=50, array_length=3),  # Generates data until len(data) == population
    l.SortFitness(),  # Sort individuals by fitness. Fitness is computed when individuals are changed or created.
    l.OverPoweredMutation(pool=pool, iterations=1, index=-1, fitness_function=fitness, range_rate=rate.get, method="smart"), #Parenting is not needed in this case
    l.KeepLength(20),  # keeps a low population
])

# Run for 100 epochs or until the prediction is 99% true
env.compile(epochs=1000, every=10, fitness=fitness.func, stop_threshold=.99, callbacks=[rate.next])  # prints every=20 epochs
hist, data = env.simulate_env()