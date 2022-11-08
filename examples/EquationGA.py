"""
This is an example of how to find good *possible* values for all variables in a singular equation with many variables.
"""
from Finch.FinchGA import generic
from Finch.FinchGA import FitnessFunctions as ff
from Finch.FinchGA import Environments, GenePools
from Finch.FinchGA import Layers as l
desired = 10
expression = """((x**y+z)*h)/j+t+l*k/x"""
eq = generic.Equation(["x", "y", "z", "h", "j","t","l","k"], expression, desired=desired)
fitness = ff.EquationFitness(desired_result=desired, equation = eq)
pool = GenePools.GenePool(list(range(1, 150)), fitness.func, mx=100, mn=0.001)

env = Environments.SequentialEnvironment(layers=[
    l.GenerateData(pool, population=30, array_length=8),
    l.SortFitness(),
    l.Mutate(pool, select_percent=50, likelihood=10),
    l.NarrowGRN(pool, delay=1, method="best", amount=1, reward=.2, penalty=.05, mn=.1, mx=200, every=1),
    l.UpdateWeights(pool),
    l.Parents(pool, gene_size=1, family_size=6, percent=100, every=4, method="best", amount=2), # parents the best ones
    l.KeepLength(100), #keeps a low population
])
env.compile(epochs=100,every=10, fitness=fitness.func, stop_threshold=.99)
hist, data = env.simulate_env()
info = env.display_history()
print("best percent: "+str(env.best))
print("best individual: "+str(env.best_ind.chromosome.get_raw()))
print(info)
env.plot()
