from Finch.FinchGenetics import *

desired = 10  # what we want our equation to equal
expression = """((x**y+z)*h)/j+t+l*k/x"""  # the equation with unknown variables

eq = Equation(["x", "y", "z", "h", "j", "t", "l", "k"], expression,
              desired=desired)  # Declare the equation in the equation class give it the variable names.
fitness = EquationFitness(desired_result=desired, equation=eq)  # Use the built-in fitness function for an equation

pool = FloatPool(3, 200, fitness.func, (len(eq.vars),), initialization="midpoint")

env = Environment([
    Generate(pool, 6),
    Parents(pool, delay=1, gene_size=1, family_size=2, percent=.5, method="best", amount=10, every=100),
    #FastMutateTop(pool),
    OPMutation(pool, fitness, iterations=300),
    SortFitness(),
    KeepLength(5),
])

data, history = env.evolve(1500, verbose=100)
genes = data[-1].genes
print("Genes ", genes)
print("Result ", eq.evaluate(genes))
