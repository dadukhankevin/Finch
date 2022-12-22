from Finch.FinchGenetics import *

desired = 10  # what we want our equation to equal
expression = """((x**y+z)*h)/j+t+l*k/x"""  # the equation with unknown variables

eq = Equation(["x", "y", "z", "h", "j", "t", "l", "k"], expression,
              desired=desired)  # Declare the equation in the equation class give it the variable names.
fitness = EquationFitness(desired_result=desired, equation=eq)  # Use the built-in fitness function for an equation

pool = FloatPool(1, 200, fitness.func, (len(eq.vars),), initialization="random")

env = Environment([
    Generate(pool, 100),
    Parents(pool, delay=1, gene_size=2, family_size=10, percent=.5, method="best", amount=10),
    FastMutateTop(pool),
    SortFitness(),
    KeepLength(90),
])

data, history = env.evolve(1000, verbose=100)
genes = data[-1].genes
print("Genes ", genes)
print("Result ", eq.evaluate(genes))
