"""
This just shows how to use the Equation class
To see an example of how to incorporate this into a GA seep EquationGA.py in the examples directory
"""
from Finch.FinchGenetics import Equation

equation = """x**y+z"""
eq = Equation(["x", "y", "z"], equation, desired=10) #desired will only be used in a fitness function, don't worry about it for now
result = eq.evaluate([1, 2, 3])
print(result)

# returns 4
