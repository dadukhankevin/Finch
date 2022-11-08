"""
This just shows how to use the Equation class
To see an example of how to incorporate this into a GA seep EquationGA.py in the examples directory
"""
from Finch.FinchGA import generic

equation = """x**y+z"""
eq = generic.Equation(["x", "y", "z"], equation)
result = eq.evaluate([1, 2, 3])
print(result)

# returns 4
