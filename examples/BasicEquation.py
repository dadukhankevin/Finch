import math

from Finch.FinchGA import generic

equation = """math.pow(x,y)+z"""
eq = generic.Equation(["x", "y", "z"], equation)
result = eq.evaluate([1, 2, 3])
print(result)

# returns 4
