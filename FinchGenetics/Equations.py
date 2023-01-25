
class Equation:
    def __init__(self, vars, equation, desired):  # TODO: make this class
        """
        :param vars: A list containing the variable names: ["x", "y"] must correspond to the index of the elements in an Individual
        :param args: The equation string to be evaluated. "x*y" or "import math\n math.pow(x, y)".
        """
        self.vars = vars
        self.equation = equation
        self.desired = desired

    def evaluate(self, individual):
        try:
            local_string = self.equation
            local_iter = iter(self.vars)
            for i in individual:
                local_string = local_string.replace(next(local_iter), str(i))
            return eval(local_string) #be wary that this is a security risk. TODO: replace later
        except (ZeroDivisionError,  OverflowError):
            return self.desired * -1