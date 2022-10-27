class Rates:
    def __init__(self, rate, arg):
        self.rate = rate
        self.arg = arg

    def slow(self):
        """Decreases rate by percent"""
        self.rate = self.rate * (1 -  self.arg)
        return round(self.rate, 5)

    def speed(self):
        self.rate = self.rate * (1 +  self.arg)
        return round(self.rate, 5)

    def multiply(self):
        self.rate = self.rate * self.arg
        return self.rate

    def constant(self):
        return self.rate

    def divide(self):
        self.rate = self.rate / self.arg
        return self.rate
    def exponential(self):
        self.rate = self.rate ** self.arg
        return self.rate
    def add(self):
        self.rate += self.arg
        return self.rate
