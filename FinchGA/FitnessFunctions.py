
class ValueWeightFunction:
    def __init__(self, maxweight):
        self.maxweight = maxweight
    def func(self, individual):
        weight = 0
        value = 0
        for i in individual:
            weight += float(i[2])
            value += float(i[1])
        if weight > self.valthresh:
            return 0
        else:
            return value
