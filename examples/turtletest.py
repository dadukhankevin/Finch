import turtle
from Finch.genetics import genepools as gp
from Finch.environmental.environments import *
from Finch.environmental.layers import *
import matplotlib.pyplot as plt
from time import sleep

wn = turtle.Screen()
t = turtle.Turtle()

sleep(10)
t.shape('turtle')
t.speed(0)
max_dist = 0
def fit(individual):
    for i in individual:
        t.right(i)
        t.fd(i)
    distance = t.distance((0,0))
    #t.clear()
    t.penup()
    t.home()
    t.pendown()
    print(distance)
    return distance

def singular_fit(individual):
    global max_dist
    for i, v in enumerate(individual):
        if i % 2 == 0:
            t.rt(v)
        else:
            t.fd(v)

    distance = t.distance((0,0))
    if distance > max_dist:
        t.clear()
        t.color("yellow")
        t.stamp()
        t.color("black")
        max_dist = distance
    t.penup()
    t.home()
    t.pendown()
    return distance
gene_pool = gp.IntPool(-1, 1, 40, singular_fit)

environment = Sequential(layers=[
    Populate(gene_pool, 8),
    IntMutateAmount(1, 7, gene_pool, min_mutation=-20, max_mutation=20),
    SortByFitness(),
    CapPopulation(7),

])

environment.evolve(100)
print(environment.history)
print(environment.individuals[-1].genes)
plt.plot(environment.history)
plt.show()
plt.plot(environment.individuals[-1].genes)
plt.show()