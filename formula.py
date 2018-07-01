import random
import operator
import time
import matplotlib.pyplot as plt
import math

def get_score (x,y,z):

    #2xz exp(-x) - 2y^3 + y^2 - 3z^3
    score = 2 * x * z * math.exp(-x) - 2 * math.pow(y, 3) + math.pow(y, 2) - 3 * math.pow(z, 3)

    return score


print(get_score(25.137657389661136, -43.985728309642205, 3.5651857882730464))

print(random.choice([-1, 1]))