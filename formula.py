import random
import operator
import time
import matplotlib.pyplot as plt
import math

def get_score (x,y,z):

    #2xz exp(-x) - 2y^3 + y^2 - 3z^3
    score = 2 * x * z * math.exp(-x) - 2 * math.pow(y, 3) + math.pow(y, 2) - 3 * math.pow(z, 3)

    return score


print(get_score(45.51319970332328, 0.33333333363504464, 6.208191667852839e-07))