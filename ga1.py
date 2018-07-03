import random
import operator
import time
import math
import numpy as np
import matplotlib.pyplot as plt

def calculate_fitness(x,y,z):
    return 2 * x * z * math.exp(-x) - 2 * math.pow(y, 3) + math.pow(y, 2) - 3 * math.pow(z, 3)

def initial_population(population_size):
    population = []

    for i in range(population_size):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        z = random.uniform(0, 100)
        population.append({'fit': 0.0, 'gene': [x,y,z]})

    return population


# 
def evaluate(population):
    for i in range(len(population)):  
        the_genome = population[i]["gene"]
        fitness = calculate_fitness(the_genome[0], the_genome[1], the_genome[2])
        population[i]["fit"] = fitness

    newlist = sorted(population, key=operator.itemgetter('fit'), reverse=True)

    return newlist, newlist[0]

# ------------------------------------------ Breeding --------------------

# Intermediate recombination.
def createChild(parent1, parent2):

    child_genome = [0.0, 0.0, 0.0]
    child = {}

    d = 0.25 # d defines the size of the area for possible offspring

    parent1_gemome = parent1["gene"]
    parent2_gemome = parent2["gene"]

    for i in range(3):
        alfa = random.uniform(-d, 1 + d) # scaling factor.
        new_value = parent1_gemome[i] * alfa + parent2_gemome[i] * (1 - alfa)

        if new_value > 100:
            new_value = 100

        if new_value < 0:
            new_value = 0

        child_genome[i] = new_value

    child = {'fit': 0.0, 'gene': child_genome}

    return child

# Create new population
def createBroodPopulation(breeders, number_of_child):
    
    broodPopulation = []
    
    for i in range(int(len(breeders)/2)):
        for j in range(number_of_child):
            broodPopulation.append(createChild(breeders[i], breeders[len(breeders) -1 -i]))

    return broodPopulation

# Selection of parents (Tournament) and Mating
def generate_offspring(current_population, population_size):

    group_size = 5
    parents = []
    
    for i in range(40):
        parent = 0
        group_max_score = 0
        
        for j in range(group_size):
            index = random.randint(0, population_size-1)
            
            if j == 0:
                parent = index
                group_max_score = current_population[index]['fit']
            else:
                if current_population[index]['fit'] > group_max_score:
                    parent = index
                    group_max_score = current_population[index]['fit']

        parents.append(current_population[parent])
    
        
    new_generation = createBroodPopulation(parents, 3) + parents


    return new_generation
    
# ------------------------------------------ Breeding --------------------

def crossover(population):
    
    # probability of pair crossover
    crossover_rate = 0.6
    
    # individs for crossover
    pairs = []

    # IDs of individs in population
    individs_ids = list(range(0, 100))

    k = 0

    while len(individs_ids) > 0:
        
        individ1_ID = random.choice(individs_ids)
        individs_ids.remove(individ1_ID)
        
        individ2_ID = random.choice(individs_ids)
        individs_ids.remove(individ2_ID)
        
        pairs.append(individ1_ID)
        pairs.append(individ2_ID)
    
    
    while k < len(pairs):
        
        # determine if this pair will crossover
        chance_to_crossover = random.random()
        
        if chance_to_crossover < crossover_rate:
            
            # if point == 1 then the crossover will be done after 0th gene
            point = random.randint(1,2)
            
            for i in range(point):
                temp_gene = population[pairs[k]]['gene'][i]
                population[pairs[k]]['gene'][i] = population[pairs[k+1]]['gene'][i]
                population[pairs[k+1]]['gene'][i] = temp_gene
        k+=2
    return population
        

def mutation(population):
    mutation_probability = 0.1
    
    mu, sigma = 0, 5 # mean and standard deviation
    for i in range(len(population)):
        for j in range(3):
            if random.random() < mutation_probability:
                rnd_value = np.random.normal(mu, sigma, 1)[0]
                # rnd_value = random.uniform(-50, 50)
                new_value = population[i]['gene'][j] + rnd_value
                
                if new_value > 100:
                    new_value = 100
                elif new_value < 0:
                    new_value = 0
                    
                population[i]['gene'][j]= new_value
    return population


# -- Plot ----------------------

# best population scores
def show_best_scores(bests, best_of_the_bests):
    plt.plot(bests)
    plt.title("Generations Bests")
    plt.xlabel('Iteration')
    plt.ylabel('Best Score of the Population')
    plt.axis([0, len(bests), 0, 1.5 * best_of_the_bests])
    plt.show()

        
#######################################################################################
### MAIN
#######################################################################################

population_size = 100
time = 300
progress = []

population = initial_population(population_size)

for i in range (time):
    
    # set fitness scores for each individual
    population, winner = evaluate(population)

    # winner score in population
    progress.append(winner["fit"])
    
    # new population
    population = generate_offspring(population, population_size)
    
    population = crossover(population)

    # mutate population
    population = mutation(population)


for i in range(time):
   print(progress[i])

print ("Best Fitness: " + str(winner["fit"])+'\n')
print("Winner: ",str(winner["gene"][0]),str(winner["gene"][1]),str(winner["gene"][2]))


show_best_scores(progress, winner["fit"])