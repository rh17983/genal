import random
import operator
import time
import math
import numpy as np
import matplotlib.pyplot as plt

# calculate fitness score
def calculate_fitness(x,y,z):
    return 2 * x * z * math.exp(-x) - 2 * math.pow(y, 3) + math.pow(y, 2) - 3 * math.pow(z, 3)

# create initial population
def initial_population(population_size):
    population = []

    for i in range(population_size):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        z = random.uniform(0, 100)
        population.append({'fit': 0.0, 'gene': [x,y,z]})

    return population


# set fitness score for each individ
def evaluate(population):
    for i in range(len(population)):  
        the_genome = population[i]["gene"]
        fitness = calculate_fitness(the_genome[0], the_genome[1], the_genome[2])
        population[i]["fit"] = fitness

    # sort just for get the individ with best score
    newlist = sorted(population, key=operator.itemgetter('fit'), reverse=True)

    return newlist, newlist[0]


### New poulation

# Create a new individ from 2 individs (meyosis)
def newIndivid(parent1, parent2):

    new_genome = [0.0, 0.0, 0.0]
    new_individ = {}

    parent1_gemome = parent1["gene"]
    parent2_gemome = parent2["gene"]

    for i in range(3):
        scaling = random.uniform(-0.25, 1.25)
        new_gene = parent1_gemome[i] * scaling + parent2_gemome[i] * (1 - scaling)

        if new_gene > 100:
            new_gene = 100

        if new_gene < 0:
            new_gene = 0

        new_genome[i] = new_gene

    new_individ = {'fit': 0.0, 'gene': new_genome}

    return new_individ

# Select parents (Tournament) and Mate
def generate_offspring(current_population, population_size):

    group_size = 5
    parents = []
    child_num = 3
    
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
    
        
    new_population = parents

    for i in range(int(len(parents)/2)):
        for j in range(child_num):
            new_population.append(newIndivid(parents[i], parents[len(parents) -1 -i]))

    return new_population

    
### Genetic Operators

# make recombination of genes of the individs in population
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
        

# make mutations of genes of the individs in population
def mutation(population):
    
    mutation_probability = 0.1
    
    mean, sigma = 0, 5 # mean and standard deviation
    for i in range(len(population)):
        for j in range(3):
            if random.random() < mutation_probability:
                rnd_value = np.random.normal(mean, sigma, 1)[0]
                new_gene = population[i]['gene'][j] + rnd_value
                
                if new_gene > 100:
                    new_gene = 100

                elif new_gene < 0:
                    new_gene = 0
                    
                population[i]['gene'][j]= new_gene
    
    return population


### Plot

# best population scores
def show_best_scores(bests, best_of_the_bests):
    plt.plot(bests)
    plt.title("Generations Bests")
    plt.xlabel('Iteration')
    plt.ylabel('Best Score of the Population')
    plt.axis([0, len(bests), 0, best_of_the_bests * 1.5])
    plt.show()

        
#######################################################################################
### MAIN
#######################################################################################

### global parameters
population_size = 100
time = 300
progress = []

# initial population
population = initial_population(population_size)

for i in range (time):
    
    # set fitness scores for each individual
    population, winner = evaluate(population)

    # winner score in population
    progress.append(winner["fit"])
    
    # new population
    population = generate_offspring(population, population_size)
    
    # crossover
    population = crossover(population)

    # mutate
    population = mutation(population)


#for i in range(time):
#   print(progress[i])

print ("Best Fitness: " + str(winner["fit"])+'\n')
print("Winner: ",str(winner["gene"][0]),str(winner["gene"][1]),str(winner["gene"][2]))


show_best_scores(progress, winner["fit"])