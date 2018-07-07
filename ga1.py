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
    
    fitness_sum = 0

    for i in range(len(population)):
        
        the_genome = population[i]["gene"]
        
        fitness = calculate_fitness(the_genome[0], the_genome[1], the_genome[2])
        population[i]["fit"] = fitness
        
        fitness_sum += fitness

    # sort just for get the individ with best score
    newlist = sorted(population, key=operator.itemgetter('fit'), reverse=True)

    # get mean
    fitness_mean = fitness_sum / len(population)

    return newlist, newlist[0], fitness_mean


### New poulation

# Create a new individ from 2 parent individs (meyosis)
def newIndivid(parent1, parent2):

    gene_count = 3
    new_genome = [0.0, 0.0, 0.0]
    new_individ = {}

    parent1_gemome = parent1["gene"]
    parent2_gemome = parent2["gene"]

    for i in range(gene_count):
        scaling = random.uniform(0, 1)
        new_gene = parent1_gemome[i] * scaling + parent2_gemome[i] * (1 - scaling)

        if new_gene > 100:
            new_gene = 100

        if new_gene < 0:
            new_gene = 0

        new_genome[i] = new_gene

    new_individ = {'fit': 0.0, 'gene': new_genome}

    return new_individ

# Select parents (Tournament) and mate the selected breeders
# Inputs:
#   current_population - array of the current popupation individuals
#   population_size - population size
#   group_size - subpart of the population for the tournament selection
#   parents_count - number of the individuals to be selected for breeding
#   child_num - number of children each breeders couple should have

def generate_offspring(current_population, population_size, group_size, parents_count, child_num):

    # array of the individuals selected for breeding
    parents = [] 

    for i in range(parents_count):
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

# make recombination of genes of the individs in population (point crossing over)
def crossover(population, population_size, crossover_rate):
    
    # individs for crossover
    pairs = []

    # IDs of individs in population
    individs_ids = list(range(0, population_size))

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
def mutation(population, individ_mutation_probability, gene_mutation_probability, mutation_step_size):

    # mean and standard deviation (used for mutation step size determination)
    mean, sigma = 0, mutation_step_size

    for i in range(len(population)):
        if random.random() < individ_mutation_probability:
            for j in range(3):
                if random.random() < gene_mutation_probability:
                    
                    rnd_value = np.random.normal(mean, sigma, 1)[0]
                    new_gene = population[i]['gene'][j] + rnd_value
                    
                    if new_gene > 100:
                        new_gene = 100

                    elif new_gene < 0:
                        new_gene = 0
                        
                    population[i]['gene'][j]= new_gene
    
    return population


### Plot

# population bests
def show_best_scores(bests):
    
    plt.title("Generations Bests")
    plt.xlabel('Iteration')
    plt.ylabel('Population Best Score')
    plt.axis([0, len(bests), 0, max(bests) * 1.5])

    plt.plot(bests)
    plt.show()

# population means
def show_mean_scores(means):
    
    plt.title("Generations Means")
    plt.xlabel('Iteration')
    plt.ylabel('Population Mean Score')
    plt.axis([0, len(means), 0, max(means) * 1.5])

    plt.plot(means)
    plt.show()

        
#######################################################################################
### MAIN
#######################################################################################

### global parameters

# population size
population_size = 100

# max iterations count
time = 300

# subpart of the population for the tournament selection
group_size = 5

# number of the individuals to be selected for breeding (selected parents go to the new population TOGETHER with their children)
parents_count = 42

# number of children each breeders couple should have
child_num = 3

# probability of the mutation of the certrain individ of the population (optiomal = 25)
individ_mutation_probability = 0.28

# probability of the mutation of the certrain gene of the certain individual (value = 1 / genes count)
gene_mutation_probability = 1/3

# mutation step size (deviation of normal distribution with mean = 0). Values = 1 - 50 (optiomal = 25)
mutation_step_size = 25

# probability of certain pair crossover
crossover_rate = 0.5


### utils
progress = []
means    = []


### run
if (parents_count * (1 + 1/2 * child_num) < population_size) or (group_size > population_size):
    print ("Population is not stable\n")
else:
    # initial population
    population = initial_population(population_size)

    for i in range (time):
        
        # set fitness scores for each individual
        population, winner, population_mean = evaluate(population)

        # collect population best scores
        progress.append(winner["fit"])

        # collect population means
        means.append(population_mean)
        
        # new population
        population = generate_offspring(population, population_size, group_size, parents_count, child_num)
        
        # crossover over new population
        population = crossover(population, population_size, crossover_rate)

        # mutate over new population
        population = mutation(population, individ_mutation_probability, gene_mutation_probability, mutation_step_size)


    #for i in range(time):
    #   print(progress[i])

    print ("Best Fitness: " + str(winner["fit"])+'\n')
    print("Winner: ",str(winner["gene"][0]),str(winner["gene"][1]),str(winner["gene"][2]))

    show_best_scores(progress)
    # show_mean_scores(means)