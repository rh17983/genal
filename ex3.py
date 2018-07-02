import random
import operator
import time
import matplotlib.pyplot as plt
import math


# First population initialization
def generateFirstPopulation(population_size):
    population = []

    score = 0.0
    i = 0
    
    while i < population_size:
        x = random.random() * 100
        y = random.random() * 100
        z = random.random() * 100
        population.append({'score': score, 'genome': [x,y,z]})
        i+=1

    return population


# ----------------------- Scoring & Breeders selection --------------------

# get individual score - value of function for the individual
def get_score (x,y,z):

    #2xz exp(-x) - 2y^3 + y^2 - 3z^3
    score = 2 * x * z * math.exp(-x) - 2 * math.pow(y, 3) + math.pow(y, 2) - 3 * math.pow(z, 3)

    return score

# set individual score for each individual in current population
def scorePopulation(population):

    individual_score = 0.0
    
    population_best_score = 0.0
    
    population_scored = []

    population_scores = {}
    population_gemomes = {}
    
    i = 0
    for individual in population:
        
        gemome = individual["genome"]
        individual_score = get_score(gemome[0], gemome[1], gemome[2])

        if individual_score > population_best_score:
            population_best_score = individual_score

        population_scores[str(i)] = individual_score
        population_gemomes[str(i)] = gemome

        i += 1

    # sort population_scores by the score. top scores first
    population_scores_sorted = sorted(population_scores.items(), key = operator.itemgetter(1), reverse=True)

    for uu in population_scores_sorted:

        # individual score
        score = uu[1]

        # individual genome
        individual_key = uu[0]
        gemome = population_gemomes[individual_key]
        
        # add the individual to the scored & sorted population array
        population_scored.append({'score': score, 'genome': gemome})
    
    # return scored & sorted population array, the best score of the generation, the individual with the best score in the generation 
    return population_scored, population_best_score, population_scored[0]


# select individuals for breeding
def selectBreeders(population, top_score_individuals_count, random_score_individuals_count):
	
    breeders = []

    # select top score individuals
    for i in range(top_score_individuals_count):
        breeders.append(population[i])

    # select random score individuals
    for i in range(random_score_individuals_count):
        breeders.append(random.choice(population))
    
    random.shuffle(breeders)
    
    return breeders

# ------------------------------------------ Breeding --------------------

# Intermediate recombination.
def createChild(parent1, parent2):

    child_genome = [0.0, 0.0, 0.0]
    child = {}

    d = 0.25 # d defines the size of the area for possible offspring

    parent1_gemome = parent1["genome"]
    parent2_gemome = parent2["genome"]

    for i in range(3):
        alfa = random.uniform(-d, 1 + d) # scaling factor.
        new_value = parent1_gemome[i] * alfa + parent2_gemome[i] * (1 - alfa)

        if new_value > 100:
            new_value = 100

        if new_value < 0:
            new_value = 0

        child_genome[i] = new_value

    child = {'score': 0.0, 'genome': child_genome}

    return child

# create new population
def createBroodPopulation(breeders, number_of_child):
    
    broodPopulation = []
    
    for i in range(int(len(breeders)/2)):
        for j in range(number_of_child):
            broodPopulation.append(createChild(breeders[i], breeders[len(breeders) -1 -i]))

    return broodPopulation

# ------------------------------------------   Mutation --------------------

# mutate the individuum. Real Valued Mutation.
def mutate(individual):

    mutation_rate = 1/3 # probability of mutation of the certain gene (1/n).

    mutation_range = 0.50 # less this parameter value is, more probabbility to stuck to the local maximum (0.037) (99% when the value is 0.1).  
    mutation_precision = 8 # minimal step-size possible
    variables_domain = 100

    for i in range(3):
        if random.random() < mutation_rate:

            sign = random.choice([-1, 1]) # direction of mutation
            u = random.random()
            step = sign * (mutation_range * variables_domain) * math.pow(2, -u * mutation_precision) # Mühlenbein, H. and Schlierkamp-Voosen, D.: Predictive Models for the Breeder Genetic Algorithm: I. Continuous Parameter Optimization. Evolutionary Computation, 1 (1), pp. 25-49, 1993.

            new_value = individual["genome"][i] + step

            if new_value > 100:
                new_value = 100

            if new_value < 0:
                new_value = 0

            individual["genome"][i] = new_value

    return individual

# mutate individuals in the popluation
def mutatePopulation(population, individual_mutation_prob):

    for i in range(len(population)):
        if random.random() * 100 < individual_mutation_prob:
            population[i] = mutate(population[i])

    return population

# ------------------------ Visualisation -----------------------------

# plot the best scores within the generations
def best_scores_plt(best_scores):
	plt.axis([0,len(best_scores),0, 2*best_individuum["score"]])
	plt.title("Maximization")

	plt.plot(best_scores)
	plt.ylabel('Generation Best Score')
	plt.xlabel('Generation')
	plt.show()


# ----------------------- main

# global parameters
population_size = 100

top_score_individuals_count = 20 # count of individuals with top best fitness score to be selected for breeding
random_score_individuals_count = 20 # count of individuals with random fitness score to be selected for breeding

number_of_child = 5 # number of chiled for each couple

iterations_count = 100 # limit of generations
individual_mutation_prob = 5 # probability of mutation for the individual

# main
if ((top_score_individuals_count + random_score_individuals_count) / 2 * number_of_child != population_size):
	print ("The size of population is not stable")
else:
    # best scores of the generations
    best_scores = []

    # 0. create initial population
    theGeneration = generateFirstPopulation(population_size)

    for i in range (iterations_count):

        # 1. calculate the score (function value) for each individual
        populationScored, best_score, best_individuum = scorePopulation(theGeneration)

        # add the generation best score to the array
        best_scores.append(best_score)

        #print("\n populationScored \n")
        #print(populationScored)
        #quit()

        # 2. select individuals for breeding
        breeders = selectBreeders(populationScored, top_score_individuals_count, random_score_individuals_count)
        
        #print("\n breeders \n")
        #print(breeders)
        #quit()

        # 3. mate breeders and create new population by recombination
        populationNew = createBroodPopulation(breeders, number_of_child)
        
        #print("\n populationNew \n")
        #print(populationNew)
        #quit()

        # 4. mutation
        theGeneration = mutatePopulation(populationNew, individual_mutation_prob)

        #print("\n theGeneration \n")
        #print(theGeneration)
        #quit()


    #for i in range (iterations_count):
       #print(best_scores[i])

    best_scores_plt(best_scores)

    print ("\nFinal:")
    print ("\nMaximum: " + str(best_individuum["score"]))
    print ("Variables: ")
    print(best_individuum["genome"][0])
    print(best_individuum["genome"][1])
    print(best_individuum["genome"][2])