import random
import operator
import time
import matplotlib.pyplot as plt
import math


# First population initialization
def generateFirstPopulation(sizePopulation):
    population = []

    score = 0.0
    i = 0
    
    while i < sizePopulation:
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

    population_scores_sorted = sorted(population_scores.items(), key = operator.itemgetter(1), reverse=True)

    for uu in population_scores_sorted:

        key = uu[0]
        score = uu[1]
        
        gemome = population_gemomes[key]
        population_scored.append({'score': score, 'genome': gemome})
    
    return population_scored, population_best_score, population_scored[0]


# select individuals for breeding
def selectBreeders(population, best_sample, lucky_few):
	
    breeders = []

    for i in range(best_sample):
        breeders.append(population[i])

    for i in range(lucky_few):
        breeders.append(random.choice(population))
    
    random.shuffle(breeders)
    
    return breeders

# ------------------------------------------ Breeding --------------------

# Intermediate recombination. Ref - http://www.geatbx.com/docu/algindex-03.html
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
    
    #print("\n Child: ")
    #print(child)
    #quit()

    return child

# create new population
def createBroodPopulation(breeders, number_of_child):
    
    broodPopulation = []
    
    for i in range(int(len(breeders)/2)):
        for j in range(number_of_child):
            broodPopulation.append(createChild(breeders[i], breeders[len(breeders) -1 -i]))

    #print("\n\n")
    #print(broodPopulation)
    #quit()

    return broodPopulation

# ------------------------------------------   Mutation --------------------

# mutate the individuum. Real Valued Mutation. Ref - http://www.geatbx.com/docu/algindex-04.html
def mutate(individual):

    mutation_range = 0.50 # less this parameter value is, more probabbility to stuck to the local maximum (0.037) (99% wheb the value is 0.1).  
    mutation_precision = 8 # minimal step-size possible
    variables_domain = 100

    mutation_rate = 1/3
    
    for i in range(3):
        if random.random() < mutation_rate:

            sign = random.choice([-1, 1])
            u = random.random()
            step = sign * (mutation_range * variables_domain) * math.pow(2, -u * mutation_precision)

            new_value = individual["genome"][i] + step

            if new_value > 100:
                new_value = 100

            if new_value < 0:
                new_value = 0

            individual["genome"][i] = new_value

    return individual

# mutate individuals in the popluation
def mutatePopulation(population, chance_of_mutation):

    for i in range(len(population)):
        if random.random() * 100 < chance_of_mutation:
            population[i] = mutate(population[i])

    return population

# ------------------------ Visualisation -----------------------------

# plot the best scores within the generations
def best_scores_plt(best_scores):
	plt.axis([0,len(best_scores),0, 2*best_individuum["score"]])
	plt.title("Best Fitness")

	plt.plot(best_scores)
	plt.ylabel('fitness best individual')
	plt.xlabel('generation')
	plt.show()


# ----------------------- main

# global parameters
size_population = 100
best_sample = 20 # count of individuals with top best fitness score to be selected for breeding
lucky_few = 20 # count of individuals with random fitness score to be selected for breeding
number_of_child = 5 # number of chiled for each couple
iterations_count = 100 # limit of generations
chance_of_mutation = 5 # probability of mutation for the individual

# main
if ((best_sample + lucky_few) / 2 * number_of_child != size_population):
	print ("The size of population is not stable")
else:

    # initial population
    theGeneration = generateFirstPopulation(size_population)

    # array with populations from all iterations
    populations = []
    populations.append(theGeneration)

    best_scores = []

    for i in range (iterations_count):

        # calculate the function value for each individual
        populationScored, best_score, best_individuum = scorePopulation(theGeneration)

        best_scores.append(best_score)

        #print("\n populationScored \n")
        #print(populationScored)
        #quit()

        # select individuals for breeding
        breeders = selectBreeders(populationScored, best_sample, lucky_few)
        
        #print("\n breeders \n")
        #print(breeders)
        #quit()

        # mate breeders and create new population by recombination
        populationNew = createBroodPopulation(breeders, number_of_child)
        
        #print("\n populationNew \n")
        #print(populationNew)
        #quit()

        # mutation
        theGeneration = mutatePopulation(populationNew, chance_of_mutation)
        
        #print("\n nextGeneration \n")
        #print(nextGeneration)
        #quit()

        populations.append(theGeneration)


    #for i in range (iterations_count):
       #print(best_scores[i])

    best_scores_plt(best_scores)

    print ("\nFinal:")
    print ("\nMaximum: " + str(best_individuum["score"]))
    print ("Variables: ")
    print(best_individuum["genome"][0])
    print(best_individuum["genome"][1])
    print(best_individuum["genome"][2])