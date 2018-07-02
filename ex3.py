#!/usr/bin/python3.5
# -*-coding:Utf-8 -*

import random
import operator
import time
import matplotlib.pyplot as plt
import math

temps1 = time.time()

# ------------------------------------------   First population initialization --------------------

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


# ------------------------------------------   Scoring & Breeders selection --------------------

def get_score (x,y,z):

    #2xz exp(-x) - 2y^3 + y^2 - 3z^3
    score = 2 * x * z * math.exp(-x) - 2 * math.pow(y, 3) + math.pow(y, 2) - 3 * math.pow(z, 3)

    return score


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

    print ("\n population_best_score: ")
    print (population_best_score)
    print ("\n")

    population_scores_sorted = sorted(population_scores.items(), key = operator.itemgetter(1), reverse=True)

    for uu in population_scores_sorted:

        key = uu[0]
        score = uu[1]
        
        gemome = population_gemomes[key]
        population_scored.append({'score': score, 'genome': gemome})
    
    return population_scored


def selectBreeders(population, best_sample, lucky_few):
	
    breeders = []

    for i in range(best_sample):
        breeders.append(population[i])

    for i in range(lucky_few):
        breeders.append(random.choice(population))
    
    random.shuffle(breeders)
    
    return breeders

# ------------------------------------------   Mating --------------------

def createChild(parent1, parent2):

    child_genome = [0.0, 0.0, 0.0]
    child = {}

    d = 0.25

    parent1_gemome = parent1["genome"]
    parent2_gemome = parent2["genome"]

    for i in range(3):
        alfa = random.uniform(-d, 1 + d)
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

def mutate(individual):

    mutation_range = 0.1
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

            #print (step)

            individual["genome"][i] = new_value

    return individual

	
def mutatePopulation(population, chance_of_mutation):

    for i in range(len(population)):
        if random.random() * 100 < chance_of_mutation:
            population[i] = mutate(population[i])

    return population


# ------------------------------------------------------------------------------------------------------

def nextGeneration (firstGeneration, best_sample, lucky_few, number_of_child, chance_of_mutation):
    
    #print("\n firstGeneration \n")
    #print(firstGeneration)
    
    populationScored = scorePopulation(firstGeneration) # desc sorted (by value) assoc array (gene => fitnes value)

    #print("\n populationScored \n")
    #print(populationScored)
    #quit()

    breeders = selectBreeders(populationScored, best_sample, lucky_few) # array of breeders [gene, .., gene]
    
    #print("\n breeders \n")
    #print(breeders)
    #quit()

    broodPopulation = createBroodPopulation(breeders, number_of_child) # mating. shuffle crossover. array of genes of new population [gene, .., gene]
    
    #print("\n broodPopulation \n")
    #print(broodPopulation)
    #quit()

    nextGeneration = mutatePopulation(broodPopulation, chance_of_mutation) # mutation. array of genes of mutated population [gene, .., gene]
    
    #print("\n nextGeneration \n")
    #print(nextGeneration)
    #quit()

    return nextGeneration


def multipleGeneration(number_of_generation, size_population, best_sample, lucky_few, number_of_child, chance_of_mutation):
	
    historic = []
    historic.append(generateFirstPopulation(size_population))
	
    for i in range (number_of_generation):
        historic.append(nextGeneration(historic[i], best_sample, lucky_few, number_of_child, chance_of_mutation))

    return historic




#print result:
def printSimpleResult(historic, password, number_of_generation): #bestSolution in historic. Caution not the last
	result = getListBestIndividualFromHistorique(historic, password)[number_of_generation-1]
	print ("solution: \"" + result[0] + "\" de fitness: " + str(result[1]))

#analysis tools
def getBestIndividualFromPopulation (population):
	return scorePopulation(population)[0]

def getListBestIndividualFromHistorique (historic, password):
	bestIndividuals = []
	for population in historic:
		bestIndividuals.append(getBestIndividualFromPopulation(population))
	return bestIndividuals

#graph
def evolutionBestFitness(historic):
	plt.axis([0,len(historic),0,105])
	plt.title("Best Fitness")
	
	evolutionFitness = []
	for population in historic:
		evolutionFitness.append(getBestIndividualFromPopulation(population)[1])
	plt.plot(evolutionFitness)
	plt.ylabel('fitness best individual')
	plt.xlabel('generation')
	plt.show()

def evolutionAverageFitness(historic, password, size_population):
	plt.axis([0,len(historic),0,105])
	plt.title(password)
	
	evolutionFitness = []
	for population in historic:
		populationPerf = scorePopulation(population)
		averageFitness = 0
		for individual in populationPerf:
			averageFitness += individual[1]
		evolutionFitness.append(averageFitness/size_population)
	plt.plot(evolutionFitness)
	plt.ylabel('Average fitness')
	plt.xlabel('generation')
	plt.show()




size_population = 100
best_sample = 20
lucky_few = 20
number_of_child = 5
number_of_generation = 100
chance_of_mutation = 5

#program
if ((best_sample + lucky_few) / 2 * number_of_child != size_population):
	print ("population size not stable")
else:
	historic = multipleGeneration(number_of_generation, size_population, best_sample, lucky_few, number_of_child, chance_of_mutation)

print(time.time() - temps1)