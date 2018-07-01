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


def score_and_sort(population):

    population_scored = []

    for individual in population:
        gemome = individual["genome"]
        score = get_score(gemome[0], gemome[1], gemome[2])
        population_scored.append({'score': score, 'genome': gemome})

    return population_scored


def selectBreeders(population, best_sample, lucky_few):
	
    breeders = []

    for i in range(best_sample):
        breeders.append(population[i])

    for i in range(lucky_few):
        breeders.append(random.choice(population))
    
    #random.shuffle(breeders)
    
    return breeders

# ------------------------------------------   Mating --------------------

def createChild(individual1, individual2):

    child_genome = [0.0, 0.0, 0.0]
    child = []

    d = 0.25

    parent1_gemome = individual1[1][1]
    parent2_gemome = individual2[1][1]

    for i in range(3):
        alfa = random.uniform(-d, 1 + d)
        child_genome[i] = parent1_gemome[i] * alfa + parent2_gemome[i] * (1 - alfa)

    
    child.append([0.0, child_genome])
    
    #print("\n Child: ")
    #print(child)
    #quit()

    return child


def createChildren(breeders, number_of_child):
    
    nextPopulation = []
    
    for i in range(int(len(breeders)/2)):
        for j in range(number_of_child):
            nextPopulation.append(createChild(breeders[i], breeders[len(breeders) -1 -i]))

    #print("\n\n")
    #print(nextPopulation)
    #quit()

    return nextPopulation

# ------------------------------------------   Mutation --------------------

def mutateWord(word):

	index_modification = int(random.random() * len(word))

	if (index_modification == 0):
		word = chr(97 + int(26 * random.random())) + word[1:]
	else:
		word = word[:index_modification] + chr(97 + int(26 * random.random())) + word[index_modification+1:]

	return word

	
def mutatePopulation(population, chance_of_mutation):

    for i in range(len(population)):
        if random.random() * 100 < chance_of_mutation:
            population[i] = mutateWord(population[i])

    return population






def nextGeneration (firstGeneration, best_sample, lucky_few, number_of_child, chance_of_mutation):
    
    print("\n firstGeneration \n")
    print(firstGeneration)
    
    populationSorted = score_and_sort(firstGeneration) # desc sorted (by value) assoc array (gene => fitnes value)

    print("\n populationSorted \n")
    print(populationSorted)
    #quit()

    nextBreeders = selectBreeders(populationSorted, best_sample, lucky_few) # array of breeders [gene, .., gene]
    
    print("\n nextBreeders \n")
    print(nextBreeders)
    quit()

    nextPopulation = createChildren(nextBreeders, number_of_child) # mating. shuffle crossover. array of genes of new population [gene, .., gene]
    
    print("\n nextPopulation \n")
    print(nextPopulation)
    #quit()

    #nextGeneration = mutatePopulation(nextPopulation, chance_of_mutation) # mutation. array of genes of mutated population [gene, .., gene]

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
def getBestIndividualFromPopulation (population, password):
	return score_and_sort(population)[0]

def getListBestIndividualFromHistorique (historic, password):
	bestIndividuals = []
	for population in historic:
		bestIndividuals.append(getBestIndividualFromPopulation(population, password))
	return bestIndividuals

#graph
def evolutionBestFitness(historic, password):
	plt.axis([0,len(historic),0,105])
	plt.title(password)
	
	evolutionFitness = []
	for population in historic:
		evolutionFitness.append(getBestIndividualFromPopulation(population, password)[1])
	plt.plot(evolutionFitness)
	plt.ylabel('fitness best individual')
	plt.xlabel('generation')
	plt.show()

def evolutionAverageFitness(historic, password, size_population):
	plt.axis([0,len(historic),0,105])
	plt.title(password)
	
	evolutionFitness = []
	for population in historic:
		populationPerf = score_and_sort(population)
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
number_of_generation = 50
chance_of_mutation = 5

#program
if ((best_sample + lucky_few) / 2 * number_of_child != size_population):
	print ("population size not stable")
else:
	historic = multipleGeneration(number_of_generation, size_population, best_sample, lucky_few, number_of_child, chance_of_mutation)
	
	#printSimpleResult(historic, password, number_of_generation)
	#evolutionBestFitness(historic, password)
	#evolutionAverageFitness(historic, password, size_population)

print(time.time() - temps1)