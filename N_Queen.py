

# In this program, we solve the N-Queens problem using a genetic algorithm to find the best solution.
# The method follows these steps:

#               1 - Initialize the population
#               2 - Calculate fitness
#               3 - Selection
#               4 - Crossover
#               5 - Mutation
#               6 - Apply crossover and mutation to the population
#               7 - Implement the N-Queens Genetic Algorithm



import numpy as np

# Step 1: Initialize population
def initPop(popSize, N):
    return np.random.randint(N, size=(popSize, N))



# Step 2: Calculate fitness


def calculateFitness(population):
    fitnessValues = []
    for x in population:
        penalty = 0
        for i in range(len(x)):  
            r = x[i]
            for j in range(len(x)):
                if i == j:
                    continue
                distance = abs(i - j)
                if x[j] in [r, r - distance, r + distance]:
                    penalty += 1
        fitnessValues.append(penalty)
    return -np.array(fitnessValues)

# Step 3: Selection


def selection(population, fitnessValues):
    probabilities = fitnessValues - fitnessValues.min() + 1
    if probabilities.sum() == 0:
        probabilities = np.ones_like(probabilities) / len(probabilities)
    else:
        probabilities = probabilities / probabilities.sum()
    N = len(population)
    indices = np.arange(N)
    selectedIndices = np.random.choice(indices, size=N, p=probabilities)
    selectedPopulation = population[selectedIndices]
    return selectedPopulation

# Step 4: Crossover


def crossOver(parent1, parent2, pc):
    r = np.random.random()
    if r < pc:
        m = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:m], parent2[m:]])
        child2 = np.concatenate([parent2[:m], parent1[m:]])
    else:
        child1, child2 = parent1.copy(), parent2.copy()
    return child1, child2

# Step 5: Mutation


def mutation(individual, pm, N):
    if np.random.random() < pm:
        m = np.random.randint(len(individual))
        new_value = np.random.randint(N)
        while new_value == individual[m]:  
            new_value = np.random.randint(N)
        individual[m] = new_value
    return individual

# Step 6: Crossover and Mutation on Population


def crossover_mutation(selectedPopulation, pc, pm, N):
    populationSize = len(selectedPopulation)
    newPopulation = np.empty((populationSize, N), dtype=int)
    for i in range(0, populationSize - 1, 2):
        parent1, parent2 = selectedPopulation[i], selectedPopulation[i + 1]
        child1, child2 = crossOver(parent1, parent2, pc)
        newPopulation[i], newPopulation[i + 1] = child1, child2
    if populationSize % 2 != 0:
        newPopulation[-1] = selectedPopulation[-1]
    for i in range(populationSize):
        mutation(newPopulation[i], pm, N)
    return newPopulation

# N-Queens Genetic Algorithm Implementation

def nQueens(popSize, maxGenerations, N, pc=0.7, pm=0.01):
    
    try :
            
            population = initPop(popSize, N)
            bestFitnessOverAll = None
            no_improvement_count = 0
            max_no_improvement = 50

            for i_gen in range(maxGenerations):
                fitnessValues = calculateFitness(population)
                best_i = fitnessValues.argmax()
                bestFitness = fitnessValues[best_i]

                if bestFitnessOverAll is None or bestFitness > bestFitnessOverAll:
                    bestFitnessOverAll = bestFitness
                    bestSolution = population[best_i]
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                print(f'\rGeneration: {i_gen:06} | Best Fitness: {-bestFitnessOverAll:03}', end='')

                if bestFitness == 0:
                    print("\nFound optimal solution!")
                    break

                if no_improvement_count >= max_no_improvement:
                    print("\nEarly stopping due to lack of improvement.")
                    break

                selectedPopulation = selection(population, fitnessValues)
                population = crossover_mutation(selectedPopulation, pc, pm, N)

            print("\nBest Solution:", bestSolution)

    except KeyboardInterrupt : 
         print("\nExecution stopping  by user. Stopping gracefully.")
         
         
         
         
         
         
if __name__ == "__main__":
    
    N = int(input("Enter the size of your chessboard : "))
    nQueens(popSize=100, maxGenerations=1000, N=N, pc=0.7, pm=0.01)
