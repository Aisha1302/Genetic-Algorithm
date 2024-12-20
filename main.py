import random


# Fitness function: We aim to maximize f(x) = x^2
def fitness(x):
    return x ** 2


# Generate the initial population
def generate_population(pop_size, bit_length):
    return [random.randint(0, 2 ** bit_length - 1) for _ in range(pop_size)]


# Convert integer to binary string
def to_binary(x, bit_length):
    return format(x, f'0{bit_length}b')


# Convert binary string to integer
def to_integer(binary_str):
    return int(binary_str, 2)


# Selection: Tournament Selection
def tournament_selection(population, fitness_values, tournament_size=3):
    selected = random.sample(list(zip(population, fitness_values)), tournament_size)
    return max(selected, key=lambda x: x[1])[0]


# Crossover: One-point crossover
def crossover(parent1, parent2, crossover_rate=0.7):
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 1)
        offspring1 = parent1[:point] + parent2[point:]
        offspring2 = parent2[:point] + parent1[point:]
        return to_integer(offspring1), to_integer(offspring2)
    else:
        return parent1, parent2


# Mutation: Flip random bits
def mutation(individual, mutation_rate=0.01):
    binary_str = to_binary(individual, len(to_binary(individual, 5)))
    mutated_str = ''.join(
        bit if random.random() > mutation_rate else str(1 - int(bit)) for bit in binary_str
    )
    return to_integer(mutated_str)


# Genetic Algorithm
def genetic_algorithm(pop_size, bit_length, generations, crossover_rate=0.7, mutation_rate=0.01):
    population = generate_population(pop_size, bit_length)

    for generation in range(generations):
        # Evaluate fitness for all individuals in the population
        fitness_values = [fitness(individual) for individual in population]

        # Selection: Create a new population via tournament selection
        new_population = []
        for _ in range(pop_size // 2):
            parent1 = tournament_selection(population, fitness_values)
            parent2 = tournament_selection(population, fitness_values)

            # Crossover to create offspring
            offspring1, offspring2 = crossover(to_binary(parent1, bit_length), to_binary(parent2, bit_length),
                                               crossover_rate)

            # Mutation
            offspring1 = mutation(offspring1, mutation_rate)
            offspring2 = mutation(offspring2, mutation_rate)

            # Add offspring to the new population
            new_population.extend([offspring1, offspring2])

        population = new_population

        # Find the best solution in the population
        best_individual = max(zip(population, fitness_values), key=lambda x: x[1])
        print(f"Generation {generation + 1}: Best Solution = {best_individual[0]}, Fitness = {best_individual[1]}")

    return best_individual


# Parameters
pop_size = 10  # Population size
bit_length = 5  # Number of bits representing an individual (x value range from 0 to 31)
generations = 20  # Number of generations to evolve

# Run the Genetic Algorithm
best_solution = genetic_algorithm(pop_size, bit_length, generations)
print(f"Best solution after {generations} generations: {best_solution[0]} with fitness: {best_solution[1]}")
