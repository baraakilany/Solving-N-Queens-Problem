import random
import time
import os
import psutil


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def solve_n_queens_genetic(n, population_size=100, generations=1000, mutation_rate=0.1):
    """
    Solves the N-Queens problem using a Genetic Algorithm.

    Returns:
        tuple: (best_individual, final_conflicts, time_taken, peak_memory)
    """
    start_time = time.time()
    max_fitness = (n * (n - 1)) / 2

    population = []
    for _ in range(population_size):
        individual = list(range(n))
        random.shuffle(individual)
        population.append(individual)

    best_individual = None
    best_fitness = -1

    for gen in range(generations):
        fitness_scores = []
        for individual in population:
            conflicts = calculate_conflicts(individual)
            fitness = max_fitness - conflicts
            fitness_scores.append(fitness)

            if fitness == max_fitness:
                end_time = time.time()
                return individual, 0, end_time - start_time, get_memory_usage()

        current_best_fitness = max(fitness_scores)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[fitness_scores.index(current_best_fitness)]

        new_population = []
        for _ in range(population_size):
            p1_idx = random.randint(0, population_size - 1)
            p2_idx = random.randint(0, population_size - 1)
            if fitness_scores[p1_idx] > fitness_scores[p2_idx]:
                new_population.append(population[p1_idx])
            else:
                new_population.append(population[p2_idx])
        population = new_population

        children = []
        for i in range(0, population_size, 2):
            parent1 = population[i]
            if i + 1 < population_size:
                parent2 = population[i + 1]
                crossover_point = random.randint(1, n - 1)
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]
                child1 = repair_child(child1, n)
                child2 = repair_child(child2, n)

                if random.random() < mutation_rate:
                    child1 = mutate(child1)
                if random.random() < mutation_rate:
                    child2 = mutate(child2)
                children.append(child1)
                children.append(child2)
            else:
                children.append(parent1)
        population = children

    end_time = time.time()
    peak_memory = get_memory_usage()
    final_conflicts = int(max_fitness - best_fitness)
    return best_individual, final_conflicts, end_time - start_time, peak_memory


def calculate_conflicts(state):
    conflicts = 0
    n = len(state)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) == abs(state[i] - state[j]):
                conflicts += 1
    return conflicts


def repair_child(child, n):
    missing = []
    counts = {i: 0 for i in range(n)}
    for gene in child:
        counts[gene] += 1
    for i in range(n):
        if counts[i] == 0:
            missing.append(i)
    for i in range(n):
        if counts[child[i]] > 1:
            counts[child[i]] -= 1
            child[i] = missing.pop(0)
    return child


def mutate(individual):
    n = len(individual)
    idx1, idx2 = random.sample(range(n), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual


def print_solution(state):
    if not state:
        print("No state to print.")
        return
    n = len(state)
    board = [['.' for _ in range(n)] for _ in range(n)]
    for row, col in enumerate(state):
        board[row][col] = 'Q'
    for r in board:
        print(" ".join(r))


if __name__ == '__main__':
    N = 200
    print(f"--- Solving N-Queens for N={N} with a Genetic Algorithm ---")

    final_state, final_conflicts, time_taken, memory_used = solve_n_queens_genetic(N)

    print("\nFinal Board State:")
    print_solution(final_state)

    print(f"\nNumber of conflicts: {final_conflicts}")
    if final_conflicts == 0:
        print("A solution was found!")
    else:
        print("No perfect solution found. The above is the best state achieved.")

    print(f"Time taken: {time_taken:.6f} seconds")
    print(f"Memory used: {memory_used:.4f} MB")