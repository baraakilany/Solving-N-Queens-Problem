Solving the N-Queens Problem with Various Search Algorithms
This repository contains Python implementations for solving the classic N-Queens problem using four different algorithmic approaches: exhaustive search (Depth-First Search), Greedy Search (Hill-Climbing with random restarts), Simulated Annealing, and a Genetic Algorithm.

The project serves as a comparative study to analyze the trade-offs between these algorithms in terms of performance, efficiency, and scalability for solving this computationally difficult puzzle.

The N-Queens Problem
The N-Queens problem is a well-known puzzle in both mathematics and computer science where the objective is to place N chess queens on an N×N chessboard in such a way that no two queens threaten each other. This means that no two queens can share the same row, column, or diagonal. The problem's complexity grows exponentially with the board size (N), making it an excellent benchmark for testing the performance of various search and optimization algorithms.

Algorithmic Approaches
This project implements and compares four distinct approaches to solving the N-Queens problem.

1. Exhaustive Search (Depth-First Search)
This approach uses a classic recursive backtracking implementation of Depth-First Search (DFS) to systematically explore every possible placement of queens. While this method is guaranteed to find all possible solutions, its factorial time complexity (O(N!)) makes it impractical for board sizes larger than N=15.

To run the DFS implementation:

python "Exhaustive search (DFS).py"

2. Greedy Search (Hill Climbing)
This is a heuristic method that uses a standard hill-climbing approach with a random-restart mechanism. It begins with a random placement of queens and iteratively makes the move that reduces the number of conflicts the most. To avoid getting stuck in local optima, the algorithm restarts with a new random board if it cannot find a better state.

To run the Greedy Search implementation:

python "Greedy Search (Hill Climbing).py"

3. Simulated Annealing
Simulated Annealing is a more advanced heuristic inspired by the metallurgical process of annealing. It starts with a high "temperature," allowing the algorithm to make random moves, including those that increase conflicts, to explore the solution space broadly. As the temperature gradually cools, the algorithm becomes more selective, converging towards a low-conflict state. This probabilistic approach helps it escape local optima that might trap simpler greedy algorithms.

To run the Simulated Annealing implementation:

python "Simulated Annealing.py"

4. Genetic Algorithm
The Genetic Algorithm (GA) is a population-based heuristic. It evolves a population of potential solutions (boards) over many generations. In each generation, solutions are selected based on their "fitness" (lower conflicts). They then "reproduce" using crossover and mutation operations to create a new generation of solutions. This method proved to be the most effective at finding conflict-free solutions for large N in this study.

To run the Genetic Algorithm implementation:

python "Genetic Algorithm.py"

Methodology
All algorithms were implemented in Python 3 and designed to use a common state representation for the chessboard: a 1D list of N integers, where the list index represents the row and the value at that index represents the column of a queen. This efficient representation inherently prevents two queens from being in the same row, reducing the search space.

The heuristic algorithms used the following parameters:

Greedy Search: Implemented with a maximum of 200 random restarts.

Simulated Annealing: Used an initial temperature of 100 and a cooling rate of 0.995.

Genetic Algorithm: A population of 100 individuals was run for 1000 generations with a mutation rate of 10%.

Summary of Results
The comparative analysis revealed a clear trade-off between the guaranteed correctness of an exhaustive search and the efficiency of heuristic methods.

Depth-First Search (DFS): Guaranteed to find a solution but becomes practically unusable for board sizes larger than N=15 due to its exponential runtime.

Greedy Search: The fastest of the heuristic methods in terms of raw speed, but it was also the least effective, often getting stuck in local optima and failing to find a perfect solution for larger boards.

Simulated Annealing: Performed better than Greedy Search due to its ability to escape local optima. However, it still began to fail on very large board sizes, ending with several conflicts.

Genetic Algorithm: Proved to be the most successful heuristic overall, providing the best balance of speed, scalability, and accuracy. It consistently found conflict-free or near-conflict-free solutions for large-scale instances of the problem (up to N=200).

Memory consumption for all algorithms was low and stable, confirming that the primary challenge of the N-Queens problem is time complexity, not space complexity.

Conclusion
For small problems (N < 15) where time is not a concern, DFS is the best choice as it guarantees a correct solution. For larger, more real-world scale challenges, heuristic approaches are the only viable option. Among the tested heuristics, the Genetic Algorithm demonstrated the most robust performance, making it the most suitable choice for finding high-quality solutions to large-scale N-Queens problems in a reasonable amount of time.

==================================================

Full Source Code
==================================================

-- File: Exhaustive search (DFS).py
import time
import os
import psutil

def get_memory_usage():
process = psutil.Process(os.getpid())
return process.memory_info().rss / (1024 * 1024)

def solve_n_queens_dfs(n):
"""
Solves the N-Queens problem using exhaustive Depth-First Search with backtracking.

Returns:
    tuple: (solution_board, conflicts, time_taken, peak_memory)
"""
start_time = time.time()
board = [['.' for _ in range(n)] for _ in range(n)]
solution = []

def is_safe(row, col):
    for i in range(col):
        if board[row][i] == 'Q':
            return False
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 'Q':
            return False
    for i, j in zip(range(row, n, 1), range(col, -1, -1)):
        if board[i][j] == 'Q':
            return False
    return True

def solve(col):
    if col >= n:
        solution.append([row[:] for row in board])
        return True

    for i in range(n):
        if is_safe(i, col):
            board[i][col] = 'Q'
            if solve(col + 1):
                return True
            board[i][col] = '.'
    return False

solve(0)
end_time = time.time()
peak_memory = get_memory_usage()

if solution:
    return solution[0], 0, end_time - start_time, peak_memory
else:
    return None, -1, end_time - start_time, peak_memory

def print_solution(board):
if board:
for row in board:
print(" ".join(row))
else:
print("No solution found.")

if name == 'main':
N = 10
print(f"--- Solving N-Queens for N={N} with Exhaustive Search (DFS) ---")

solution_board, conflicts, time_taken, memory_used = solve_n_queens_dfs(N)

print("\nBoard:")
print_solution(solution_board)

if conflicts != -1:
    print(f"\nNumber of conflicts: {conflicts}")
    print("A solution was found!")
else:
    print("\nNo solution exists for this N.")

print(f"Time taken: {time_taken:.6f} seconds")
print(f"Memory used: {memory_used:.4f} MB")

-- File: Greedy Search (Hill Climbing).py
import random
import time
import os
import psutil

def get_memory_usage():
process = psutil.Process(os.getpid())
return process.memory_info().rss / (1024 * 1024)

def solve_n_queens_greedy(n, max_restarts=100):
"""
Solves the N-Queens problem using Greedy Hill Climbing with random restarts.

Returns:
    tuple: (best_state, best_conflicts, time_taken, peak_memory)
"""
start_time = time.time()
best_state = []
best_conflicts = n + 1

for restart in range(max_restarts):
    current_state = list(range(n))
    random.shuffle(current_state)
    current_conflicts = calculate_conflicts(current_state)

    if current_conflicts == 0:
        end_time = time.time()
        return current_state, 0, end_time - start_time, get_memory_usage()

    while True:
        better_move_found = False
        for row1 in range(n):
            for row2 in range(row1 + 1, n):
                next_state = list(current_state)
                next_state[row1], next_state[row2] = next_state[row2], next_state[row1]
                new_conflicts = calculate_conflicts(next_state)

                if new_conflicts < current_conflicts:
                    current_state = next_state
                    current_conflicts = new_conflicts
                    better_move_found = True

        if not better_move_found:
            break

    if current_conflicts < best_conflicts:
        best_conflicts = current_conflicts
        best_state = current_state

    if best_conflicts == 0:
        break

end_time = time.time()
peak_memory = get_memory_usage()
return best_state, best_conflicts, end_time - start_time, peak_memory

def calculate_conflicts(state):
conflicts = 0
n = len(state)
for i in range(n):
for j in range(i + 1, n):
if abs(i - j) == abs(state[i] - state[j]):
conflicts += 1
return conflicts

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

if name == 'main':
N = 200
print(f"--- Solving N-Queens for N={N} with Greedy Search ---")

final_state, final_conflicts, time_taken, memory_used = solve_n_queens_greedy(N, max_restarts=200)

print("\nFinal Board State:")
print_solution(final_state)

print(f"\nNumber of conflicts: {final_conflicts}")
if final_conflicts == 0:
    print("A solution was found!")
else:
    print("No perfect solution found. The above is the best state achieved.")

print(f"Time taken: {time_taken:.6f} seconds")
print(f"Memory used: {memory_used:.4f} MB")

-- File: Simulated Annealing.py
import random
import math
import time
import os
import psutil

def get_memory_usage():
process = psutil.Process(os.getpid())
return process.memory_info().rss / (1024 * 1024)

def solve_n_queens_annealing(n, initial_temp=100.0, cooling_rate=0.995, max_iterations=50000):
"""
Solves the N-Queens problem using Simulated Annealing.

Returns:
    tuple: (best_state, best_conflicts, time_taken, peak_memory)
"""
start_time = time.time()

current_state = list(range(n))
random.shuffle(current_state)
current_conflicts = calculate_conflicts(current_state)

best_state = list(current_state)
best_conflicts = current_conflicts
temp = initial_temp

for i in range(max_iterations):
    if current_conflicts == 0:
        break

    row1, row2 = random.sample(range(n), 2)
    next_state = list(current_state)
    next_state[row1], next_state[row2] = next_state[row2], next_state[row1]
    next_conflicts = calculate_conflicts(next_state)

    energy_delta = next_conflicts - current_conflicts

    if energy_delta < 0 or random.random() < math.exp(-energy_delta / temp):
        current_state = next_state
        current_conflicts = next_conflicts

    if current_conflicts < best_conflicts:
        best_state = current_state
        best_conflicts = current_conflicts

    temp *= cooling_rate

    if temp < 1e-5:
        break

end_time = time.time()
peak_memory = get_memory_usage()
return best_state, best_conflicts, end_time - start_time, peak_memory

def calculate_conflicts(state):
conflicts = 0
n = len(state)
for i in range(n):
for j in range(i + 1, n):
if abs(i - j) == abs(state[i] - state[j]):
conflicts += 1
return conflicts

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

if name == 'main':
N = 100
print(f"--- Solving N-Queens for N={N} with Simulated Annealing ---")

final_state, final_conflicts, time_taken, memory_used = solve_n_queens_annealing(N)

print("\nFinal Board State:")
print_solution(final_state)

print(f"\nNumber of conflicts: {final_conflicts}")
if final_conflicts == 0:
    print("A solution was found!")
else:
    print("No perfect solution found. The above is the best state achieved.")

print(f"Time taken: {time_taken:.6f} seconds")
print(f"Memory used: {memory_used:.4f} MB")

-- File: Genetic Algorithm.py
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

if name == 'main':
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
