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


if __name__ == '__main__':
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