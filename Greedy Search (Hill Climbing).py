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


if __name__ == '__main__':
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