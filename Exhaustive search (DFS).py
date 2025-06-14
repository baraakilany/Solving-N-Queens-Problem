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


if __name__ == '__main__':
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