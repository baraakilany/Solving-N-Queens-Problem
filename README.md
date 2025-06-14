N-Queens Problem Solvers
This repository contains Python implementations of four different algorithms to solve the classic N-Queens puzzle. The N-Queens puzzle is the problem of placing N chess queens on an N×N chessboard so that no two queens threaten each other.

Algorithms Implemented
This project explores and compares four distinct approaches to solving the N-Queens problem:

Exhaustive Search (DFS): This uses a classic Depth-First Search (DFS) with backtracking to find an exact solution. It systematically explores every possible configuration until a valid solution is found. While it guarantees a solution for smaller N, its runtime becomes impractical for larger board sizes.

Greedy Search (Hill Climbing): This local search algorithm starts with a random placement of queens and iteratively makes the best possible move to reduce the number of conflicts. To avoid getting stuck in local optima, this implementation uses random restarts.

Simulated Annealing: This is a probabilistic local search technique that is also used to find the global optimum. It is similar to hill climbing, but to escape local optima, it may accept moves that increase the number of conflicts, based on a "temperature" parameter that decreases over time.

Genetic Algorithm: Inspired by natural selection, this algorithm evolves a "population" of board configurations over "generations". Solutions are selected based on their "fitness" (fewer conflicts), and new solutions are created using crossover and mutation operators.

File Descriptions
Exhaustive search (DFS).py: A solver using a backtracking Depth-First Search algorithm.
Greedy Search (Hill Climbing).py: A solver using a Hill Climbing algorithm with random restarts.
Simulated Annealing.py: A solver using a Simulated Annealing algorithm.
Genetic Algorithm.py: A solver using a Genetic Algorithm.
How to Run
To run any of the solvers, you need Python and the psutil library installed.

Install dependencies:

Bash

pip install psutil
Run a script:
You can run any of the algorithm scripts directly from your terminal.

For example, to run the Genetic Algorithm solver:

Bash

python "Genetic Algorithm.py"
Changing N:
To change the board size (the value of N), you can edit the script file and modify the N variable at the bottom of the file. For example, in Greedy Search (Hill Climbing).py:

Python

if __name__ == '__main__':
    N = 200 # Change this value for a different board size
Output
Each script will print the following to the console:

The final board state, showing the positions of the queens.
The number of remaining conflicts in the final state.
Whether a perfect solution was found.
The total time taken for the execution in seconds.
The peak memory used by the process in megabytes.
