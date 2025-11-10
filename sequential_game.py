from time import perf_counter, sleep
import os
import numpy as np

def step(grid):
    rows, cols = grid.shape
    new_grid = np.zeros((rows, cols), dtype=int)

    for r in range(rows):
        for c in range(cols):
            # Count neighbours (8 possible directions)
            live_neighbors = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue  # omit cell itself
                    rr = (r + dr) % rows  # row index with modulo wrapping (torus topology)
                    cc = (c + dc) % cols  # cell index with modulo wrapping
                    live_neighbors += grid[rr, cc]

            if grid[r, c] == 1:
                # Living cell survives, if it has 2 or 3 neighbors
                if live_neighbors in (2, 3):
                    new_grid[r, c] = 1
            else:
                # Dead cell reborn, if it has exactly 3 neighbours
                if live_neighbors == 3:
                    new_grid[r, c] = 1

    return new_grid

def print_grid(grid):
    os.system('cls' if os.name == 'nt' else 'clear')
    for row in grid:
        print(''.join('█' if cell else ' ' for cell in row))
    print('-' * grid.shape[1])

def sequential_game(grid):

    num_of_iterations = 0
    total_time = 0 # Time only counts execution of: grid = step(grid)
    try:
        while True:
            print_grid(grid)
            st = perf_counter()
            grid = step(grid)
            end = perf_counter()
            total_time += (end - st)
            sleep(1)
            num_of_iterations += 1
    except KeyboardInterrupt:
        print("\nGame finished.")
        print(f"\nAverage execution time of the step: "
              f"{total_time/num_of_iterations:.6f} seconds")