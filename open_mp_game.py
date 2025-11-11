from time import perf_counter, sleep
import os
import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads

@njit(parallel=True)
def step(grid_old, grid_new):
    rows, cols = grid_old.shape

    # parallel execution
    for r in prange(rows):
        for c in range(cols):
            # Count neighbours (8 possible directions)
            live_neighbors = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue  # omit cell itself
                    rr = (r + dr) % rows  # row index with modulo wrapping (torus topology)
                    cc = (c + dc) % cols  # cell index with modulo wrapping
                    live_neighbors += grid_old[rr, cc]

            if grid_old[r, c] == 1:
                grid_new[r, c] = 1 if live_neighbors in (2, 3) else 0
            else:
                grid_new[r, c] = 1 if live_neighbors == 3 else 0
    return grid_new

def print_grid(grid):
    os.system('cls' if os.name == 'nt' else 'clear')
    for row in grid:
        print(''.join('█' if cell else ' ' for cell in row))
    print('-' * grid.shape[1])

def open_mp_game(grid_old, grid_new, steps = 100):

    num_of_iterations = 0
    steps = steps
    total_time = 0
    set_num_threads(1)
    try:
        while num_of_iterations < steps:
            #print_grid(grid_old)
            st = perf_counter()
            grid_new = step(grid_old, grid_new)
            grid_old, grid_new = grid_new, grid_old
            end = perf_counter()
            total_time += (end - st)
            #sleep(1)
            num_of_iterations += 1

        total_time_1 = total_time

        print(f"\nAverage execution time of the step: "
              f"{total_time_1 / num_of_iterations:.8f} seconds for "
              f"1 thread\nTotal execution time for {steps} steps:"
              f"{total_time_1:.8f} seconds")

        num_of_iterations = 0
        total_time = 0
        set_num_threads(4)
        while num_of_iterations < steps:
            #print_grid(grid_old)
            st = perf_counter()
            grid_new = step(grid_old, grid_new)
            grid_old, grid_new = grid_new, grid_old
            end = perf_counter()
            total_time += (end - st)
            #sleep(1)
            num_of_iterations += 1

        total_time_4 = total_time
        print(f"\nAverage execution time of the step: "
              f"{total_time_4 / num_of_iterations:.8f} seconds for "
              f"4 threads\nTotal execution time for {steps} steps:"
              f"{total_time_4:.8f} seconds")

        print("Speedup  :", total_time_1 / total_time_4)

    except KeyboardInterrupt:
        print("\nOpen_mp_game finished by KeyboardInterrupt.")
        print(f"\nAverage execution time of the step: "
              f"{total_time/num_of_iterations:.8f} seconds for "
              f"{get_num_threads()} threads")