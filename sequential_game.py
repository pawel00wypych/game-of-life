from time import perf_counter, sleep
import os

def step(grid_old, grid_new):
    rows, cols = grid_old.shape

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
                    live_neighbors += grid_old[rr, cc]

            if grid_old[r, c] == 1:
                # Living cell survives, if it has 2 or 3 neighbors
                if live_neighbors in (2, 3):
                    grid_new[r, c] = 1
            else:
                # Dead cell reborn, if it has exactly 3 neighbours
                if live_neighbors == 3:
                    grid_new[r, c] = 1

    return grid_new

def print_grid(grid):
    os.system('cls' if os.name == 'nt' else 'clear')
    for row in grid:
        print(''.join('█' if cell else ' ' for cell in row))
    print('-' * grid.shape[1])

def sequential_game(grid_old, grid_new, steps=100):

    num_of_iterations = 0
    total_time = 0 # Time only counts execution of: grid = step(grid)
    steps = steps

    try:
        while num_of_iterations <  steps:
            print_grid(grid_old)
            st = perf_counter()
            grid_new = step(grid_old, grid_new)
            grid_old, grid_new = grid_new, grid_old
            end = perf_counter()
            total_time += (end - st)
            sleep(1)
            num_of_iterations += 1
        print(f"\nAverage execution time of the step: "
              f"{total_time / num_of_iterations:.8f} seconds")
        print(f"Total time for {steps} steps: {total_time:.8f} seconds")
    except KeyboardInterrupt:
        print("\nSequential_game finished by KeyboardInterrupt.")
        print(f"\nAverage execution time of the step: "
              f"{total_time/num_of_iterations:.8f} seconds")
        print(f"Total time for {steps} steps: {total_time:.8f} seconds")