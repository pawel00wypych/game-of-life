from mpi4py import MPI
import numpy as np
from time import perf_counter, sleep
import os
import sys


def step(grid_old, grid_new):
    rows, cols = grid_old.shape

    for r in range(1, rows-1):  # skip ghost rows
        for c in range(1, cols-1):  # skip ghost columns if any
            live_neighbors = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr = r + dr
                    cc = c + dc
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

def mpi_game(steps=100, rows=20, cols=40):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rows = rows
    cols = cols

    num_of_iterations = 0
    total_time = 0 # Time only counts execution of: grid = step(grid)
    steps = steps

    # Split rows among processes
    rows_per_proc = rows // size
    extra = rows % size

    # Determine local grid size for each rank (+2 for ghost rows)
    local_rows = rows_per_proc + (1 if rank < extra else 0)
    local_rows += 2  # 1 ghost row top, 1 ghost row bottom

    local_grid_old = np.zeros((local_rows, cols), dtype=np.int32)
    local_grid_new = np.zeros_like(local_grid_old)

    # Initialize only the "real" rows
    np.random.seed(rank)
    local_grid_old[1:-1, :] = (
                np.random.rand(local_rows - 2, cols) < 0.2).astype(np.int32)

    top_neighbor = rank - 1 if rank > 0 else MPI.PROC_NULL
    bottom_neighbor = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    try:
        while num_of_iterations <  steps:
            #print_grid(local_grid_old)
            st = perf_counter()
            # Exchange ghost rows
            comm.Sendrecv(local_grid_old[1, :], dest=top_neighbor,
                          recvbuf=local_grid_old[-1, :],
                          source=bottom_neighbor)
            comm.Sendrecv(local_grid_old[-2, :], dest=bottom_neighbor,
                          recvbuf=local_grid_old[0, :], source=top_neighbor)

            # Update local grid
            step(local_grid_old, local_grid_new)

            # Swap buffers
            local_grid_old, local_grid_new = local_grid_new, local_grid_old

            end = perf_counter()
            iteration_time = end - st
            max_time = comm.reduce(iteration_time, op=MPI.MAX, root=0)
            if rank == 0:
                total_time += max_time

            #sleep(1)
            num_of_iterations += 1


        # Step 1: compute counts in elements
        counts = comm.gather((local_rows - 2) * cols, root=0)

        if rank == 0:
            full_grid = np.zeros((rows * cols,), dtype=np.int32)  # flat array
            # compute displacements
            displs = [0] + np.cumsum(counts[:-1]).tolist()
        else:
            full_grid = None
            displs = None

        # Step 2: flatten local array for Gatherv
        local_flat = local_grid_old[1:-1, :].flatten()

        comm.Gatherv(sendbuf=local_flat,
                     recvbuf=(full_grid, counts, displs, MPI.INT),
                     root=0)

        # Step 3: reshape on root
        if rank == 0:
            full_grid = full_grid.reshape((rows, cols))

        if rank == 0:
            print(f"\nAverage execution time of the step: "
              f"{total_time / num_of_iterations:.8f} seconds")
            print(f"Total time for {steps} steps: {total_time:.8f} seconds")
    except KeyboardInterrupt:
        print("\nmpi_game finished by KeyboardInterrupt.")
        print(f"\nAverage execution time of the step: "
              f"{total_time/num_of_iterations:.8f} seconds")



def main():
    if len(sys.argv) != 4:
        print("Usage: python mpi_game.py <rows> <cols> <steps>")
        print("Launching: python mpi_game.py with defaults")
        mpi_game()
        exit(0)

    rows = int(sys.argv[1])
    cols = int(sys.argv[2])
    steps = int(sys.argv[3])
    mpi_game(steps=steps, rows=rows, cols=cols)

if __name__ == "__main__":
    main()