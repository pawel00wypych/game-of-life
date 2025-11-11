# cuda_game.py
# Conway's Game of Life — CUDA (Numba) with shared-memory tiling and toroidal borders.
# Compatible with main.py: from cuda_game import cuda_game  -> cuda_game()

import numpy as np
from time import perf_counter

# Check CUDA availability
CUDA_OK = False
try:
    from numba import cuda
    CUDA_OK = cuda.is_available()
except Exception:
    CUDA_OK = False


# ====== GPU kernel (uint8 grid, shared memory tile) ======
if CUDA_OK:
    TILE = 16  # working tile 16x16; shared memory has halo (+2)
    @cuda.jit
    def life_step_shared(grid_old, grid_new, rows, cols):
        ty = cuda.threadIdx.y
        tx = cuda.threadIdx.x
        by = cuda.blockIdx.y
        bx = cuda.blockIdx.x
        bdy = cuda.blockDim.y
        bdx = cuda.blockDim.x

        r = by * bdy + ty
        c = bx * bdx + tx

        # Shared memory tile: (TILE+2)x(TILE+2) with halo
        sm = cuda.shared.array(shape=(TILE + 2, TILE + 2), dtype=np.uint8)

        # Toroidal wrapping
        def wrap(x, n):
            if x < 0:
                x += n
                if x < 0:
                    x %= n
            elif x >= n:
                x -= n
                if x >= n:
                    x %= n
            return x

        # Load the center cell
        if r < rows and c < cols:
            sm[ty + 1, tx + 1] = grid_old[r, c]
        else:
            sm[ty + 1, tx + 1] = 0

        # Load left/right halo
        if tx == 0:
            sm[ty + 1, 0] = grid_old[wrap(r, rows), wrap(c - 1, cols)] if r < rows else 0
        if tx == bdx - 1:
            sm[ty + 1, TILE + 1] = grid_old[wrap(r, rows), wrap(c + 1, cols)] if r < rows else 0

        # Load top/bottom halo
        if ty == 0:
            sm[0, tx + 1] = grid_old[wrap(r - 1, rows), wrap(c, cols)] if c < cols else 0
        if ty == bdy - 1:
            sm[TILE + 1, tx + 1] = grid_old[wrap(r + 1, rows), wrap(c, cols)] if c < cols else 0

        # Load corners of halo
        if tx == 0 and ty == 0:
            sm[0, 0] = grid_old[wrap(r - 1, rows), wrap(c - 1, cols)]
        if tx == bdx - 1 and ty == 0:
            sm[0, TILE + 1] = grid_old[wrap(r - 1, rows), wrap(c + 1, cols)]
        if tx == 0 and ty == bdy - 1:
            sm[TILE + 1, 0] = grid_old[wrap(r + 1, rows), wrap(c - 1, cols)]
        if tx == bdx - 1 and ty == bdy - 1:
            sm[TILE + 1, TILE + 1] = grid_old[wrap(r + 1, rows), wrap(c + 1, cols)]

        cuda.syncthreads()

        if r >= rows or c >= cols:
            return

        # Sum of 8 neighboring cells from shared memory
        s = (sm[ty, tx] + sm[ty, tx + 1] + sm[ty, tx + 2] +
             sm[ty + 1, tx] +                     sm[ty + 1, tx + 2] +
             sm[ty + 2, tx] + sm[ty + 2, tx + 1] + sm[ty + 2, tx + 2])

        alive = sm[ty + 1, tx + 1]
        new_alive = 1 if ((alive == 1 and (s == 2 or s == 3)) or (alive == 0 and s == 3)) else 0
        grid_new[r, c] = new_alive


def cuda_game():
    """
    Entry point called by main.py (option 4).
    - If CUDA is unavailable: prints a clear message and exits.
    - If CUDA is available: runs the GPU kernel, measures times, and prints statistics.
    """
    rows, cols = 500, 500
    steps = 100
    p = 0.2

    if not CUDA_OK:
        print("[CUDA] CUDA is not available on this machine (e.g., MacBook with Apple Silicon).")
        print("[CUDA] The code will run correctly on a computer with an NVIDIA GPU and CUDA drivers.")
        return

    # Initialize the board: random 0/1 with probability p
    grid_old_h = (np.random.rand(rows, cols) < p).astype(np.uint8)
    grid_new_h = np.zeros_like(grid_old_h)

    # Allocate GPU memory
    d_old = cuda.to_device(grid_old_h)
    d_new = cuda.to_device(grid_new_h)

    # Configure grid/block dimensions
    TILE = 16
    threads = (TILE, TILE)
    blocks = ((cols + TILE - 1) // TILE, (rows + TILE - 1) // TILE)

    num_iter = 0
    total_time = 0.0

    try:
        while num_iter < steps:
            start = perf_counter()
            life_step_shared[blocks, threads](d_old, d_new, rows, cols)
            cuda.synchronize()
            # Swap buffers
            d_old, d_new = d_new, d_old
            end = perf_counter()
            total_time += (end - start)
            num_iter += 1

        avg = total_time / num_iter if num_iter > 0 else 0.0
        print(f"\nAverage execution time per step: {avg:.8f} seconds (CUDA)")
        print(f"Total time for {steps} steps: {total_time:.8f} seconds (CUDA)")

    except KeyboardInterrupt:
        if num_iter > 0:
            print("\nCUDA Game interrupted by user.")
            print(f"\nAverage execution time per step: {total_time/num_iter:.8f} seconds (CUDA)")
