# cuda_game.py
# Conway's Game of Life — GPU step with bit-packed state + shared memory (Numba CUDA)
# Fallback-friendly: on machines without CUDA (e.g., MacBook Apple Silicon), the module
# runs and prints a clear message instead of crashing.

import sys
import numpy as np

# === Optional CUDA import ===
CUDA_AVAILABLE = False
try:
    from numba import cuda, int32, uint32
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False


# =========================
# Bit-pack helpers (CPU)
# =========================

def words_per_row(n: int) -> int:
    """Number of 32-bit words needed to store one row of n bits."""
    return (n + 31) // 32


def pack_bits_u32(grid_u8: np.ndarray) -> np.ndarray:
    """
    Pack a 2D uint8 grid of shape (n, n) with values {0,1} into uint32 bitfield
    of shape (n * words_per_row,).
    """
    n = grid_u8.shape[0]
    wpr = words_per_row(n)
    out = np.zeros(n * wpr, dtype=np.uint32)
    # Vectorized-ish pack row by row for clarity
    for y in range(n):
        row = grid_u8[y]
        for x in range(n):
            if row[x] != 0:
                wi = y * wpr + (x >> 5)
                out[wi] |= (1 << (x & 31))
    return out


def unpack_bits_u32(bitfield: np.ndarray, n: int) -> np.ndarray:
    """Unpack uint32 bitfield back to uint8 grid {0,1} of shape (n, n)."""
    wpr = words_per_row(n)
    out = np.zeros((n, n), dtype=np.uint8)
    for y in range(n):
        for x in range(n):
            wi = y * wpr + (x >> 5)
            out[y, x] = (bitfield[wi] >> (x & 31)) & 1
    return out


# =========================
# CUDA kernels (Numba)
# =========================
if CUDA_AVAILABLE:
    # Tile size (work region) — 16x16 threads; shared tile is (16+2)x(16+2) for halo
    TILE = 16

    @cuda.jit(device=True, inline=True)
    def _wrap(n, v):
        # Toroidal wrap (periodic)
        if v < 0:
            v += n
            if v < 0:
                v %= n
        elif v >= n:
            v -= n
            if v >= n:
                v %= n
        return v

    @cuda.jit(device=True, inline=True)
    def _get_bit(d_in, n, wpr, y, x):
        x = _wrap(n, x)
        y = _wrap(n, y)
        wi = y * wpr + (x >> 5)
        return (d_in[wi] >> (x & 31)) & 1

    @cuda.jit
    def kernel_clear(d_out):
        i = cuda.grid(1)
        if i < d_out.size:
            d_out[i] = 0

    @cuda.jit
    def kernel_life_bitpacked(d_in, d_out, n, wpr):
        """
        Compute next generation.
        Global memory stores state as uint32 bitfield (1 bit per cell).
        Shared memory tile caches current generation in bytes (0/1) for locality.
        We zero d_out beforehand and only atomically OR bits of live cells.
        """
        ty = cuda.threadIdx.y
        tx = cuda.threadIdx.x
        by = cuda.blockIdx.y
        bx = cuda.blockIdx.x
        bdy = cuda.blockDim.y
        bdx = cuda.blockDim.x

        # Global coordinates of the thread's cell
        x = bx * bdx + tx
        y = by * bdy + ty

        # Shared tile with 1-cell halo on each side
        # Static shape for Numba: (TILE+2, TILE+2)
        sm = cuda.shared.array(shape=(TILE + 2, TILE + 2), dtype=uint32)

        # Load center cell into shared memory
        if x < n and y < n:
            sm[ty + 1, tx + 1] = _get_bit(d_in, n, wpr, y, x)
        else:
            sm[ty + 1, tx + 1] = 0

        # Halo loads (only threads at the edges do extra work)
        if tx == 0:
            xx = x - 1
            yy = y
            val = 0
            if y < n:
                val = _get_bit(d_in, n, wpr, yy, xx)
            sm[ty + 1, 0] = val

        if tx == bdx - 1:
            xx = x + 1
            yy = y
            val = 0
            if y < n:
                val = _get_bit(d_in, n, wpr, yy, xx)
            sm[ty + 1, TILE + 1] = val

        if ty == 0:
            xx = x
            yy = y - 1
            val = 0
            if x < n:
                val = _get_bit(d_in, n, wpr, yy, xx)
            sm[0, tx + 1] = val

        if ty == bdy - 1:
            xx = x
            yy = y + 1
            val = 0
            if x < n:
                val = _get_bit(d_in, n, wpr, yy, xx)
            sm[TILE + 1, tx + 1] = val

        # Corners of halo
        if tx == 0 and ty == 0:
            sm[0, 0] = _get_bit(d_in, n, wpr, y - 1, x - 1) if (x < n and y < n) else 0
        if tx == bdx - 1 and ty == 0:
            sm[0, TILE + 1] = _get_bit(d_in, n, wpr, y - 1, x + 1) if (x < n and y < n) else 0
        if tx == 0 and ty == bdy - 1:
            sm[TILE + 1, 0] = _get_bit(d_in, n, wpr, y + 1, x - 1) if (x < n and y < n) else 0
        if tx == bdx - 1 and ty == bdy - 1:
            sm[TILE + 1, TILE + 1] = _get_bit(d_in, n, wpr, y + 1, x + 1) if (x < n and y < n) else 0

        cuda.syncthreads()

        if x >= n or y >= n:
            return

        # Neighbor sum from shared memory (8 neighbors)
        s = (sm[ty, tx] + sm[ty, tx + 1] + sm[ty, tx + 2] +
             sm[ty + 1, tx] + sm[ty + 1, tx + 2] +
             sm[ty + 2, tx] + sm[ty + 2, tx + 1] + sm[ty + 2, tx + 2])

        alive = sm[ty + 1, tx + 1]
        new_alive = 1 if ((alive == 1 and (s == 2 or s == 3)) or (alive == 0 and s == 3)) else 0

        if new_alive:
            wi = y * wpr + (x >> 5)
            mask = uint32(1 << (x & 31))
            # We zeroed d_out beforehand; set bit atomically to avoid word collisions
            cuda.atomic.or_(d_out, wi, mask)


def run_cuda(n: int, gens: int, periodic: int = 1, out_path: str | None = None,
             backend: str = "auto", device_index: int = 0):
    """
    Run Game of Life on GPU (Numba CUDA). Uses:
      - bit-packed uint32 state in global memory,
      - shared-memory tiles for neighbor access,
      - toroidal borders (periodic=1).
    On non-CUDA machines prints a friendly message and returns.
    """
    if not CUDA_AVAILABLE or backend.lower() == "none":
        print("[CUDA] Brak dostępnej platformy CUDA na tej maszynie (np. MacBook ARM). "
              "Kod jest dostarczony, ale uruchomienie wymaga karty NVIDIA i sterowników.")
        return

    # Select device (if multiple)
    try:
        cuda.select_device(device_index)
    except Exception as e:
        print(f"[CUDA] Nie można wybrać urządzenia {device_index}: {e}")
        return

    # Init grid (random 0/1) — użyj tego samego rozkładu co w wersjach CPU
    rng = np.random.default_rng(1234)
    host_grid = (rng.random((n, n)) < 0.2).astype(np.uint8)

    # Pack to bitfield
    wpr = words_per_row(n)
    a_bits = pack_bits_u32(host_grid)
    b_bits = np.zeros_like(a_bits, dtype=np.uint32)

    # Move to device
    d_a = cuda.to_device(a_bits)
    d_b = cuda.to_device(b_bits)

    # Configure grid
    TILE = 16
    threads = (TILE, TILE)
    blocks = ((n + TILE - 1) // TILE, (n + TILE - 1) // TILE)

    # Main loop
    for _ in range(gens):
        # Clear output buffer
        tpb_clear = 256
        blocks_clear = (d_b.size + tpb_clear - 1) // tpb_clear
        kernel_clear[blocks_clear, tpb_clear](d_b)

        # Step
        kernel_life_bitpacked[blocks, threads](d_a, d_b, n, wpr)
        cuda.synchronize()

        # Swap
        d_a, d_b = d_b, d_a

    # Copy back and (optionally) save final frame
    res_bits = d_a.copy_to_host()
    final_grid = unpack_bits_u32(res_bits, n)

    if out_path:
        # Save as simple PGM (white=alive)
        try:
            with open(out_path, "wb") as f:
                f.write(f"P5\n{n} {n}\n255\n".encode("ascii"))
                # Map 0->0, 1->255
                img = (final_grid * 255).astype(np.uint8).tobytes()
                f.write(img)
            print(f"[CUDA] Zapisano wynik do: {out_path}")
        except Exception as e:
            print(f"[CUDA] Nie udało się zapisać {out_path}: {e}")


# =========================
# CLI
# =========================
def _parse_argv(argv):
    n = 1024
    gens = 200
    periodic = 1
    out = None
    backend = "auto"
    dev = 0
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "-n" and i + 1 < len(argv):
            n = int(argv[i + 1]); i += 2
        elif a == "-g" and i + 1 < len(argv):
            gens = int(argv[i + 1]); i += 2
        elif a == "-p" and i + 1 < len(argv):
            periodic = int(argv[i + 1]); i += 2
        elif a == "-o" and i + 1 < len(argv):
            out = argv[i + 1]; i += 2
        elif a == "-b" and i + 1 < len(argv):
            backend = argv[i + 1]; i += 2
        elif a == "-d" and i + 1 < len(argv):
            dev = int(argv[i + 1]); i += 2
        else:
            i += 1
    return n, gens, periodic, out, backend, dev


if __name__ == "__main__":
    n, gens, periodic, out, backend, dev = _parse_argv(sys.argv)
    run_cuda(n=n, gens=gens, periodic=periodic, out_path=out, backend=backend, device_index=dev)
