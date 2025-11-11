import numpy as np
import subprocess
from sequential_game import sequential_game
from open_mp_game import open_mp_game
from mpi_game import mpi_game
from cuda_game import cuda_game

def game():
    rows, cols = 500, 500
    p = 0.2  # probability of living cell at the start
    steps = 100

    print("Choose game type:\n[1] - sequential\n[2] - openMP\n[3] - MPI "
          "game\n[4] - CUDA game")
    try:
        game_num = int(input("Provide game number:"))
        grid_old = (np.random.rand(rows, cols) < p).astype(np.int32)
        grid_new = np.zeros_like(grid_old)

        match game_num:
            case 1:
                sequential_game(grid_old, grid_new, steps)
            case 2:
                open_mp_game(grid_old, grid_new, steps)
            case 3:
                subprocess.run(["mpiexec", "-n", "4", "python", "mpi_game.py",
                                str(rows), str(cols), str(steps)])
            case 4:
                cuda_game()
            case _:
                print(f"{game_num} is not a valid number, choose between ["
                      f"1-4]")
    except Exception as e:
        print(f"Choosing number failed. Reason: {e}")


if __name__ == "__main__":
    game()
