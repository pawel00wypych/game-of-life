import numpy as np
import time
import os

def step(grid):
    rows, cols = grid.shape
    new_grid = np.zeros((rows, cols), dtype=int)

    for r in range(rows):
        for c in range(cols):
            # Liczymy sąsiadów (8 możliwych kierunków)
            live_neighbors = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue  # pomijamy samą komórkę
                    rr = (r + dr) % rows  # indeks wiersza (z owijaniem)
                    cc = (c + dc) % cols  # indeks kolumny (z owijaniem)
                    live_neighbors += grid[rr, cc]

            if grid[r, c] == 1:
                # Żywa komórka przeżywa, jeśli ma 2 lub 3 sąsiadów
                if live_neighbors in (2, 3):
                    new_grid[r, c] = 1
            else:
                # Martwa komórka ożywa, jeśli ma dokładnie 3 sąsiadów
                if live_neighbors == 3:
                    new_grid[r, c] = 1

    return new_grid

def print_grid(grid):
    os.system('cls' if os.name == 'nt' else 'clear')
    for row in grid:
        print(''.join('█' if cell else ' ' for cell in row))
    print('-' * grid.shape[1])

def game():
    rows, cols = 20, 40
    p = 0.2  # prawdopodobieństwo żywej komórki na starcie

    grid = (np.random.rand(rows, cols) < p).astype(int)

    try:
        while True:
            print_grid(grid)
            grid = step(grid)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKoniec gry.")

if __name__ == "__main__":
    game()
