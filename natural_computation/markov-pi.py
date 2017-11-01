import random


if __name__ == '__main__':
    x, y, n_hits = 1, 1, 0
    δ = 0.5

    for it in range(1000000):
        # Take a random vector
        Δx = random.uniform(-δ, δ)
        Δy = random.uniform(-δ, δ)
        # If we stay inside de square
        if abs(x + Δx) < 1 and abs(y + Δy) < 1:
            x = x + Δx
            y = y + Δy
        # Hit inside the circle ?
        if x**2 + y**2 < 1:
            n_hits += 1.

    print (n_hits / it * 4)
