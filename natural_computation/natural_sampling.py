import random

n_hits = 0. # Numbers of hits
it = 1000000 # Numnbers of iteration

for _ in range(it):
    # Take a random position
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    # Point in circle, formula: Racine((px - cx)**2 + (py - cy)**2) < rayon
    # px and py is are both 0
    # Inside the circle ? Racine(px**2 + py**2) <  rayon
    # rayon is 1 so
    # Inside the circle ? px**2 + py**2 < rayon
    if x**2 + y**2 < 1:
        n_hits += 1.

print (n_hits / it * 4)
