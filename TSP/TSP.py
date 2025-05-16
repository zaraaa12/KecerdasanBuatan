import random
import numpy as np
import matplotlib.pyplot as plt

NUM_CITIES = 10
POP_SIZE = 100
GENERATIONS = 300
MUTATION_RATE = 0.01

cities = np.random.rand(NUM_CITIES, 2) * 100  # (x, y) coordinates

def distance(city1, city2):
    return np.linalg.norm(city1 - city2)

def total_distance(route):
    dist = 0
    for i in range(len(route)):
        dist += distance(cities[route[i]], cities[route[(i+1) % NUM_CITIES]])
    return dist

def fitness(route):
    return 1 / total_distance(route)

def initial_population():
    return [random.sample(range(NUM_CITIES), NUM_CITIES) for _ in range(POP_SIZE)]

def selection(population):
    tournament = random.sample(population, 5)
    tournament.sort(key=total_distance)
    return tournament[0]

def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(NUM_CITIES), 2))
    child = [-1] * NUM_CITIES
    child[start:end] = parent1[start:end]
    pointer = 0
    for city in parent2:
        if city not in child:
            while child[pointer] != -1:
                pointer += 1
            child[pointer] = city
    return child

def mutate(route):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(NUM_CITIES), 2)
        route[i], route[j] = route[j], route[i]
    return route

def genetic_algorithm():
    population = initial_population()
    best_route = min(population, key=total_distance)
    best_distances = []

    for gen in range(GENERATIONS):
        new_population = []

        for _ in range(POP_SIZE):
            parent1 = selection(population)
            parent2 = selection(population)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population
        current_best = min(population, key=total_distance)

        if total_distance(current_best) < total_distance(best_route):
            best_route = current_best

        best_distances.append(total_distance(best_route))
        if gen % 50 == 0:
            print(f"Gen {gen}: Best Distance = {total_distance(best_route):.2f}")

    return best_route, best_distances

def plot_route(route):
    x = [cities[i][0] for i in route + [route[0]]]
    y = [cities[i][1] for i in route + [route[0]]]
    plt.plot(x, y, 'bo-')
    for i, (xi, yi) in enumerate(cities):
        plt.text(xi+1, yi+1, str(i), fontsize=12)
    plt.title("Best Route Found")
    plt.show()

def plot_progress(best_distances):
    plt.plot(best_distances)
    plt.title("Best Distance Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.show()

best_route, best_distances = genetic_algorithm()
plot_route(best_route)
plot_progress(best_distances)
