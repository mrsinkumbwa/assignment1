import math
import random
import matplotlib.pyplot as plt

# Define towns and distance matrix
towns = [
    "Windhoek", "Swakopmund", "Walvis Bay", "Otjiwarongo", "Tsumeb",
    "Groottontein", "Mariental", "Keetmanshoop", "Ondangwa", "Oshakati"
]

distance_matrix = [
    [0, 361, 395, 249, 433, 459, 268, 497, 678, 712],
    [361, 0, 35.5, 379, 562, 589, 541, 859, 808, 779],
    [395, 35.5, 0, 413, 597, 623, 511, 732, 884, 855],
    [249, 379, 413, 0, 260, 183, 519, 768, 514, 485],
    [433, 562, 597, 260, 0, 60, 682, 921, 254, 288],
    [459, 589, 623, 183, 60, 0, 708, 947, 308, 342],
    [268, 541, 511, 519, 682, 708, 0, 231, 909, 981],
    [497, 859, 732, 768, 921, 947, 231, 0, 1175, 1210],
    [678, 808, 884, 514, 254, 308, 909, 1175, 0, 30],
    [712, 779, 855, 485, 288, 342, 981, 1210, 30, 0],
]

class TSP:
    def __init__(self, towns, distance_matrix):
        self.towns = towns
        self.distance_matrix = distance_matrix
        self.town_to_index = {town: idx for idx, town in enumerate(towns)}
    
    def calculate_total_distance(self, route):
        total = 0.0
        for i in range(len(route)):
            current = route[i]
            if i < len(route) - 1:
                next_town = route[i+1]
            else:
                next_town = route[0]  # Return to Windhoek
            idx_current = self.town_to_index[current]
            idx_next = self.town_to_index[next_town]
            total += self.distance_matrix[idx_current][idx_next]
        return total

class SimulatedAnnealingSolver:
    def __init__(self, tsp, initial_temp=10000, cooling_rate=0.995, num_iterations=10000):
        self.tsp = tsp
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.num_iterations = num_iterations
        self.best_route = None
        self.best_distance = float('inf')
        self.initial_route = None
        self.history = []
    
    def generate_initial_route(self):
        other_towns = self.tsp.towns[1:]
        shuffled = other_towns.copy()
        random.shuffle(shuffled)
        return [self.tsp.towns[0]] + shuffled
    
    def get_neighbor(self, route):
        new_route = route.copy()
        i, j = random.sample(range(1, len(route)), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]
        return new_route
    
    def solve(self):
        current_route = self.generate_initial_route()
        self.initial_route = current_route.copy()
        current_distance = self.tsp.calculate_total_distance(current_route)
        self.best_route = current_route.copy()
        self.best_distance = current_distance
        self.history.append(current_distance)
        
        T = self.initial_temp
        
        for _ in range(self.num_iterations):
            new_route = self.get_neighbor(current_route)
            new_distance = self.tsp.calculate_total_distance(new_route)
            
            if new_distance < current_distance:
                current_route = new_route
                current_distance = new_distance
                if new_distance < self.best_distance:
                    self.best_route = new_route.copy()
                    self.best_distance = new_distance
            else:
                delta = new_distance - current_distance
                acceptance_prob = math.exp(-delta / T)
                if random.random() < acceptance_prob:
                    current_route = new_route
                    current_distance = new_distance
            
            T *= self.cooling_rate
            self.history.append(current_distance)
        
        return self.best_route, self.best_distance

def plot_route(tsp, route, title):
    plt.figure(figsize=(12, 4))
    x = list(range(len(route) + 1))  # +1 to include return to start
    y_indices = [tsp.town_to_index[town] for town in route] + [tsp.town_to_index[route[0]]]
    plt.plot(x, y_indices, marker='o', linestyle='-', markersize=8)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Town Index")
    plt.xticks(x)
    plt.yticks(range(len(tsp.towns)), tsp.towns)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_convergence(history):
    plt.figure(figsize=(12, 4))
    plt.plot(history, color='blue', linewidth=1)
    plt.title("Distance Convergence Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Total Distance (km)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main execution
tsp = TSP(towns, distance_matrix)
solver = SimulatedAnnealingSolver(tsp, initial_temp=10000, cooling_rate=0.995, num_iterations=10000)
best_route, best_distance = solver.solve()

# Output results
print(f"Initial Route: {solver.initial_route}")
print(f"Initial Distance: {tsp.calculate_total_distance(solver.initial_route):.2f} km\n")
print(f"Optimized Route: {best_route}")
print(f"Optimized Distance: {best_distance:.2f} km")

# Plotting
plot_route(tsp, solver.initial_route, "Initial Route")
plot_route(tsp, best_route, "Optimized Route")
plot_convergence(solver.history)