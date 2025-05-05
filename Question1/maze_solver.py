import os
import heapq
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Create a sample maze.txt if it doesn't exist
maze_file = "maze.txt"
if not os.path.exists(maze_file):
    sample_maze = [
        "##########",
        "#A   #   #",
        "# #  # # #",
        "# #     B#",
        "##########"
    ]
    with open(maze_file, "w") as f:
        f.write("\n".join(sample_maze))
    print("maze.txt file created.")
else:
    print("maze.txt already exists.")

# Step 2: Node class
class Node:
    def __init__(self, state, parent=None, action=None, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost

    def __lt__(self, other):
        return False  # Required for heapq (but doesn't affect priority)

# Step 3: Maze class
class Maze:
    def __init__(self, filename):
        with open(filename) as f:
            self.grid = [list(line.strip()) for line in f.readlines()]
        self.height = len(self.grid)
        self.width = len(self.grid[0])
        self.start = self.goal = None
        self.walls = set()

        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == "A":
                    self.start = (y, x)
                elif cell == "B":
                    self.goal = (y, x)
                elif cell == "#":
                    self.walls.add((y, x))

        if self.start is None or self.goal is None:
            raise ValueError("Maze must have a start (A) and goal (B)")

    def in_bounds(self, pos):
        y, x = pos
        return 0 <= y < self.height and 0 <= x < self.width

    def passable(self, pos):
        return pos not in self.walls

    def neighbors(self, pos):
        y, x = pos
        candidates = [(y-1,x), (y+1,x), (y,x-1), (y,x+1)]
        return [p for p in candidates if self.in_bounds(p) and self.passable(p)]

    def manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def solve(self, algorithm="greedy"):
        frontier = []
        start_node = Node(self.start, cost=0)
        heapq.heappush(frontier, (self.manhattan(self.start, self.goal), start_node))

        explored = set()
        came_from = {}
        cost_so_far = {self.start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)

            if current.state == self.goal:
                return self.reconstruct_path(current), explored

            explored.add(current.state)

            for neighbor in self.neighbors(current.state):
                new_cost = current.cost + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = self.manhattan(neighbor, self.goal) if algorithm == "greedy" else new_cost + self.manhattan(neighbor, self.goal)
                    node = Node(state=neighbor, parent=current, cost=new_cost)
                    heapq.heappush(frontier, (priority, node))

        raise Exception("No path found")

    def reconstruct_path(self, node):
        path = []
        while node.parent:
            path.append(node.state)
            node = node.parent
        path.reverse()
        return path

    def visualize(self, path, explored, filename="maze_solution.png"):
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for y in range(self.height):
            for x in range(self.width):
                if (y, x) in self.walls:
                    image[y, x] = [0, 0, 0]  # black
                elif (y, x) == self.start:
                    image[y, x] = [0, 255, 0]  # green
                elif (y, x) == self.goal:
                    image[y, x] = [255, 0, 0]  # red
                elif (y, x) in path:
                    image[y, x] = [0, 0, 255]  # blue
                elif (y, x) in explored:
                    image[y, x] = [200, 200, 200]  # gray
                else:
                    image[y, x] = [255, 255, 255]  # white

        plt.figure(figsize=(6,6))
        plt.imshow(image)
        plt.axis('off')
        plt.title("Maze Solution Path")
        plt.savefig(filename)
        plt.show()

# Step 4: Load maze and solve
maze = Maze("maze.txt")

# Choose algorithm: "greedy" or "astar"
algorithm = "astar"
path, explored = maze.solve(algorithm=algorithm)
print(f"{algorithm.upper()} found a path with {len(path)} steps.")
maze.visualize(path, explored)
