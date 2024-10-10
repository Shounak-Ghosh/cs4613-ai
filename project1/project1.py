import numpy as np
import heapq
from math import sqrt, atan2, pi

class Node:
    def __init__(self, position, g=0, h=0, parent=None, action=None):
        self.position = position
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent
        self.action = action

    def __lt__(self, other):
        return self.f < other.f

def read_input(filename):
    with open(filename, 'r') as file:
        start_goal = list(map(int, file.readline().split()))
        grid = np.array([list(map(int, file.readline().split())) for _ in range(30)])
    return (start_goal[0], start_goal[1]), (start_goal[2], start_goal[3]), np.flipud(grid)

def write_output(filename, path, num_nodes, grid):
    with open(filename, 'w') as file:
        file.write(f"{len(path) - 1}\n")
        file.write(f"{num_nodes}\n")
        file.write(' '.join(str(node.action) for node in path[1:]) + '\n')
        file.write(' '.join(f"{node.f:.2f}" for node in path) + '\n')
        
        for row in np.flipud(grid):
            file.write(' '.join(map(str, row)) + '\n')

def heuristic(a, b):
    return sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

def get_neighbors(grid, node):
    directions = [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]
    neighbors = []
    for i, (dx, dy) in enumerate(directions):
        new_pos = (node.position[0] + dx, node.position[1] + dy)
        if 0 <= new_pos[0] < grid.shape[0] and 0 <= new_pos[1] < grid.shape[1] and grid[new_pos] != 1:
            neighbors.append((new_pos, i))
    return neighbors

def calculate_angle_cost(prev_action, new_action, k):
    if prev_action is None:
        return 0
    angle_diff = abs(new_action - prev_action)
    if angle_diff > 4:
        angle_diff = 8 - angle_diff
    return k * (angle_diff * 45) / 180

def calculate_distance_cost(action):
    return sqrt(2) if action % 2 == 1 else 1

def a_star(grid, start, goal, k):
    start_node = Node(start, h=heuristic(start, goal))
    open_set = [start_node]
    closed_set = set()
    num_nodes = 1

    while open_set:
        current = heapq.heappop(open_set)
        
        if current.position == goal:
            path = []
            while current:
                path.append(current)
                current = current.parent
            return path[::-1], num_nodes
        
        closed_set.add(current.position)
        
        for neighbor_pos, action in get_neighbors(grid, current):
            if neighbor_pos in closed_set:
                continue
            
            angle_cost = calculate_angle_cost(current.action, action, k)
            distance_cost = calculate_distance_cost(action)
            g = current.g + angle_cost + distance_cost
            h = heuristic(neighbor_pos, goal)
            
            neighbor = Node(neighbor_pos, g, h, current, action)
            num_nodes += 1
            
            if neighbor not in open_set:
                heapq.heappush(open_set, neighbor)
            else:
                idx = open_set.index(neighbor)
                if open_set[idx].g > g:
                    open_set[idx] = neighbor
                    heapq.heapify(open_set)
    
    return None, num_nodes  # No path found

def main():
    input_file = input("Enter input file name: ")
    output_file = input("Enter output file name: ")
    k = float(input("Enter k value: "))
    
    start, goal, grid = read_input(input_file)
    
    path, num_nodes = a_star(grid, start, goal, k)
    
    if path:
        for node in path[1:-1]:
            grid[node.position] = 4  # Mark path
        
        write_output(output_file, path, num_nodes, grid)
        print(f"Path found! Check {output_file} for results.")
    else:
        print("No path found.")

if __name__ == '__main__':
    main()