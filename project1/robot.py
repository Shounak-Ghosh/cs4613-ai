import sys
import heapq
import numpy as np
from collections import deque, defaultdict

# Constants for the workspace dimensions
ROWS = 30
COLS = 50

# Define possible moves with associated actions
MOVE_SET = [
    (1, 0, 0),   # Move down
    (1, 1, 1),   # Move down-right
    (0, 1, 2),   # Move right
    (-1, 1, 3),  # Move up-right
    (-1, 0, 4),  # Move up
    (-1, -1, 5), # Move up-left
    (0, -1, 6),  # Move left
    (1, -1, 7)   # Move down-left
]

class Node:
    """
    Represents a node in the search tree with properties:
    - position: (row, col) tuple for current position
    - g: cost from the start node
    - h: heuristic estimated cost to goal
    - f: total cost (g + h)
    - parent: reference to parent node in the path
    - action: movement action to reach this node from parent
    """
    def __init__(self, position, g, h, parent=None, action=None):
        self.position = position
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent
        self.action = action

    def __lt__(self, other):
        # Comparison for priority queue based on f value
        return self.f < other.f

class PriorityQueue:
    """
    A priority queue implemented with a min-heap.
    """
    def __init__(self):
        self.elements = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = 0

    def add_task(self, task, priority=0):
        """
        Adds a new task or updates the priority of an existing task.
        """
        if task in self.entry_finder:
            self.remove_task(task)
        count = self.counter
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heapq.heappush(self.elements, entry)
        self.counter += 1

    def remove_task(self, task):
        """
        Marks an existing task as removed without deleting from the heap.
        """
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_task(self):
        """
        Removes and returns the task with the lowest priority from the queue.
        """
        while self.elements:
            priority, count, task = heapq.heappop(self.elements)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')

def read_input(filename):
    """
    Reads the input file and returns start, goal positions, and the workspace grid.
    - filename: input file containing the grid and start/goal positions
    """
    with open(filename, 'r') as f:
        # Read start and goal positions from the first line
        start_goal = list(map(int, f.readline().split()))
        workspace = []
        for _ in range(30):
            # Each row of the workspace grid
            row = list(map(int, f.readline().split()))
            workspace.append(row[:50])  # Ensure each row has exactly 50 elements
    return tuple(start_goal[:2]), tuple(start_goal[2:]), np.array(workspace)

def write_output(filename, depth, nodes_generated, path, f_values, workspace):
    """
    Writes the result to the output file in the required format.
    - depth: depth of the solution path
    - nodes_generated: total nodes generated during search
    - path: list of actions from start to goal
    - f_values: list of f values along the path
    - workspace: modified workspace grid showing the path
    """
    with open(filename, 'w') as f:
        f.write(f"{depth}\n")
        f.write(f"{nodes_generated}\n")
        f.write(' '.join(map(str, path)) + '\n')
        f.write(' '.join(f"{x:.1f}" for x in f_values) + '\n')
        for row in workspace:
            f.write(' '.join(map(str, row)) + '\n')

def heuristic(node, goal):
    """
    Computes the Euclidean distance heuristic between a node and the goal.
    """
    return np.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

def get_neighbors(position, workspace):
    """
    Returns all possible neighbors of a node, along with the action taken to reach each neighbor.
    Excludes neighbors that are out of bounds or obstacles.
    """
    i, j = position
    neighbors = []
    for di, dj, action in MOVE_SET:
        ni, nj = i + di, j + dj
        # Account for origin at the bottom left of the grid
        if 0 <= (ROWS - nj - 1) < workspace.shape[0] and 0 <= ni < workspace.shape[1] and workspace[ROWS - nj - 1][ni] != 1:
            neighbors.append(((ni, nj), action))
    return neighbors

def theta(s):
    """
    Calculates the angle in degrees of a vector from the origin to s.
    """
    return np.arctan2(s[1], s[0]) * 180 / np.pi

def cost(s, a, s_prime, k):
    """
    Computes the cost to move from s to s_prime with action a, incorporating both angular and distance costs.
    - s: current position
    - a: action to move to s_prime
    - s_prime: new position
    - k: scaling factor for angular cost
    """
    if s == s_prime:
        return 0
    angle_cost = k * min(abs(theta(s_prime) - theta(s)), 360 - abs(theta(s_prime) - theta(s))) / 180
    distance_cost = 1 if a in [0, 2, 4, 6] else np.sqrt(2)  # Diagonal moves cost more
    return angle_cost + distance_cost

def a_star_search(start, goal, workspace, k):
    """
    Implements the A* search algorithm to find the optimal path from start to goal.
    - start: starting position
    - goal: goal position
    - workspace: grid representing obstacles and free spaces
    - k: scaling factor for angular cost
    """
    start_node = Node(start, 0, heuristic(start, goal))
    open_queue = PriorityQueue()
    open_queue.add_task(start_node, start_node.f)
    closed_set = deque()  # Stores visited nodes
    nodes_generated = 1
    nodes_closed = 0

    while open_queue.elements:
        current = open_queue.pop_task()
        
        # Goal check
        if current.position == goal:
            path = []
            f_values = []
            depth = 0
            while current.parent:
                path.append(current.action)
                f_values.append(current.f)
                current = current.parent
                depth += 1
            f_values.append(current.f)
            return depth, nodes_generated, path[::-1], f_values[::-1]

        closed_set.append(current.position)
        nodes_closed += 1

        # print("nodes opened:", nodes_generated, "nodes closed:", nodes_closed,
        #       "current position:", current.position, "f value:", current.f)
        
        for neighbor, action in get_neighbors(current.position, workspace):
            if neighbor in closed_set:
                continue
            
            g = current.g + cost(current.position, action, neighbor, k)
            h = heuristic(neighbor, goal)
            neighbor_node = Node(neighbor, g, h, current, action)
            
            if neighbor_node not in open_queue.entry_finder:
                open_queue.add_task(neighbor_node, neighbor_node.f)
                nodes_generated += 1
            else:
                # Update if a better path is found
                existing_node = open_queue.entry_finder[neighbor_node].task
                if existing_node.g > g:
                    existing_node.g = g
                    existing_node.f = g + h
                    existing_node.parent = current
                    existing_node.action = action
                    open_queue.add_task(existing_node, existing_node.f)

    return None, nodes_generated, None, None

def main():
    """
    Main function to read input, run A* search, and output results.
    """
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <input_file> <output_file> <k_value>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    k = float(sys.argv[3])

    start, goal, workspace = read_input(input_file)
    depth, nodes_generated, path, f_values = a_star_search(start, goal, workspace, k)

    # Update workspace with the path if a solution is found
    if path:
        current = start
        for action in path:
            ni, nj = current
            if action in [0, 1, 7]:
                ni += 1
            elif action in [3, 4, 5]:
                ni -= 1
            if action in [1, 2, 3]:
                nj += 1
            elif action in [5, 6, 7]:
                nj -= 1
            if workspace[ROWS - nj - 1][ni] == 0:
                workspace[ROWS - nj - 1][ni] = 4
            current = (ni, nj)
        write_output(output_file, depth, nodes_generated, path, f_values, workspace)
        print(f"Path found and output to {output_file}.")
    else:
        print("No path found.")

if __name__ == "__main__":
    main()
