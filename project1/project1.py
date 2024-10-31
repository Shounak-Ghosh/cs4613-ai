import sys
import heapq
import numpy as np

ROWS = 30
COLS = 50
class Node:
    def __init__(self, position, g, h, parent=None, action=None):
        self.position = position
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent
        self.action = action

    def __lt__(self, other):
        return self.f < other.f

def read_input(filename):
    with open(filename, 'r') as f:
        start_goal = list(map(int, f.readline().split()))
        workspace = []
        for _ in range(30):
            row = list(map(int, f.readline().split()))
            workspace.append(row[:50])  # Ensure each row has exactly 50 elements
    return tuple(start_goal[:2]), tuple(start_goal[2:]), np.array(workspace)

def write_output(filename, depth, nodes_generated, path, f_values, workspace):
    with open(filename, 'w') as f:
        f.write(f"{depth}\n")
        f.write(f"{nodes_generated}\n")
        f.write(' '.join(map(str, path)) + '\n')
        f.write(' '.join(f"{x:.1f}" for x in f_values) + '\n')
        for row in workspace:
            f.write(' '.join(map(str, row)) + '\n')

def heuristic(node, goal):
    return np.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

def get_neighbors(position, workspace):
    i, j = position
    neighbors = []
    for di, dj, action in [(1,0,0), (1,1,1), (0,1,2), (-1,1,3), (-1,0,4), (-1,-1,5), (0,-1,6), (1,-1,7)]:
        ni, nj = i + di, j + dj
        if 0 <= (ROWS - nj - 1) < workspace.shape[0] and 0 <= ni < workspace.shape[1] and workspace[ROWS-nj-1][ni] != 1:
            neighbors.append(((ni, nj), action))
    return neighbors

def theta(s):
    return np.arctan2(s[1], s[0]) * 180 / np.pi

def cost(s, a, s_prime, k):
    if s == s_prime:
        return 0
    angle_cost = k * min(abs(theta(s_prime) - theta(s)), 360 - abs(theta(s_prime) - theta(s))) / 180
    distance_cost = 1 if a in [0, 2, 4, 6] else np.sqrt(2)
    return angle_cost + distance_cost

def a_star_search(start, goal, workspace, k):
    start_node = Node(start, 0, heuristic(start, goal))
    open_list = [start_node]
    closed_set = set()
    nodes_generated = 1
    
    while open_list:
        current = heapq.heappop(open_list)
        
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
        
        closed_set.add(current.position)
        
        for neighbor, action in get_neighbors(current.position, workspace):
            if neighbor in closed_set:
                continue
            
            g = current.g + cost(current.position, action, neighbor, k)
            h = heuristic(neighbor, goal)
            neighbor_node = Node(neighbor, g, h, current, action)
            
            if neighbor_node not in open_list:
                heapq.heappush(open_list, neighbor_node)
                nodes_generated += 1
            else:
                idx = open_list.index(neighbor_node)
                if open_list[idx].g > g:
                    open_list[idx] = neighbor_node
                    heapq.heapify(open_list)
    
    return None, nodes_generated, None, None

def main():
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <input_file> <output_file> <k_value>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    k = float(sys.argv[3])

    start, goal, workspace = read_input(input_file)
    depth, nodes_generated, path, f_values = a_star_search(start, goal, workspace, k)

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
            if workspace[ROWS-nj-1][ni] == 0:
                workspace[ROWS-nj-1][ni] = 4
            current = (ni, nj)
        print(f"Path found and output to {output_file}.")
    else:
        print("No path found.")

    write_output(output_file, depth, nodes_generated, path, f_values, workspace)

if __name__ == "__main__":
    main()