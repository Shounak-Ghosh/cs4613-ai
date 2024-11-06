#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <queue>
#include <tuple>
#include <set>
#include <algorithm>

using namespace std;

// Constants for the grid dimensions and possible movements
const int ROWS = 30;
const int COLS = 50;
const vector<tuple<int, int, int>> MOVE_SET = {{1, 0, 0}, {1, 1, 1}, {0, 1, 2}, {-1, 1, 3},
                                               {-1, 0, 4}, {-1, -1, 5}, {0, -1, 6}, {1, -1, 7}};

// Node structure to represent each position in the search tree
struct Node {
    pair<int, int> position;  // Current position (row, col)
    double g, h, f;           // g: cost from start, h: heuristic to goal, f: total cost (g + h)
    Node* parent;             // Pointer to parent node for path reconstruction
    int action;               // Action taken to reach this node from its parent

    // Node constructor
    Node(pair<int, int> pos, double g_cost, double h_cost, Node* par = nullptr, int act = -1)
        : position(pos), g(g_cost), h(h_cost), f(g + h), parent(par), action(act) {}

    // Operator overload for priority queue comparison (min-heap based on f)
    bool operator<(const Node& other) const {
        return f > other.f;  // Inverted for priority queue (smallest f on top)
    }
};

// Function to read the input file and return start, goal positions, and the workspace grid
tuple<pair<int, int>, pair<int, int>, vector<vector<int>>> read_input(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: File does not exist or could not be opened.\n";
        exit(1);
    }

    int start_x, start_y, goal_x, goal_y;
    file >> start_x >> start_y >> goal_x >> goal_y;

    // Initialize and populate the workspace grid
    vector<vector<int>> workspace(ROWS, vector<int>(COLS));
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            file >> workspace[i][j];
        }
    }
    return {{start_x, start_y}, {goal_x, goal_y}, workspace};
}

// Function to write the results to the output file in the required format
void write_output(const string& filename, int depth, int nodes_generated, const vector<int>& path,
                  const vector<double>& f_values, const vector<vector<int>>& workspace) {
    ofstream file(filename);
    file << depth << "\n" << nodes_generated << "\n";

    // Write the sequence of actions taken in the solution path
    for (int action : path) file << action << " ";
    file << "\n";

    // Write the f values along the path with 1 decimal precision
    for (double f_val : f_values) file << fixed << setprecision(1) << f_val << " ";
    file << "\n";

    // Write the final workspace grid with the path marked
    for (const auto& row : workspace) {
        for (int cell : row) file << cell << " ";
        file << "\n";
    }
}

// Heuristic function to estimate cost from a node to the goal (Euclidean distance)
double heuristic(const pair<int, int>& node, const pair<int, int>& goal) {
    return sqrt(pow(node.first - goal.first, 2) + pow(node.second - goal.second, 2));
}

// Function to get all valid neighbors of a node, with movement actions
vector<pair<pair<int, int>, int>> get_neighbors(const pair<int, int>& position, const vector<vector<int>>& workspace) {
    int i = position.first, j = position.second;
    vector<pair<pair<int, int>, int>> neighbors;

    // Loop through possible moves and check for validity (bounds, obstacles)
    // unpack move as structured binding (di, dj, action)
    for (const auto& [di, dj, action] : MOVE_SET) {
        int ni = i + di, nj = j + dj;
        if (0 <= nj && nj < ROWS && 0 <= ni && ni < COLS && workspace[ROWS - nj - 1][ni] != 1) {
            neighbors.push_back({{ni, nj}, action});
        }
    }
    return neighbors;
}

// Helper function to calculate angle (in degrees) of a vector from the origin to a point s
double theta(const pair<int, int>& s) {
    return atan2(s.second, s.first) * 180.0 / M_PI;
}

// Function to compute movement cost between nodes based on action and angle difference
double cost(const pair<int, int>& s, int a, const pair<int, int>& s_prime, double k) {
    if (s == s_prime) return 0.0;  // No cost for staying in the same position

    double angle_cost = k * min(abs(theta(s_prime) - theta(s)), 360 - abs(theta(s_prime) - theta(s))) / 180;
    double distance_cost = (a % 2 == 0) ? 1 : sqrt(2);  // Diagonal movements cost more
    return angle_cost + distance_cost;
}

// A* search algorithm to find the optimal path from start to goal, returning path details
tuple<int, int, vector<int>, vector<double>> a_star_search(const pair<int, int>& start,
                                                           const pair<int, int>& goal,
                                                           vector<vector<int>>& workspace,
                                                           double k) {
    priority_queue<Node> open_list;    // Priority queue for open nodes
    set<pair<int, int>> closed_set;    // Set to keep track of visited nodes
    int nodes_generated = 1;

    Node* start_node = new Node(start, 0, heuristic(start, goal));
    open_list.push(*start_node);

    while (!open_list.empty()) {
        Node current = open_list.top();
        open_list.pop();

        // Check if goal is reached
        if (current.position == goal) {
            vector<int> path;
            vector<double> f_values;
            int depth = 0;

            // Backtrack to build the path from start to goal
            for (Node* node = &current; node != nullptr; node = node->parent) {
                if (node->action != -1) path.push_back(node->action);
                f_values.push_back(node->f);
                depth++;
            }

            cout << path.size() << " " << f_values.size() << endl;

            reverse(path.begin(), path.end());
            reverse(f_values.begin(), f_values.end());

            delete start_node;
            return {depth, nodes_generated, path, f_values};
        }

        closed_set.insert(current.position);

        // Expand neighbors of the current node
        for (const auto& [neighbor, action] : get_neighbors(current.position, workspace)) {
            // Skip if neighbor is already visited
            if (closed_set.find(neighbor) != closed_set.end()) continue;

            double g = current.g + cost(current.position, action, neighbor, k);
            double h = heuristic(neighbor, goal);
            Node* neighbor_node = new Node(neighbor, g, h, new Node(current), action);

            open_list.push(*neighbor_node);
            nodes_generated++;
        }
    }

    delete start_node;
    return {0, nodes_generated, {}, {}};
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file> <k_value>\n";
        return 1;
    }

    string input_file = argv[1];
    string output_file = argv[2];
    double k = stod(argv[3]);

    // Read input file for start and goal positions, workspace grid
    auto [start, goal, workspace] = read_input(input_file);
    auto [depth, nodes_generated, path, f_values] = a_star_search(start, goal, workspace, k);

    // Update workspace with the path if a solution is found
    if (!path.empty()) {
        pair<int, int> current = start;
        for (int action : path) {
            int ni = current.first, nj = current.second;
            if (action == 0 || action == 1 || action == 7) ni += 1;
            if (action == 3 || action == 4 || action == 5) ni -= 1;
            if (action == 1 || action == 2 || action == 3) nj += 1;
            if (action == 5 || action == 6 || action == 7) nj -= 1;
            if (workspace[ROWS - nj - 1][ni] == 0) workspace[ROWS - nj - 1][ni] = 4;
            current = {ni, nj};
        }
        write_output(output_file, depth, nodes_generated, path, f_values, workspace);
        cout << "Path found and output to " << output_file << ".\n";
    } else {
        cout << "No path found.\n";
    }

    return 0;
}
