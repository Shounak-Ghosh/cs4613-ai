#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>

using namespace std;

const int N = 9; // The size of the Sudoku grid (9x9)
const int EMPTY = 0; // The constant representing an empty cell in the Sudoku grid

// A struct to represent each cell in the Kropki Sudoku grid.
struct Cell {
    int value;         
    vector<int> domain;
    // Constructor initializes an empty cell with all domain possible values
    Cell() : value(EMPTY), domain{1,2,3,4,5,6,7,8,9} {} 
};

class KropkiSudoku {
private:
    Cell board[N][N]; 
    int horizontal_dots[N][N-1]; 
    int vertical_dots[N-1][N];

    // Method to check if placing a number in a cell is valid
    bool is_valid(int row, int col, int num) {
        // Check the row and column for conflicts
        for (int i = 0; i < N; i++) {
            if (board[row][i].value == num || board[i][col].value == num) return false; 
        }

        // Check the 3x3 box for conflicts
        int box_row = row - row % 3, box_col = col - col % 3;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (board[box_row + i][box_col + j].value == num) return false; 
            }
        }

        // Check the Kropki dot constraints (horizontal and vertical)
        // 1 represents a white dot, 2 represents a black dot
        if (col > 0) { // Check left (horizontal dot)
            int left = board[row][col-1].value;
            if (left != EMPTY) {
                if (horizontal_dots[row][col-1] == 1 && abs(num - left) != 1) return false; 
                if (horizontal_dots[row][col-1] == 2 && num != 2*left && left != 2*num) return false;
            }
        }
        if (col < N-1) { // Check right (horizontal dot)
            int right = board[row][col+1].value;
            if (right != EMPTY) {
                if (horizontal_dots[row][col] == 1 && abs(num - right) != 1) return false;
                if (horizontal_dots[row][col] == 2 && num != 2*right && right != 2*num) return false;
            }
        }
        if (row > 0) { // Check up (vertical dot)
            int up = board[row-1][col].value;
            if (up != EMPTY) {
                if (vertical_dots[row-1][col] == 1 && abs(num - up) != 1) return false;
                if (vertical_dots[row-1][col] == 2 && num != 2*up && up != 2*num) return false;
            }
        }
        if (row < N-1) { // Check down (vertical dot)
            int down = board[row+1][col].value;
            if (down != EMPTY) {
                if (vertical_dots[row][col] == 1 && abs(num - down) != 1) return false;
                if (vertical_dots[row][col] == 2 && num != 2*down && down != 2*num) return false;
            }
        }

        return true;
    }

    // Select an unassigned variable (cell) to make the next move
    pair<int, int> select_unassigned_variable() {
        // initialize to -1
        int min_remaining = 10, max_degree = -1;
        pair<int, int> selected = {-1, -1};

        // Iterate over all cells in the grid
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (board[i][j].value == EMPTY) { 
                    int remaining = board[i][j].domain.size(); // Number of possible values for the cell
                    int degree = count_constraints(i, j); 

                    // Select the cell with the fewest remaining values, and if tied, the one with the most constraints
                    if (remaining < min_remaining || (remaining == min_remaining && degree > max_degree)) {
                        min_remaining = remaining;
                        max_degree = degree;
                        selected = {i, j};
                    }
                }
            }
        }

        return selected;
    }

    // Count the number of constraints (empty cells that are in the same row, column, or 3x3 box)
    int count_constraints(int row, int col) {
        int count = 0;
        // Check row and column
        for (int i = 0; i < N; i++) {
            if (i != col && board[row][i].value == EMPTY) count++;
            if (i != row && board[i][col].value == EMPTY) count++;
        }

        // Check the 3x3 box
        int box_row = row - row % 3, box_col = col - col % 3;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (box_row + i != row && box_col + j != col && board[box_row + i][box_col + j].value == EMPTY) count++;
            }
        }
        return count; 
    }

    // Perform forward checking: if placing a number in a cell causes a domain to become empty in any neighboring cell, return false
    bool forward_checking(int row, int col, int num) {
        // Check the row and column for any affected cells
        for (int i = 0; i < N; i++) {
            if (i != col && board[row][i].value == EMPTY) {
                auto& domain = board[row][i].domain;
                // Remove the number from the domain of the affected cell
                domain.erase(remove(domain.begin(), domain.end(), num), domain.end());
                if (domain.empty()) return false; // If the domain becomes empty, return false
            }
            if (i != row && board[i][col].value == EMPTY) {
                auto& domain = board[i][col].domain;
                domain.erase(remove(domain.begin(), domain.end(), num), domain.end());
                if (domain.empty()) return false;
            }
        }

        // Check the 3x3 box for affected cells
        int box_row = row - row % 3, box_col = col - col % 3;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (box_row + i != row && box_col + j != col && board[box_row + i][box_col + j].value == EMPTY) {
                    auto& domain = board[box_row + i][box_col + j].domain;
                    domain.erase(remove(domain.begin(), domain.end(), num), domain.end());
                    if (domain.empty()) return false;
                }
            }
        }

        return true; // No empty domains
    }

    // Restore the domains of affected cells after a backtrack
    void restore_domains(const vector<pair<int, int>>& affected_cells, int num) {
        for (const auto& cell : affected_cells) {
            board[cell.first][cell.second].domain.push_back(num);
        }
    }

    // The recursive backtracking algorithm for solving the puzzle
    bool solve() {
        auto [row, col] = select_unassigned_variable(); // Select the next unassigned variable
        if (row == -1 && col == -1) return true; // If there are no unassigned variables, the puzzle is solved

        // Try each value from the domain of the selected cell
        for (int num : board[row][col].domain) {
            if (is_valid(row, col, num)) { // Check if placing the number is valid
                board[row][col].value = num; // Assign the number to the cell

                // Store the affected cells (those in the same row, column, and box)
                vector<pair<int, int>> affected_cells;
                for (int i = 0; i < N; i++) {
                    if (i != col && board[row][i].value == EMPTY) affected_cells.emplace_back(row, i);
                    if (i != row && board[i][col].value == EMPTY) affected_cells.emplace_back(i, col);
                }
                int box_row = row - row % 3, box_col = col - col % 3;
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        if (box_row + i != row && box_col + j != col && board[box_row + i][box_col + j].value == EMPTY) {
                            affected_cells.emplace_back(box_row + i, box_col + j);
                        }
                    }
                }

                // Perform forward checking and try to solve recursively
                if (forward_checking(row, col, num)) {
                    if (solve()) return true;
                }

                board[row][col].value = EMPTY; // Undo the assignment
                restore_domains(affected_cells, num); // Restore the domains of affected cells
            }
        }

        return false; // If no valid number can be placed, return false (backtrack)
    }

public:
    // Read the Sudoku puzzle from the input file
    void read_input(const string& filename) {
        ifstream file(filename);
        // Read the Sudoku board values
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                file >> board[i][j].value;
                // The cell has a value
                if (board[i][j].value != EMPTY) {
                    board[i][j].domain.clear(); // Reset the domain to the value
                    board[i][j].domain.push_back(board[i][j].value);
                }
            }
        }
        file.ignore(); // Ignore any newline characters
        // Read the horizontal dots
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N-1; j++) {
                file >> horizontal_dots[i][j];
            }
        }
        file.ignore();
        // Read the vertical dots
        for (int i = 0; i < N-1; i++) {
            for (int j = 0; j < N; j++) {
                file >> vertical_dots[i][j];
            }
        }
        file.close();
    }

    // Write the solution to the output file
    void write_output(const string& filename) {
        ofstream file(filename);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                file << board[i][j].value << " ";
            }
            file << "\n";
        }
        file.close();
    }

    // Solve the Sudoku puzzle
    bool solve_puzzle() {
        return solve();
    }
};

// Main function to handle the command-line input/output
int main(int argc, char* argv[]) {
    if (argc != 3) { // Check if the correct number of arguments are provided
        cout << "Usage: " << argv[0] << " <input_filename> <output_filename>" << endl;
        return 1; // Exit with an error code
    }

    string input_filename = argv[1]; // Input file name (from command-line argument)
    string output_filename = argv[2]; // Output file name (from command-line argument)

    KropkiSudoku puzzle;
    puzzle.read_input(input_filename); // Read the puzzle from the input file
    if (puzzle.solve_puzzle()) { // Try to solve the puzzle
        puzzle.write_output(output_filename); // Write the solution to the output file
        cout << "Puzzle solved successfully!" << endl;
    } else {
        cout << "No solution exists." << endl;
    }
    return 0;
}
