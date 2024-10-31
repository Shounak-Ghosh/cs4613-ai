import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

def plot_maze(lines_discarded,file_path):
    try:
        # Load the maze from the .txt file
        with open(file_path, "r") as file:
            maze = np.loadtxt(file_path, delimiter=" ",skiprows=lines_discarded, usecols=range(50))

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 7))
        nrows, ncols = maze.shape

        # Define colors for each tile type
        colors = {0: "white", 1: "black", 2: "red", 3: "white", 4: "yellow", 5: "green"}

        # Plot each tile
        for row in range(nrows):
            for col in range(ncols):
                tile_color = colors.get(
                    maze[row, col], "white"
                )  # Default to white if unknown
                rect = Rectangle(
                    (col, row), 1, 1, edgecolor="black", facecolor=tile_color
                )
                ax.add_patch(rect)

        plt.xlim(0, ncols)
        plt.ylim(0, nrows)
        plt.gca().invert_yaxis()  # Invert y-axis to match array layout
        plt.axis("off")  # Turn off the axis
        plt.show()

    except Exception as e:
        print(f"Failed to read the file: {e}")

def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <num_lines_to_discard> <filename>")
        sys.exit(1)
    
    file_path = sys.argv[2]
    lines_discarded = int(sys.argv[1])
    plot_maze(lines_discarded , file_path)

if __name__ == "__main__":
    main()