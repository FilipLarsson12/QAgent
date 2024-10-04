import matplotlib.pyplot as plt
import numpy as np
import random

class Board:
    def __init__(self, width, height, obstacle_ratio=0.2):
        self.width = width
        self.height = height
        self.obstacle_ratio = obstacle_ratio
        self.grid = self.generate_grid()
        self.start_pos = self.get_random_empty_cell()
        self.goal_pos = self.get_random_empty_cell()
        self.place_start_and_goal()

    def generate_grid(self):
        grid = np.zeros((self.height, self.width), dtype=int)
        num_obstacles = int(self.width * self.height * self.obstacle_ratio)
        obstacles = random.sample(
            [(i, j) for i in range(self.height) for j in range(self.width)],
            num_obstacles
        )
        for (i, j) in obstacles:
            grid[i][j] = -1
        return grid

    def get_random_empty_cell(self):
        empty_cells = [(i, j) for i in range(self.height)
                       for j in range(self.width) if self.grid[i][j] == 0]
        return random.choice(empty_cells)

    def place_start_and_goal(self):
        self.grid[self.start_pos] = 1
        self.grid[self.goal_pos] = 2

    def visualize(self, agent_pos=None):
        plt.figure(figsize=(6, 6))
        cmap = plt.cm.get_cmap('Accent', 4)
        plt.imshow(self.grid, cmap=cmap, origin='upper', extent=[0, self.width, self.height, 0])
        if agent_pos:
            plt.scatter(agent_pos[1], agent_pos[0], c='red', s=200, label='Agent')
        plt.xticks(np.arange(self.width + 1))
        plt.yticks(np.arange(self.height + 1))
        plt.grid(True)
        plt.legend()
        plt.show()

def main():
    board = Board(width=20, height=20)
    board.visualize()

if __name__ == "__main__":
    main()
