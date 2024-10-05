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
        plt.figure(figsize=(10, 10))
        cmap = plt.cm.get_cmap('Accent', 4)
        plt.imshow(self.grid, cmap=cmap, origin='upper', extent=[0, self.width, self.height, 0])
        plot_buffer = 0.5
        if agent_pos:
            plt.scatter(agent_pos[1] + plot_buffer, agent_pos[0] + plot_buffer, c='black', s=8, label='Agent')
        ticks_to_skip = self.width // 20
        plt.xticks(np.arange(0, self.width + 1, ticks_to_skip))
        plt.yticks(np.arange(0, self.height + 1, ticks_to_skip))
        minor_ticks = np.arange(0, max(self.width, self.height) + 1, 1)
        plt.gca().set_xticks(minor_ticks, minor=True)
        plt.gca().set_yticks(minor_ticks, minor=True)
        plt.grid(True, which='both', linestyle='-', linewidth=0.5)
        plt.grid(True, which='minor', color='gray', linestyle='-', linewidth=0.5)
        plt.legend()
        plt.show()

class Agent:
    def __init__(self, board):
        self.board = board
        self.position = board.start_pos

    def move(self, action):
        actions = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        if action in actions:
            new_row = self.position[0] + actions[action][0]
            new_col = self.position[1] + actions[action][1]
            if 0 <= new_row < self.board.height and 0 <= new_col < self.board.width:
                if self.board.grid[new_row][new_col] != -1:
                    self.position = (new_row, new_col)
                    return True
        return False

    def visualize(self):
        self.board.visualize(agent_pos=self.position)

def main():
    board = Board(width=100, height=100, obstacle_ratio=0.2)
    agent = Agent(board)
    agent.visualize()

if __name__ == "__main__":
    main()
