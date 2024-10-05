import matplotlib.pyplot as plt
import numpy as np
import random

plt.ion()

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
        for i in range(self.width):
            grid[0, i] = -1
            grid[self.height-1, i] = -1
        for i in range(self.height):
            grid[i, 0] = -1
            grid[i, self.width-1] = -1
        num_obstacles = int((self.width - 2) * (self.height - 2) * self.obstacle_ratio)
        possible_positions = [
            (i, j)
            for i in range(1, self.height - 1)
            for j in range(1, self.width - 1)
        ]
        obstacles = random.sample(possible_positions, num_obstacles)
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
        plt.clf()
        cmap = plt.cm.get_cmap('Accent', 4)
        plt.imshow(self.grid, cmap=cmap, origin='upper', extent=[0, self.width, self.height, 0])
        plot_buffer = 0.5
        if agent_pos:
            plt.scatter(agent_pos[1] + plot_buffer, agent_pos[0] + plot_buffer, c='black', s=40, label='Agent')
        ticks_to_skip = self.width // 10
        plt.xticks(np.arange(0, self.width + 1, ticks_to_skip))
        plt.yticks(np.arange(0, self.height + 1, ticks_to_skip))
        minor_ticks = np.arange(0, max(self.width, self.height) + 1, 1)
        plt.gca().set_xticks(minor_ticks, minor=True)
        plt.gca().set_yticks(minor_ticks, minor=True)
        plt.grid(True, which='both', linestyle='-', linewidth=0.5)
        plt.grid(True, which='minor', color='gray', linestyle='-', linewidth=0.5)
        plt.legend(loc='upper right')
        plt.draw()
        plt.pause(0.2)

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
        reward = -1
        goal_reached = False
        obstacle_hit = False
        if action in actions:
            new_row = self.position[0] + actions[action][0]
            new_col = self.position[1] + actions[action][1]
            if self.board.grid[new_row][new_col] != -1:
                self.position = (new_row, new_col)
                if self.position == self.board.goal_pos:
                    reward = 100
                    goal_reached = True
            else:
                reward = -100
                obstacle_hit = True
        return reward, self.position, goal_reached, obstacle_hit

    def visualize(self):
        self.board.visualize(agent_pos=self.position)

def main():
    board = Board(width=40, height=40, obstacle_ratio=0.2)
    agent = Agent(board)
    actions = ['up', 'down', 'left', 'right']
    reward_sum = 0
    for _ in range(100):
        agent.visualize()
        action = random.choice(actions)
        reward, new_pos, goal_reached, obstacle_hit = agent.move(action)
        reward_sum += reward
        if goal_reached:
            print("Goal reached!")
            print("Accumulated reward: ", reward_sum)
            break
        if obstacle_hit:
            print("Obstacle hit!")
            print("Accumulated reward: ", reward_sum)
            break
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
