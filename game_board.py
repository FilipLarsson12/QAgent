import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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
        self.actions = ['up', 'down', 'left', 'right']
        
        self.fig, (self.ax_cbar, self.ax_maze) = plt.subplots(
            1, 2,
            figsize=(12, 6),
            gridspec_kw={'width_ratios': [0.05, 1]}
        )
        plt.subplots_adjust(left=0.1, right=0.9)

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
                       for j in range(self.width) if self.grid[i][j] == 0 and self.has_free_neighbor(i, j)]
        return random.choice(empty_cells)

    def place_start_and_goal(self):
        self.grid[self.start_pos] = 1
        self.grid[self.goal_pos] = 2

    def has_free_neighbor(self, i, j):
        neighbors = [
            (i-1, j),
            (i+1, j),
            (i, j-1),
            (i, j+1)
        ]
        for ni, nj in neighbors:
            if self.grid[ni][nj] == 0:
                return True
        return False

    def visualize(
        self,
        agent_pos=None,
        step_number=None,
        current_episode=None,
        current_score=None,
        frame_rate=5.0,
        split="Train",
        show_q_values=False,
        q_table=None
    ):
        self.ax_maze.cla()
        self.ax_cbar.cla()
        
        cmap = plt.cm.get_cmap('Accent', 4)
        self.ax_maze.imshow(
            self.grid,
            cmap=cmap,
            origin='upper',
            extent=[0, self.width, self.height, 0]
        )
        
        plot_buffer = 0.5
        
        if self.goal_pos:
            self.ax_maze.scatter(
                self.goal_pos[1] + plot_buffer,
                self.goal_pos[0] + plot_buffer,
                c='red',
                marker='*',
                s=100,
                label='Goal'
            )
        
        if agent_pos:
            self.ax_maze.scatter(
                agent_pos[1] + plot_buffer,
                agent_pos[0] + plot_buffer,
                c='black',
                s=1400//self.width,
                label='Agent'
            )
        
        if show_q_values and q_table is not None:
            max_q_matrix = np.full((self.height, self.width), np.nan)
            for i in range(self.height):
                for j in range(self.width):
                    state = (i, j)
                    if self.grid[i][j] != -1:
                        q_values = [q_table.get((state, action), 0) for action in self.actions]
                        max_q = max(q_values) if q_values else 0
                        max_q_matrix[i][j] = max_q
            
            if np.nanmax(max_q_matrix) == np.nanmin(max_q_matrix):
                normalized_q = np.zeros_like(max_q_matrix)
            else:
                normalized_q = (max_q_matrix - np.nanmin(max_q_matrix)) / (np.nanmax(max_q_matrix) - np.nanmin(max_q_matrix) + 1e-6)
            
            q_heatmap = self.ax_maze.imshow(
                normalized_q,
                cmap='viridis',
                alpha=0.5,
                origin='upper',
                extent=[0, self.width, self.height, 0],
                interpolation='nearest'
            )
            
            cbar = self.fig.colorbar(
                q_heatmap,
                cax=self.ax_cbar,
                orientation='vertical'
            )
            cbar.set_label('Max Q-Value', rotation=270, labelpad=15)
        
        title_parts = [f"{split} Split"]
        if current_episode is not None:
            title_parts.append(f"Episode: {current_episode}")
        if step_number is not None:
            title_parts.append(f"Step: {step_number}")
        if current_score is not None:
            title_parts.append(f"Score: {current_score}")
        if agent_pos == self.goal_pos:
            title_parts.append("Goal Reached!")
        
        title = " | ".join(title_parts)
        self.ax_maze.set_title(title)
        
        legend_elements = [
            Line2D([0], [0], marker='*', color='w', label='Goal', markerfacecolor='red', markersize=15),
            Line2D([0], [0], marker='o', color='w', label='Agent', markerfacecolor='black', markersize=15)
        ]
        
        existing_legend = self.ax_maze.legend(handles=legend_elements, loc='upper right')
        self.ax_maze.add_artist(existing_legend)
        
        rewards_legend_elements = [
            Patch(facecolor='lightgray', edgecolor='black', label='Empty (-1)'),
            Patch(facecolor='black', edgecolor='black', label='Obstacle (-100)'),
            Line2D([0], [0], marker='*', color='w', label='Goal (+100)', markerfacecolor='red', markersize=15)
        ]
        
        rewards_legend = self.ax_maze.legend(
            handles=rewards_legend_elements,
            loc='lower right',
            title='Rewards'
        )
        
        self.ax_maze.add_artist(rewards_legend)
        
        ticks_to_skip = self.width // 10 if self.width >= 10 else 1
        self.ax_maze.set_xticks(np.arange(0, self.width + 1, ticks_to_skip))
        self.ax_maze.set_yticks(np.arange(0, self.height + 1, ticks_to_skip))
        minor_ticks = np.arange(0, max(self.width, self.height) + 1, 1)
        self.ax_maze.set_xticks(minor_ticks, minor=True)
        self.ax_maze.set_yticks(minor_ticks, minor=True)
        self.ax_maze.grid(True, which='both', linestyle='-', linewidth=0.5)
        self.ax_maze.grid(True, which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(1.0 / frame_rate)

class Agent:
    def __init__(self, board):
        self.board = board
        self.position = board.start_pos
        self.actions = ['up', 'down', 'left', 'right']
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.actions)
        else:
            q_values = [self.q_table.get((state, a), 0) for a in self.actions]
            max_q = max(q_values)
            max_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
            action = random.choice(max_actions)
        return action
    
    def update_q_value(self, state, action, reward, next_state):
        old_value = self.q_table.get((state, action), 0)
        next_max = max([self.q_table.get((next_state, a), 0) for a in self.actions])
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[(state, action)] = new_value

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

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

    def get_state(self):
        return self.position

    def visualize(
        self,
        step_number=None,
        current_episode=None,
        current_score=None,
        frame_rate=5.0,
        split="Train",
        show_q_values=False
    ):
        self.board.visualize(
            agent_pos=self.position,
            step_number=step_number,
            current_episode=current_episode,
            current_score=current_score,
            frame_rate=frame_rate,
            split=split,
            show_q_values=show_q_values,
            q_table=self.q_table if show_q_values else None
        )

def main():
    board = Board(width=40, height=40, obstacle_ratio=0.2)
    agent = Agent(board)
    num_episodes = 100000

    for episode in range(num_episodes):
        agent.position = board.start_pos
        state = agent.get_state()
        total_reward = 0
        done = False

        for step in range(500):
            if (episode) % 50000 == 0:
                agent.visualize(
                    step_number=step,
                    current_episode=episode,
                    current_score=total_reward,
                    split="Train",
                    show_q_values=True
                )

            action = agent.choose_action(state)
            reward, next_state_pos, goal_reached, obstacle_hit = agent.move(action)
            next_state = next_state_pos
            total_reward += reward

            agent.update_q_value(state, action, reward, next_state)

            state = next_state

            if goal_reached or obstacle_hit:
                break

        agent.decay_epsilon()

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    agent.epsilon = 0
    agent.position = board.start_pos
    state = agent.get_state()
    done = False
    path = [agent.position]
    step = 0

    while not done:
        agent.visualize(step, frame_rate=5, split="Test", show_q_values=True)
        action = agent.choose_action(state)
        reward, next_state_pos, goal_reached, obstacle_hit = agent.move(action)
        state = next_state_pos
        path.append(agent.position)

        if goal_reached:
            print("Goal reached!")
            agent.visualize(split="Test", show_q_values=True)
            break
        if obstacle_hit:
            print("Obstacle hit during test!")
            agent.visualize(split="Test", show_q_values=True)
            break

        if len(path) > 500:
            print("Failed to reach the goal within step limit.")
            break
        step += 1

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
