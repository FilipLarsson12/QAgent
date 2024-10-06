# Maze Navigator with Q-Learning

Welcome to the **Maze Navigator Game with Q-Learning Agent**! This project focuses on teaching an agent how to navigate through a randomly generated maze using Q-Learning. 
By updating what action the agent takes given a certain state, the agent learns the best path from the start to the goal while making sure not moving into any obstacles.

## 🧠 What is Q-Learning?

You can think of **Q-Learning** as teaching someone to find their way through a maze by rewarding them when they make good moves and discouraging them when they hit dead ends. It's a type of **reinforcement learning**, where the agent (our maze navigator) learns by interacting with its environment and learning from the consequences of its actions.

So we, the programmers, define different rewards for different outcomes inside the game. These rewards give feedback to the agent when it takes different actions. Based on this feedback, the agent updates what actions it takes in different states.

Here is an explanation that is a little bit more precise and technical using this maze-game as the example:

### **Defining Rewards**

First step is to define rewards for events in the maze-game:

- **+100** points for reaching the goal.
- **-100** points for hitting an obstacle.
- **-1** point for going to an empty cell.

You might be confused why I give a negative reward for going to a new empty cell, but the reason for this is that this reward will make the agent choose paths to the goal that are as short as possible.

### **Implementing the Q-Table**

Great, now our rewards are defined. Now we implement what is called a **Q table**. The Q table contains the **"Quality"** of an action in a given state. 

For example, the "Quality" of moving "up" when we stand on the (5,4) cell. The Q table contains entries for every state-action pair, for example, for the state (5,4) it contains one entry for every action in `['up', 'down', 'left', 'right']`.

At the start of the game, the Q table contains zeros for all state-action combinations. In a given state, the agent takes the action that has the highest Q value (Quality) in the Q table. So when we start to update this table, we want to give certain actions in certain states higher Q values than others.

These updates are based on rewards received for choosing certain actions in certain states, and below we will go through the equation responsible for actually updating the "Qualities" for different state-action pairs.

### **Understanding Bellman's Equation**

At the core of Q-Learning is **Bellman's Equation**.

The role of this equation is to update which actions we take in a particular state based on immediate feedback (reward) and estimated maximum future rewards that we can get from the new state that we go to.

Imagine we are in a particular spot in the maze (**state**) and choose to go in a particular direction (**action**), and from that action, we get a particular reward.

These are the variables we need to be able to write down Bellman's equation:

- **s** = state (already explained)
- **a** = action (already explained)
- **r** = reward (already explained)
- **lr** = learning_rate (how much should we update a Q value based on new information?)
- **df** = discount_factor (how much do we care about long-term effects of taking an action compared to short-term effects?)
- **s'** = next_state (state we reach by taking action **a** in state **s**)
- **a'** = next_action (best action we can take in the next state: **s'**, i.e. the action that will give the highest estimated accumulative reward in state **s'**)

So, remember that Bellman's equation takes this information and updates the Q value for that state-action pair based on the reward we got.

This is the mathematical definition of how Bellman's equation updates a Q value:

Q(s, a) = Q(s, a) + lr * (r + df * Q(s', a') - Q(s, a))

### **Intuitive Example**

I know it might be a lot to take in, but imagine this scenario:

- We are standing **two cells** from the goal.
- We choose an action that takes us **one cell** from the goal.

When we update this action based on Bellman's equation, the term **Q(s', a')** will get very large because it represents the maximum estimated accumulative reward from taking the best action in state **s'**.

In our case, state **s'** is **one cell** from the goal. So, the best action is to go to the goal and gain a massive reward. Therefore, **Q(s', a')** becomes big. 

Then, as I mentioned above, the action we took when we were **2 cells** away from the goal will also become big since **Q(s', a')** was big. 

These rewards from reaching the goal kind of roll back to actions that took us closer to the goal when we stood **3, 4, 5, 6 cells** away from the goal and so on. 

In this way, we will get high Q values consistently for actions that took us closer to the goal irrespective of how far away we stand from the goal currently. I hope this makes some sense.


## 🗺️ Project Structure

### **1. Board Class**

- **Setting Up the Maze (`__init__`)**: Creates a grid-based maze with walls and obstacles. It randomly places the start and goal positions, making sure they're not stuck.
- **Generating the Grid (`generate_grid`)**: Designs the maze layout by placing walls around the edges and sprinkling in obstacles based on a specified ratio.
- **Choosing Start and Goal (`get_random_empty_cell` & `place_start_and_goal`)**: Randomly selects empty spots for the agent to start and where the goal is, ensuring they're reachable.
- **Visualization (`visualize`)**: Draws the maze, the agent, the goal, and a colorful map showing where the agent thinks the best paths are (Q-values). It also includes legends and a handy colorbar to make everything clear.

### **2. Agent Class**

- **Getting Started (`__init__`)**: Sets up the agent with parameters like how fast it learns, how much it cares about future rewards, and how often it tries new moves.
- **Deciding Moves (`choose_action`)**: Balances between exploring new paths and sticking to known good paths based on what it has learned.
- **Learning from Moves (`update_q_value`)**: Updates its knowledge about the maze using Bellman's Equation whenever it makes a move.
- **Exploration Rate (`decay_epsilon`)**: Gradually reduces the tendency to make random moves as the agent becomes more confident in its learned paths.
- **Moving Around (`move`)**: Changes the agent's position in the maze based on the chosen action and assigns rewards or penalties.
- **Keeping Track (`get_state` & `visualize`)**: Keeps tabs on where it is and uses the Board's visualization to show its progress.

### **3. Main Function**

- **Training the Agent**: Runs lots of episodes where the agent tries to find the goal, learns from its successes and mistakes, and gets better over time.
- **Testing the Agent**: After training, lets the agent navigate the maze without any randomness to see how well it has learned the best path.

## 🚀 Getting Started

### **Prerequisites**

Make sure you have Python 3.x installed. You'll also need the following Python libraries:

- `matplotlib`
- `numpy`

You can install them using pip:

```bash
pip install matplotlib numpy
```
### **Run the Program**

If you want to run the program, simply execute the `board_game.py` file and pass along the parameters: `width`, `height`, and `obstacle_ratio`.

I haven't scaled the game much to account for massively different sizes, but if you just want to visualize things, run:

```bash
python3 board_game.py 40 40 0.2
