# Please do not make changes to this file - it will be overwritten with a clean
# version when your work is marked.
#
# This file contains code for the racetrack environment that you will be using
# as part of the second part of the CM50270: Reinforcement Learning coursework.
#
# Code Author: Joshua Evans (jbe25@bath.ac.uk)

import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from IPython.display import clear_output

from matplotlib import colors

class RacetrackEnv(object) :
    """
    Class representing a race-track environment inspired by exercise 5.12 in Sutton & Barto 2018 (p.111).
    Please do not make changes to this class - it will be overwritten with a clean version when it comes to marking.

    The dynamics of this environment are detailed in this coursework exercise's jupyter notebook, although I have
    included rather verbose comments here  for those of you who are interested in how the environment has been
    implemented (though this should not impact your solution code).

    If you find any *bugs* with this code, please let me know immediately - thank you for finding them, sorry that I didn't!
    However, please do not suggest optimisations - some things have been purposely simplified for readability's sake.
    """


    ACTIONS_DICT = {
        0 : (1, -1),  # Acc Vert., Brake Horiz.
        1 : (1, 0),   # Acc Vert., Hold Horiz.
        2 : (1, 1),   # Acc Vert., Acc Horiz.
        3 : (0, -1),  # Hold Vert., Brake Horiz.
        4 : (0, 0),   # Hold Vert., Hold Horiz.
        5 : (0, 1),   # Hold Vert., Acc Horiz.
        6 : (-1, -1), # Brake Vert., Brake Horiz.
        7 : (-1, 0),  # Brake Vert., Hold Horiz.
        8 : (-1, 1)   # Brake Vert., Acc Horiz.
    }


    CELL_TYPES_DICT = {
        0 : "track",
        1 : "wall",
        2 : "start",
        3 : "goal"
    }


    def __init__(self) :
        # Load racetrack map from file.
        self.track = np.flip(np.loadtxt("./track.txt", dtype = int), axis = 0)


        # Discover start grid squares.
        self.initial_states = []
        for y in range(self.track.shape[0]) :
            for x in range(self.track.shape[1]) :
                if (self.CELL_TYPES_DICT[self.track[y, x]] == "start") :
                    self.initial_states.append((y, x))


        self.is_reset = False

        #print("Racetrack Environment File Loaded Successfully.")
        #print("Be sure to call .reset() before starting to initialise the environment and get an initial state!")


    def step(self, action : int) :
        """
        Takes a given action in the environment's current state, and returns a next state,
        reward, and whether the next state is terminal or not.

        Arguments:
            action {int} -- The action to take in the environment's current state. Should be an integer in the range [0-8].

        Raises:
            RuntimeError: Raised when the environment needs resetting.\n
            TypeError: Raised when an action of an invalid type is given.\n
            ValueError: Raised when an action outside the range [0-8] is given.\n

        Returns:
            A tuple of:\n
                {(int, int, int, int)} -- The next state, a tuple of (y_pos, x_pos, y_velocity, x_velocity).\n
                {int} -- The reward earned by taking the given action in the current environment state.\n
                {bool} -- Whether the environment's next state is terminal or not.\n

        """

        # Check whether a reset is needed.
        if (not self.is_reset) :
            raise RuntimeError(".step() has been called when .reset() is needed.\n" +
                               "You need to call .reset() before using .step() for the first time, and after an episode ends.\n" +
                               ".reset() initialises the environment at the start of an episode, then returns an initial state.")

        # Check that action is the correct type (either a python integer or a numpy integer).
        if (not (isinstance(action, int) or isinstance(action, np.integer))) :
            raise TypeError("action should be an integer.\n" +
                            "action value {} of type {} was supplied.".format(action, type(action)))

        # Check that action is an allowed value.
        if (action < 0 or action > 8) :
            raise ValueError("action must be an integer in the range [0-8] corresponding to one of the legal actions.\n" +
                             "action value {} was supplied.".format(action))


        # Update Velocity.
        # With probability, 0.85 update velocity components as intended.
        if (np.random.uniform() < 0.8) :
            (d_y, d_x) = self.ACTIONS_DICT[action]
        # With probability, 0.15 Do not change velocity components.
        else :
            (d_y, d_x) = (0, 0)

        self.velocity = (self.velocity[0] + d_y, self.velocity[1] + d_x)

		# Keep velocity within bounds (-10, 10).
        if (self.velocity[0] > 10) :
            self.velocity = (10, self.velocity[1])
        elif (self.velocity[0] < -10) :
            self.velocity = (-10, self.velocity[1])
        if (self.velocity[1] > 10) :
            self.velocity = (self.velocity[0], 10)
        elif (self.velocity[1] < -10) :
            self.velocity = (self.velocity[0], -10)

        # Update Position.
        new_position = (self.position[0] + self.velocity[0], self.position[1] + self.velocity[1])

        reward = 0
        terminal = False

        # If position is out-of-bounds, return to start and set velocity components to zero.
        if (new_position[0] < 0 or new_position[1] < 0 or new_position[0] >= self.track.shape[0] or new_position[1] >= self.track.shape[1]) :
            self.position = random.choice(self.initial_states)
            self.velocity = (0, 0)
            reward -= 10
        # If position is in a wall grid-square, return to start and set velocity components to zero.
        elif (self.CELL_TYPES_DICT[self.track[new_position]] == "wall") :
            self.position = random.choice(self.initial_states)
            self.velocity = (0, 0)
            reward -= 10
        # If position is in a track grid-squre or a start-square, update position.
        elif (self.CELL_TYPES_DICT[self.track[new_position]] in ["track", "start"]) :
            self.position = new_position
        # If position is in a goal grid-square, end episode.
        elif (self.CELL_TYPES_DICT[self.track[new_position]] == "goal") :
            self.position = new_position
            reward += 10
            terminal = True
        # If this gets reached, then the student has touched something they shouldn't have. Naughty!
        else :
            raise RuntimeError("Please don't modify things you shouldn't.")

        # Penalise every timestep.
        reward -= 1

        # Require a reset if the current state is terminal.
        if (terminal) :
            self.is_reset = False

        # Return next state, reward, and whether the episode has ended.
        return (self.position[0], self.position[1], self.velocity[0], self.velocity[1]), reward, terminal


    def reset(self) :
        """
        Resets the environment, ready for a new episode to begin, then returns an initial state.
        The initial state will be a starting grid square randomly chosen using a uniform distribution,
        with both components of the velocity being zero.

        Returns:
            {(int, int, int, int)} -- an initial state, a tuple of (y_pos, x_pos, y_velocity, x_velocity).
        """

        # Pick random starting grid-square.
        self.position = random.choice(self.initial_states)

        # Set both velocity components to zero.
        self.velocity = (0, 0)

        self.is_reset = True

        return (self.position[0], self.position[1], self.velocity[0], self.velocity[1])


    def render(self, sleep_time : float = 0.1) :
        """
        Renders a pretty matplotlib plot representing the current state of the environment.
        Calling this method on subsequent timesteps will update the plot.
        This is VERY VERY SLOW and wil slow down training a lot. Only use for debugging/testing.

        Arguments:
            sleep_time {float} -- How many seconds (or partial seconds) you want to wait on this rendered frame.

        """
        # Turn interactive mode on.
        plt.ion()
        fig = plt.figure(num = "env_render")
        ax = plt.gca()
        ax.clear()
        clear_output(wait = True)

        # Prepare the environment plot and mark the car's position.
        env_plot = np.copy(self.track)
        env_plot[self.position] = 4
        env_plot = np.flip(env_plot, axis = 0)

        # Plot the gridworld.
        cmap = colors.ListedColormap(["white", "black", "green", "red", "yellow"])
        bounds = list(range(6))
        norm = colors.BoundaryNorm(bounds, cmap.N)
        ax.imshow(env_plot, cmap = cmap, norm = norm, zorder = 0)

        # Plot the velocity.
        if (not self.velocity == (0, 0)) :
            ax.arrow(self.position[1], self.track.shape[0] - 1 - self.position[0], self.velocity[1], -self.velocity[0],
                     path_effects=[pe.Stroke(linewidth=1, foreground='black')], color = "yellow", width = 0.1, length_includes_head = True, zorder = 2)

        # Set up axes.
        ax.grid(which = 'major', axis = 'both', linestyle = '-', color = 'k', linewidth = 2, zorder = 1)
        ax.set_xticks(np.arange(-0.5, self.track.shape[1] , 1));
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(-0.5, self.track.shape[0], 1));
        ax.set_yticklabels([])

        # Draw everything.
        #fig.canvas.draw()
        #fig.canvas.flush_events()

        plt.show()

        # Sleep if desired.
        if (sleep_time > 0) :
            time.sleep(sleep_time)


    def get_actions(self) :
        """
        Returns the available actions in the current state - will always be a list
        of integers in the range [0-8].
        """
        return [*self.ACTIONS_DICT]

def plot_results(mc_rewards = None, sarsa_rewards = None, q_learning_rewards = None) :
	num_episodes = 150
	
	plt.figure(figsize=(10.666, 6))
	
	# MC Control
	if (mc_rewards is not None):
		
		# Average Returns Lists
		num_agents = len(mc_rewards)
		mc_average_episode_rewards = []
		for episode in range(0, num_episodes) :
			reward = 0
			for agent in range(0, num_agents) :
				reward += mc_rewards[agent][episode]
			mc_average_episode_rewards.append(reward / num_agents)
		
		
		# Load example returns for comparison.
		with open("correct_returns_mc.json", "r") as f:
			example_mc_rewards = json.load(f)
			
		# Average Example Returns Lists
		example_num_agents = len(example_mc_rewards)
		example_mc_average_episode_rewards = []
		for episode in range(0, num_episodes) :
			reward = 0
			for agent in range(0, example_num_agents) :
				reward += example_mc_rewards[agent][episode]
			example_mc_average_episode_rewards.append(reward / example_num_agents) 
		
		# Plot unsmoothed learning curve.
		plt.plot(range(num_episodes), mc_average_episode_rewards, label = "MC Agent", c = plt.cm.tab20c(0))
		plt.plot(range(num_episodes), example_mc_average_episode_rewards, label = "Example MC Agent", c = plt.cm.tab20c(4), alpha = 0.75)
		plt.title("Racetrack Average Learning Curve\n{} MC Control Agents' Performance Averaged".format(num_agents))
		plt.xlabel("Episodes of Training")
		plt.ylabel("Average Undiscounted Return")
		plt.legend()
		plt.grid()
		plt.show()
		
	# Sarsa
	if (sarsa_rewards is not None):
		num_agents = len(sarsa_rewards)
		
		# Average Returns Lists
		sarsa_average_episode_rewards = []
		for episode in range(0, num_episodes) :
			reward = 0
			for agent in range(0, num_agents) :
				reward += sarsa_rewards[agent][episode]
			sarsa_average_episode_rewards.append(reward / num_agents)
			
		# Load example returns for comparison.
		with open("correct_returns_sarsa.json", "r") as f:
			example_sarsa_rewards = json.load(f)
			
		# Average Example Returns Lists
		example_num_agents = len(example_sarsa_rewards)
		example_sarsa_average_episode_rewards = []
		for episode in range(0, num_episodes) :
			reward = 0
			for agent in range(0, example_num_agents) :
				reward += example_sarsa_rewards[agent][episode]
			example_sarsa_average_episode_rewards.append(reward / example_num_agents)
			
		# Plot unsmoothed learning curve.    
		plt.plot(range(num_episodes), sarsa_average_episode_rewards, label = "Sarsa Agent", c = plt.cm.tab20c(0))
		plt.plot(range(num_episodes), example_sarsa_average_episode_rewards, label = "Example Sarsa Agent", c = plt.cm.tab20c(4), alpha = 0.75)
		plt.title("Racetrack Average Learning Curve\n{} Sarsa Agents' Performance Averaged".format(num_agents))
		plt.xlabel("Episodes of Training")
		plt.ylabel("Average Undiscounted Return")
		plt.legend()
		plt.grid()
		plt.show()
		
	# Q-Learning
	if (q_learning_rewards is not None):
		num_agents = len(q_learning_rewards)
		
		# Average Returns Lists
		q_learning_average_episode_rewards = []
		for episode in range(0, num_episodes) :
			reward = 0
			for agent in range(0, num_agents) :
				reward += q_learning_rewards[agent][episode]
			q_learning_average_episode_rewards.append(reward / num_agents)
		
		# Load example returns for comparison.
		with open("correct_returns_q.json", "r") as f:
			example_q_rewards = json.load(f)
			
		# Average Example Returns Lists
		example_num_agents = len(example_q_rewards)
		example_q_average_episode_rewards = []
		for episode in range(0, num_episodes) :
			reward = 0
			for agent in range(0, example_num_agents) :
				reward += example_q_rewards[agent][episode]
			example_q_average_episode_rewards.append(reward / example_num_agents)
		
		# Plot unsmoothed learning curve.    
		plt.plot(range(num_episodes), q_learning_average_episode_rewards, label = "Q-Learning Agent", c = plt.cm.tab20c(0))
		plt.plot(range(num_episodes), example_q_average_episode_rewards, label = "Example Q-Learning Agent", c = plt.cm.tab20c(4), alpha = 0.75)
		plt.title("Racetrack Average Learning Curve\n{} Q-Learning Agents' Performance Averaged".format(num_agents))
		plt.xlabel("Episodes of Training")
		plt.ylabel("Average Undiscounted Return")
		plt.legend()
		plt.grid()
		plt.show()
		
def plot_combined_results() :
	num_episodes = 150
	
	# MC Control
	with open("correct_returns_mc.json", "r") as f1:
		mc_rewards = json.load(f1)
	if (mc_rewards is not None):
		num_agents = len(mc_rewards)
		
		# Average Returns Lists
		mc_average_episode_rewards = []
		for episode in range(0, num_episodes) :
			reward = 0
			for agent in range(0, num_agents) :
				reward += mc_rewards[agent][episode]
			mc_average_episode_rewards.append(reward / num_agents)
			
	# Sarsa
	with open("correct_returns_sarsa.json", "r") as f2:
		sarsa_rewards = json.load(f2)
	if (sarsa_rewards is not None):
		num_agents = len(sarsa_rewards)
		
		# Average Returns Lists
		sarsa_average_episode_rewards = []
		for episode in range(0, num_episodes) :
			reward = 0
			for agent in range(0, num_agents) :
				reward += sarsa_rewards[agent][episode]
			sarsa_average_episode_rewards.append(reward / num_agents)
			
	# Q-Learning
	with open("correct_returns_q.json", "r") as f3:
		q_learning_rewards = json.load(f3)
	if (q_learning_rewards is not None):
		num_agents = len(q_learning_rewards)
		
		# Average Returns Lists
		q_learning_average_episode_rewards = []
		for episode in range(0, num_episodes) :
			reward = 0
			for agent in range(0, num_agents) :
				reward += q_learning_rewards[agent][episode]
			q_learning_average_episode_rewards.append(reward / num_agents)
			
	
	# Uncropped Learning Curves
	plt.figure(figsize=(10.666, 6))
	plt.plot(range(num_episodes), mc_average_episode_rewards, label = "MC Control")
	plt.plot(range(num_episodes), sarsa_average_episode_rewards, label = "Sarsa")
	plt.plot(range(num_episodes), q_learning_average_episode_rewards, label = "Q-Learning")
	plt.title("Racetrack Average Learning Curve")
	plt.xlabel("Episodes")
	plt.ylabel("Average Return")
	plt.legend()
	plt.grid()
	plt.show()
	
	# Cropped Learning Curves
	plt.figure(figsize=(10.666, 6))
	plt.plot(range(num_episodes), mc_average_episode_rewards, label = "MC Control")
	plt.plot(range(num_episodes), sarsa_average_episode_rewards, label = "Sarsa")
	plt.plot(range(num_episodes), q_learning_average_episode_rewards, label = "Q-Learning")
	plt.title("Racetrack Average Learning Curve (Cropped)")
	plt.xlabel("Episodes")
	plt.ylabel("Average Return")
	plt.ylim((-500, 0))
	plt.legend()
	plt.grid()
	plt.show()
	
def plot_modified_agent_results(modified_agent_rewards = None) :
	num_episodes = 150
			
	# Q-Learning
	with open("correct_returns_q.json", "r") as f:
		q_learning_rewards = json.load(f)
	if (q_learning_rewards is not None):
		num_agents = len(q_learning_rewards)
		
		# Average Returns Lists
		q_learning_average_episode_rewards = []
		for episode in range(0, num_episodes) :
			reward = 0
			for agent in range(0, num_agents) :
				reward += q_learning_rewards[agent][episode]
			q_learning_average_episode_rewards.append(reward / num_agents)
			
	# Modified Agent
	if (modified_agent_rewards is not None):
		num_agents_mod = len(modified_agent_rewards)
		
		# Average Returns Lists
		modified_agent_average_episode_rewards = []
		for episode in range(0, num_episodes) :
			reward = 0
			for agent in range(0, num_agents_mod) :
				reward += modified_agent_rewards[agent][episode]
			modified_agent_average_episode_rewards.append(reward / num_agents_mod)
			
	
	# Uncropped Learning Curves
	plt.figure(figsize=(10.666, 6))
	plt.plot(range(num_episodes), q_learning_average_episode_rewards, label = "Q-Learning")
	plt.plot(range(num_episodes), modified_agent_average_episode_rewards, label = "Modified Agent")
	plt.title("Racetrack Average Learning Curve")
	plt.xlabel("Episodes Played")
	plt.ylabel("Average Return")
	plt.legend()
	plt.grid()
	plt.show()
	
	# Cropped Learning Curves
	plt.figure(figsize=(10.666, 6))
	plt.plot(range(num_episodes), q_learning_average_episode_rewards, label = "Q-Learning")
	plt.plot(range(num_episodes), modified_agent_average_episode_rewards, label = "Modified Agent")
	plt.title("Racetrack Average Learning Curve (Cropped)")
	plt.xlabel("Episodes Played")
	plt.ylabel("Average Return")
	plt.ylim((-500, 0))
	plt.legend()
	plt.grid()
	plt.show()

def simple_issue_checking(results, modified_agent = False) :
	# Check that at least 20 agents have been trained.
	if (not modified_agent) :
		if (len(results) < 20) :
			print("[Marking Advice] Performance only averaged across {} agents. Please train at least 20 agents.".format(len(results)))
	else :
		if (len(results) < 2) :
			print("[Marking Advice] Performance only averaged across {} agents. Please train at least 2 agents.".format(len(results)))
	
	# Check that all agents have been trained for 150 episodes.
	for i in range(len(results)) :			
		if (len(results[i]) != 150) :
			print("[Marking Advice] Agent {} was trained for {} episodes. Please train each of your agents for exactly 150 episodes.".format(i, len(results[i])))

# num_steps = 1000000

# env = RacetrackEnv()
# state = env.reset()
# print(state)

# for _ in range(num_steps) :

#     next_state, reward, terminal = env.step(random.choice(env.get_actions()))
#     print(next_state)
#     env.render()

#     if (terminal) :
#         _ = env.reset()
