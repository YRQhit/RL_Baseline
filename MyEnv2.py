import torch
import time
import random
import numpy as np
from gym import spaces, Env
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

class TwoDBOXGame(Env):  # 对抗
    def __init__(self):
        super(TwoDBOXGame, self).__init__()
        self.min_action = -1
        self.max_action = 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.chase = None
        self.escape = None
        self.initaldistance = None
        self.time = 1
        self.Predistance = None
        self.time_penalty_lambda = 1.0
        self.threshold = 3
        self.grid_size = 61  # Assuming a 61x61 grid for visualization

    def reset(self):
        self.time = 1
        self.initaldistance = self._initialize_positions()
        observation = self._get_observation()
        return observation

    def _initialize_positions(self):
        current_time = int(time.time())
        random.seed(current_time)
        self.chase = torch.tensor([30, 30], dtype=torch.float32)
        self.escape = torch.tensor([45, 45], dtype=torch.float32)
        self.initaldistance = (torch.abs(self.chase[0] - self.escape[0]) + torch.abs(self.chase[1] - self.escape[1]))
        self.Predistance = (torch.abs(self.chase[0] - self.escape[0]) + torch.abs(self.chase[1] - self.escape[1]))
        return self.initaldistance

    def step(self, action):
        action = self.choose_action(action)
        self.chase = self.update_state(self.chase, action)
        action = self.choose_action(random.randint(0, 4))  # Random action for escape
        self.escape = self.update_state(self.escape, action)
        observation = self._get_observation()
        reward = self.reward().item()  # Ensure reward is a scalar
        done = self.check_termination()
        self.time += 1
        return observation, reward, done, {}

    def choose_action(self, action):
        if action == 0:
            output = torch.tensor([1, 0], dtype=torch.float32)
        elif action == 1:
            output = torch.tensor([0, 1], dtype=torch.float32)
        elif action == 2:
            output = torch.tensor([0, -1], dtype=torch.float32)
        elif action == 3:
            output = torch.tensor([-1, 0], dtype=torch.float32)
        elif action == 4:
            output = torch.tensor([0, 0], dtype=torch.float32)
        else:
            raise ValueError("Invalid action: {}".format(action))
        return output

    def update_state(self, state, action):
        state = state + action
        state[0] = torch.clamp(state[0], 0, self.grid_size - 1)
        state[1] = torch.clamp(state[1], 0, self.grid_size - 1)
        return state

    def _get_observation(self):
        observation = torch.cat((self.chase, self.escape))
        return observation

    def check_termination(self):
        if torch.abs(self.escape[0] - self.chase[0]) + torch.abs(self.escape[1] - self.chase[1]) < self.threshold:
            return True
        return False

    def reward(self):
        distance = (torch.abs(self.chase[0] - self.escape[0]) + torch.abs(self.chase[1] - self.escape[1]))
        reward = 10.0 * (self.Predistance - distance)

        # 时间惩罚
        time_penalty = self.time_penalty_lambda * self.time
        reward -= time_penalty

        self.Predistance = distance

        if distance < self.threshold:
            reward += 1000  # Use += instead of = to ensure time penalty takes effect

        return reward

    # def render(self):
    #     fig, ax = plt.subplots()
    #     ax.set_xlim(0, self.grid_size)
    #     ax.set_ylim(0, self.grid_size)
    #     ax.set_aspect('equal', adjustable='box')
    #
    #     chase_pos = ax.scatter([], [], marker='o', color='blue', label='Chase')
    #     escape_pos = ax.scatter([], [], marker='x', color='red', label='Escape')
    #     ax.legend()
    #
    #     def init():
    #         chase_pos.set_offsets([[float('nan'), float('nan')]])  # Empty initialization
    #         escape_pos.set_offsets([[float('nan'), float('nan')]])  # Empty initialization
    #         return chase_pos, escape_pos,
    #
    #     def update(frame):
    #         chase_pos.set_offsets([self.chase[0].item(), self.chase[1].item()])
    #         escape_pos.set_offsets([self.escape[0].item(), self.escape[1].item()])
    #         return chase_pos, escape_pos,
    #
    #     ani = animation.FuncAnimation(fig, update, frames=10, repeat=False, blit=True)
    #     plt.show(block=False)
    #     plt.pause(0.5)
    #     plt.close(fig)

    # def render(self):
    #     fig, ax = plt.subplots()
    #     ax.set_xlim(0, self.grid_size)
    #     ax.set_ylim(0, self.grid_size)
    #     ax.set_aspect('equal', adjustable='box')
    #
    #     chase_pos = ax.scatter([], [], marker='o', color='blue', label='Chase')
    #     escape_pos = ax.scatter([], [], marker='x', color='red', label='Escape')
    #     ax.legend()
    #
    #     def init():
    #         chase_pos.set_offsets([[], []])  # Empty initialization
    #         escape_pos.set_offsets([[], []])  # Empty initialization
    #         return chase_pos, escape_pos,
    #
    #     def update(frame):
    #         ax.clear()
    #         ax.set_xlim(0, self.grid_size)
    #         ax.set_ylim(0, self.grid_size)
    #         ax.set_aspect('equal', adjustable='box')
    #         ax.scatter(self.chase[0].item(), self.chase[1].item(), marker='o', color='blue', label='Chase')
    #         ax.scatter(self.escape[0].item(), self.escape[1].item(), marker='x', color='red', label='Escape')
    #         ax.legend()
    #         return chase_pos, escape_pos,
    #
    #     ani = animation.FuncAnimation(fig, update, init_func=init, frames=10, repeat=False, blit=False)
    #     plt.show(block=False)
    #     plt.pause(0.1)
    #     plt.close(fig)

    def render(self):
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(0, self.grid_size)
            self.ax.set_ylim(0, self.grid_size)
            self.ax.set_aspect('equal', adjustable='box')
            self.chase_pos = self.ax.scatter([], [], marker='o', color='blue', label='Chase')
            self.escape_pos = self.ax.scatter([], [], marker='x', color='red', label='Escape')
            self.ax.legend()
            self.fig.show()

        self.chase_pos.set_offsets([[self.chase[0].item(), self.chase[1].item()]])
        self.escape_pos.set_offsets([[self.escape[0].item(), self.escape[1].item()]])
        self.fig.canvas.draw()
        print(self.chase)
        print(self.escape)

        self.fig.canvas.flush_events()