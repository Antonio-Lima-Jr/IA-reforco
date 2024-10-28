from collections import deque
import random

import torch
import torch.optim as optim
import torch.nn as nn
from math import degrees


BATCH_SIZE = 2000
GAMMA = 0.95

class DQNAgent:
    def __init__(self, model: nn.Module, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters())
        self.memory = deque(maxlen=2000)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def select_action(self, state, actions_possible):
        action = 0
        if random.random() < self.epsilon:
            action = random.randint(0, actions_possible - 1)
        else:
            with torch.no_grad():
                action = self.model(torch.FloatTensor(state)).argmax().item()
        return action

    def train_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        state_PQ, action_PQ, reward_PQ, next_state_PQ, done_PQ = zip(*batch)

        state = torch.FloatTensor(state_PQ)
        action = torch.LongTensor(action_PQ)
        reward = torch.FloatTensor(reward_PQ)
        next_state = torch.FloatTensor(next_state_PQ)
        done = torch.FloatTensor(done_PQ)

        current_q = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q = self.model(next_state).max(1)[0]
        expected_q = reward + (GAMMA * next_q * (1 - done))

        loss = nn.MSELoss()(current_q, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return self.model.named_parameters()

    def remember(self, state, action, reward, next_state, done):
        new_reward = reward  if reward == 0 else self.calculate_reward(state[2], next_state[2], state[0], next_state[0], action)
        self.memory.append((state, action, new_reward, next_state, done))
        return new_reward

    def calculate_reward(self, before_angle, after_angle, before_position, after_position, action):
        # Absolute distances from zero for angle and position
        angle_distance_before = abs(before_angle)
        angle_distance_after = abs(after_angle)
        position_distance_before = abs(before_position)
        position_distance_after = abs(after_position)
        
        # Check if there's a change in direction for the angle
        angle_changed_direction = (before_angle * after_angle < 0)
        
        # Reward calculation for angle
        if angle_changed_direction:
            angle_reward = 1.0  # Max reward for changing direction
        elif angle_distance_after < angle_distance_before:
            angle_reward = 0.7 + 0.3 * (1 - angle_distance_after / 24)  # Higher reward if moving closer to zero
        else:
            angle_reward = 0.3 * (1 - angle_distance_after / 24)  # Lower reward if moving further away

        # Reward calculation for cart position
        if position_distance_after < position_distance_before:
            position_reward = 0.7 + 0.3 * (1 - position_distance_after / 80)  # Higher reward if moving closer to zero
        else:
            position_reward = 0.3 * (1 - position_distance_after / 80)  # Lower reward if moving further away

        # Final reward as a weighted average
        total_reward = 0.6 * angle_reward + 0.4 * position_reward
        
        return total_reward

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay