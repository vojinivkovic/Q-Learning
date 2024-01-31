import inspect
import random
import sys
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn

import pygame.surfarray

import config
from artifacts import Agent, Goal, Food, Water
from image import Image
from tiles import TilesFactory, Hole, Grass


class Quit(Exception):
    pass


class Action(Enum):
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3


class Environment:
    def __init__(self, path):
        self.field_map = []
        self.artifacts_map = {obj.kind(): obj()
                              for name, obj in sorted(inspect.getmembers(sys.modules['artifacts']), reverse=True)
                              if inspect.isclass(obj) and name != 'Artifact'}

        with open(path, 'r') as file:
            for i, line in enumerate(file):
                self.field_map.append([])
                for j, c in enumerate(line.strip()):
                    self.field_map[-1].append(TilesFactory.generate_tile(c
                                                                         if c not in self.artifacts_map.keys()
                                                                         else Grass.kind(), [i, j]))
                    if c in self.artifacts_map.keys():
                        self.artifacts_map[c].set_position([i, j])
        self.all_actions = [act for act in Action]
        for key in self.artifacts_map:
            if self.artifacts_map[key].get_position() is None and key in {Agent.kind(), Goal.kind()}:
                raise Exception(f'Environment map is missing agent or goal!')
            if key == Agent.kind():
                self.agent_start_position = self.artifacts_map[key].get_position().copy()
        self.display = None
        self.clock = None
        self.q_tab = np.zeros((len(self.field_map) * len(self.field_map[0]), len(self.all_actions)))
        self.q_water = np.zeros((len(self.field_map) * len(self.field_map[0]), len(self.all_actions)))
        self.q_food = np.zeros((len(self.field_map) * len(self.field_map[0]), len(self.all_actions)))



    def __del__(self):
        if self.display:
            pygame.quit()

    def reset(self):
        self.artifacts_map[Agent.kind()].set_position(self.agent_start_position.copy())

    def get_agent_position(self):
        return self.artifacts_map[Agent.kind()].get_position()

    def get_goal_position(self):
        return self.artifacts_map[Goal.kind()].get_position()

    def get_artifact_position(self, kind):
        return self.artifacts_map[kind].get_position()

    def get_all_actions(self):
        return self.all_actions

    def get_random_action(self):
        return self.all_actions[random.randint(0, len(self.all_actions) - 1)]

    def get_field_map(self):
        return self.field_map

    def render_textual(self):
        """
            Rendering textual representation of the current state of the environment.
        """
        text = ''.join([''.join([t.kind() for t in row]) for row in self.field_map])
        gp = (len(self.field_map[0]) * self.artifacts_map[Goal.kind()].get_position()[0] +
              self.artifacts_map[Goal.kind()].get_position()[1])
        text = text[:gp] + Goal.kind() + text[gp + 1:]
        ap = (len(self.field_map[0]) * self.artifacts_map[Agent.kind()].get_position()[0] +
              self.artifacts_map[Agent.kind()].get_position()[1])
        text = text[:ap] + Agent.kind() + text[ap + 1:]
        cols = len(self.field_map[0])
        print('\n'.join(text[i:i + cols] for i in range(0, len(text), cols)))

    def render(self, fps):
        """
            Rendering current state of the environment in FPS (frames per second).
        """
        if not self.display:
            pygame.init()
            pygame.display.set_caption('Pyppy Adventure')
            self.display = pygame.display.set_mode((len(self.field_map[0]) * config.TILE_SIZE,
                                                    len(self.field_map) * config.TILE_SIZE))
            self.clock = pygame.time.Clock()
        for i in range(len(self.field_map)):
            for j in range(len(self.field_map[0])):
                self.display.blit(Image.get_image(self.field_map[i][j].image_path(),
                                                  (config.TILE_SIZE, config.TILE_SIZE)),
                                  (j * config.TILE_SIZE, i * config.TILE_SIZE))
        for a in self.artifacts_map:
            self.display.blit(Image.get_image(self.artifacts_map[a].image_path(),
                                              (config.TILE_SIZE, config.TILE_SIZE), config.GREEN),
                              (self.artifacts_map[self.artifacts_map[a].kind()].get_position()[1] * config.TILE_SIZE,
                               self.artifacts_map[self.artifacts_map[a].kind()].get_position()[0] * config.TILE_SIZE))
        pygame.display.flip()
        self.clock.tick(fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                raise Quit

    def step(self, action):
        """
            Actions: UP(0), LEFT(1), DOWN(2), RIGHT(3)
            Returns: new_state, reward, done
                new_state - agent position in new state obtained by applying chosen action in the current state.
                reward - immediate reward awarded for transitioning to the new state.
                done - boolean flag specifying whether the new state is terminal state.
        """
        if action not in self.all_actions:
            raise Exception(f'Illegal action {action}! Legal actions: {self.all_actions}.')
        if action == Action.UP and self.artifacts_map[Agent.kind()].get_position()[0] > 0:
            self.artifacts_map[Agent.kind()].get_position()[0] -= 1
        elif action == Action.LEFT and self.artifacts_map[Agent.kind()].get_position()[1] > 0:
            self.artifacts_map[Agent.kind()].get_position()[1] -= 1
        elif action == Action.DOWN and self.artifacts_map[Agent.kind()].get_position()[0] < len(self.field_map) - 1:
            self.artifacts_map[Agent.kind()].get_position()[0] += 1
        elif action == Action.RIGHT and self.artifacts_map[Agent.kind()].get_position()[1] < len(self.field_map[0]) - 1:
            self.artifacts_map[Agent.kind()].get_position()[1] += 1

        agent_position = self.artifacts_map[Agent.kind()].get_position()
        reward = self.field_map[agent_position[0]][agent_position[1]].reward()
        done = (agent_position == self.artifacts_map[Goal.kind()].get_position() or
                self.field_map[agent_position[0]][agent_position[1]].kind() == Hole.kind())
        return agent_position, reward, done

    def get_action_eps_greedy_policy(self, q_tab, state, eps):
        prob = random.uniform(0, 1)
        return self.all_actions[np.argmax(q_tab[state])] if prob > eps else self.get_random_action()

    def train(self, num_episodes=7000, max_steps=100, lr=0.05, gamma=0.95, eps_min=0.005, eps_max=1.0, eps_dec_rate=0.001):
        self.avg_returns = []
        self.avg_steps = []
        for episode in range(num_episodes):
            self.avg_returns.append(0.)
            self.avg_steps.append(0)
            eps = eps_min + (eps_max - eps_min) * np.exp(-eps_dec_rate * episode)
            self.reset()
            state = 0
            for step in range(max_steps):
                action = self.get_action_eps_greedy_policy(self.q_tab, state, eps)
                agent_position, reward, done = self.step(action)
                self.avg_returns[-1] += reward
                new_state = len(self.field_map[0]) * agent_position[0] + agent_position[1]
                self.q_tab[state][action.value] = self.q_tab[state][action.value] + lr * (reward + gamma * np.max(self.q_tab[new_state]) - self.q_tab[state][action.value])
                if done:
                    self.avg_steps[-1] += step + 1
                    break
                state = new_state
        return self.q_tab, self.avg_returns, self.avg_steps

    def train_food(self, num_episodes=7000, max_steps=100, lr=0.05, gamma=0.95, eps_min=0.005, eps_max=1.0, eps_dec_rate=0.001):
        for episode in range(num_episodes):
            eps = eps_min + (eps_max - eps_min) * np.exp(-eps_dec_rate * episode)
            self.reset()
            state = 0
            for step in range(max_steps):
                action = self.get_action_eps_greedy_policy(self.q_food, state, eps)
                agent_position, reward, done = self.step(action)
                new_state = len(self.field_map[0]) * agent_position[0] + agent_position[1]
                self.q_food[state][action.value] = self.q_food[state][action.value] + lr * (reward + gamma * np.max(self.q_food[new_state]) - self.q_food[state][action.value])
                if agent_position == self.artifacts_map[Food.kind()].get_position():
                    break
                state = new_state
        return self.q_food

    def train_water(self, num_episodes=7000, max_steps=100, lr=0.05, gamma=0.95, eps_min=0.005, eps_max=1.0, eps_dec_rate=0.001):
        place_food = self.artifacts_map[Food.kind()].get_position()
        x_pos = place_food[0]
        y_pos = place_food[1]
        for episode in range(num_episodes):
            eps = eps_min + (eps_max - eps_min) * np.exp(-eps_dec_rate * episode)
            state = x_pos * len(self.field_map[0]) + y_pos
            self.artifacts_map[Agent.kind()].set_position([x_pos, y_pos])
            for step in range(max_steps):
                action = self.get_action_eps_greedy_policy(self.q_water, state, eps)
                agent_position, reward, done = self.step(action)
                new_state = len(self.field_map[0]) * agent_position[0] + agent_position[1]
                self.q_water[state][action.value] = self.q_water[state][action.value] + lr * (
                            reward + gamma * np.max(self.q_water[new_state]) - self.q_water[state][action.value])
                if agent_position == self.artifacts_map[Water.kind()].get_position():
                    break
                state = new_state
        return self.q_water


    def line_plot(self, data, name, show):
        plt.figure(f"Average {name} per episode: {np.mean(data):.2f}")
        df = pd.DataFrame({
            name: [np.mean(data[i * config.chunk:(i+1) * config.chunk])
                   for i in range(config.number_of_episodes // config.chunk)],
            "episode": [config.chunk * i for i in range(config.number_of_episodes // config.chunk)]})
        plot = seaborn.lineplot(data=df, x="episode", y=name, markers='o',
                                markersize=5, markerfacecolor='red')
        plot.get_figure().savefig(f'{name}.png')
        if show:
            plt.show()












