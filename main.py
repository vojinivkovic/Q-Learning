import numpy as np

import config
from environment import Environment, Quit
from artifacts import Food, Water

environment = Environment(f'maps/map.txt')


try:
    environment.render(config.FPS)
    q_tab, avg_returns, avg_steps = environment.train()
    q_tab_food = environment.train_food()
    q_tab_water = environment.train_water()
    #environment.line_plot(avg_returns, "returns", True)
    #environment.line_plot(avg_steps, "steps", True)
    environment.reset()
    print("Training finished")
    acc_reward = 0
    q_tab_use = q_tab_food
    while True:
        agent_pos = environment.get_agent_position()
        state = agent_pos[0] * len(environment.field_map[0]) + agent_pos[1]
        action = environment.all_actions[np.argmax(q_tab_use[state])]
        agent_pos, reward, done = environment.step(action)
        acc_reward += reward
        environment.render(config.FPS)
        if agent_pos == environment.artifacts_map[Food.kind()].get_position():
            q_tab_use = q_tab_water
        if agent_pos == environment.artifacts_map[Water.kind()].get_position():
            q_tab_use = q_tab
        if done:
            print(f'Accumulated reward: {acc_reward}')
            break

except Quit:
    pass
