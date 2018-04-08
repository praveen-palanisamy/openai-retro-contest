#!/usr/bin/env python
#from retro_contest.local import make
from retro import make
from pynput.keyboard import Listener,Key
import numpy as np
from datetime import datetime

keys_pressed = set()
class GenesisKeyboard:
    def __init__(self, *keys):
        global keys_pressed
        self.combination = {*keys}
        self.is_pressed = False

        listener = Listener(on_press=self._on_press, on_release=self._on_release)
        listener.start()

    def _on_press(self, key):
        if key in self.combination:
            keys_pressed.add(key)
        #print("keys_pressed=", keys_pressed)

    def _on_release(self, key):
        try:
            keys_pressed.remove(key)

        except KeyError:
            pass


action_space = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'],
                                                ['DOWN'], ['DOWN', 'B'], ['B']]

buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT",
                                                    "C", "Y", "X", "Z"]
action_button_map = []
for action in action_space:
    arr = np.array([False] * 12)
    for button in action:
        arr[buttons.index(button)] = True
    action_button_map.append(arr)


def action_to_buttons(action_index):
    return action_button_map[action_index]


def get_action():
    keyboard_actions = list()
    if Key.down in keys_pressed: keyboard_actions.append('DOWN')
    if Key.left in keys_pressed: keyboard_actions.append('LEFT')
    if Key.right in keys_pressed: keyboard_actions.append('RIGHT')
    if Key.shift in keys_pressed: keyboard_actions.append('B')

    # 1. Convert keyboard actions to action-space of the agent
    # 2. Convert action-space of agent to buttons
    if 'B' in keyboard_actions and 'DOWN' in keyboard_actions:
        combo = ['DOWN', 'B']
        action_index = action_space.index(combo)
        return action_to_buttons(action_index)

    if 'RIGHT' in keyboard_actions and 'DOWN' in keyboard_actions:
        combo = ['RIGHT', 'DOWN']
        action_index = action_space.index(combo)
        return action_to_buttons(action_index)

    if 'LEFT' in keyboard_actions and 'DOWN' in keyboard_actions:
        combo = ['LEFT', 'DOWN']
        action_index = action_space.index(combo)
        return action_to_buttons(action_index)

    # Below are the remaining actions that are single key and not combo
    for key in [['LEFT'], ['RIGHT'], ['DOWN'], ['B']]:
        if key[0] in keyboard_actions:
            action_index = action_space.index(key)
            return action_to_buttons(action_index)

    # If no key was pressed, return a noop
    return np.array([False] * len(buttons))

def main():
    btn = GenesisKeyboard(Key.left, Key.right, Key.down, Key.shift)
    game = 'SonicTheHedgehog-Genesis'
    state = 'LabyrinthZone.Act1'
    env = make(game=game, state=state, record='.')
    summary_output = open("summary" + game + state +
                          str(datetime.now()).replace(" ", "") + ".csv")
    summary_output.write("episode_num, reward, num_steps")
    total_reward = 0.0
    num_steps = 0
    episode_num = 0
    obs = env.reset()
    env.render()
    while True:
        action = get_action()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        num_steps += 1
        env.render()
        if done:
            obs = env.reset()
            summary_data = str(episode_num) + ',' + str(reward) + ',' + \
                                         str(num_steps)
            summary_output.write(summary_data + "\n")
            total_reward = 0.0
            num_steps = 0
            episode_num += 1
    summary_output.close()

if __name__ == "__main__":
    main()

