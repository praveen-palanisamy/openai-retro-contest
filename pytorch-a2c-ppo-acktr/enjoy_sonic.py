import argparse
import os
import types

import numpy as np
import torch

from sonic_util import make_local_env
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-stack', type=int, default=4,
                    help='number of frames to stack (default: 4)')
parser.add_argument('--model-path', default='./trained_models/ppo/Sonic-GHZA1.pt',
                    help='Path to the agent Policy to be loaded (default: ./trained_models/ppo/Sonic-GHZA1.pt)')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
parser.add_argument('--num-episodes', type=int, default=100,
                    help="Number of episodes to test/run the agent for")
parser.add_argument('--log-dir', type=str, default='logs',
                    help='Log directory to store the tensorboard summary files')

#summary_file_path_prefix =
writer = SummaryWriter()

args = parser.parse_args()

env = make_local_env(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1',stack=False, scale_rew =False)

actor_critic, ob_rms, saved_rew = torch.load(args.model_path)
print("Loaded Policy that got a mean reward of:", saved_rew)

render_func = env.render

obs_shape = env.observation_space.shape
obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
current_obs = torch.zeros(1, *obs_shape)
states = torch.zeros(1, actor_critic.state_size)
masks = torch.zeros(1, 1)


def update_current_obs(obs):
    shape_dim0 = env.observation_space.shape[0]
    obs = torch.from_numpy(obs).float()
    if args.num_stack > 1:
        current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
    current_obs[:, -shape_dim0:] = obs



episode = 0
while episode < args.num_episodes:
    obs = env.reset()
    update_current_obs(obs)
    render_func('human')
    done = False
    cum_reward = 0
    episode += 1
    while not done:

        with torch.no_grad():
            value, action, _, states = actor_critic.act(current_obs,
                                                        states,
                                                        masks,
                                                        deterministic=True)
        cpu_actions = action.squeeze(1).cpu().numpy()[0]
        obs, reward, done, _ = env.step(cpu_actions)
        cum_reward += reward


        if current_obs.dim() == 4:
            current_obs *= masks.unsqueeze(2).unsqueeze(2)
        else:
            current_obs *= masks
        update_current_obs(obs)

        render_func('human')

        print("Episode#", episode, "cum_reward:", cum_reward, end='\r')
    print("\n")
