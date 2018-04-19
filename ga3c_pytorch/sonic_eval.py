from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
from utils import read_config, setup_logger
from model import A3Clstm
from player_util import Agent
import gym
import logging
#from gym.configuration import undo_logger_setup

from sonic_util import make_local_env, AllowBacktracking


#undo_logger_setup()
parser = argparse.ArgumentParser(description='A3C_EVAL')
parser.add_argument(
    '--env',
    default='Sonic',
    metavar='ENV',
    help='environment to train on (default: Sonic)')
parser.add_argument(
    '--env-config',
    default='sonic_config.json',
    metavar='EC',
    help='environment to crop and resize info (default: sonic_config.json)')
parser.add_argument(
    '--num-episodes',
    type=int,
    default=100,
    metavar='NE',
    help='how many episodes in evaluation (default: 100)')
parser.add_argument(
    '--load-model-dir',
    default='trained_models/',
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--log-dir',
    default='logs/',
    metavar='LG',
    help='folder to save logs')
parser.add_argument(
    '--render',
    default=True,
    metavar='R',
    help='Watch game as it being played')
parser.add_argument(
    '--render-freq',
    type=int,
    default=1,
    metavar='RF',
    help='Frequency to watch rendered game play')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=20000,
    metavar='M',
    help='maximum length of an episode (default: 100000)')
parser.add_argument(
    '--gpu-id',
    type=int,
    default=-1,
    help='GPU to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--skip-rate',
    type=int,
    default=0,
    metavar='SR',
    help='frame skip rate (default: 0)')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--new-gym-eval',
    default=False,
    metavar='NGE',
    help='Create a gym evaluation for upload')
args = parser.parse_args()

setup_json = read_config(args.env_config)
env_conf = setup_json["Test"]

gpu_id = args.gpu_id

torch.manual_seed(args.seed)
if gpu_id >= 0:
    torch.cuda.manual_seed(args.seed)

saved_state = torch.load(
    '{0}{1}.dat'.format(args.load_model_dir, args.env),
    map_location=lambda storage, loc: storage)

log = {}
setup_logger('{}_mon_log'.format(args.env), r'{0}{1}_mon_log'.format(
    args.log_dir, args.env))
log['{}_mon_log'.format(args.env)] = logging.getLogger(
    '{}_mon_log'.format(args.env))

d_args = vars(args)
for k in d_args.keys():
    log['{}_mon_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

env = make_local_env(env_conf["game"], env_conf["level"], stack=False, scale_rew=False)
num_tests = 0
reward_total_sum = 0
player = Agent(None, env, args, None)
player.model = A3Clstm(player.env.observation_space.shape[0],
                       player.env.action_space)
player.gpu_id = gpu_id
if gpu_id >= 0:
    with torch.cuda.device(gpu_id):
        player.model = player.model.cuda()
if args.new_gym_eval:
    player.env = gym.wrappers.Monitor(
        player.env, "{}_monitor".format(args.env), force=True)

if gpu_id >= 0:
    with torch.cuda.device(gpu_id):
        player.model.load_state_dict(saved_state)
else:
    player.model.load_state_dict(saved_state)

player.model.eval()
for i_episode in range(args.num_episodes):
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
    player.eps_len += 2
    reward_sum = 0
    while True:
        if args.render:
            if i_episode % args.render_freq == 0:
                player.env.render()

        player.action_test()
        reward_sum += player.reward

        if player.done or player.max_length:
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['{}_mon_log'.format(args.env)].info(
                "reward sum: {0}, reward mean: {1:.4f}".format(
                    reward_sum, reward_mean))
            player.eps_len = 0
            break
