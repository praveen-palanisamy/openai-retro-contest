#!/usr/bin/env python

from sonic_util import make_local_env
from argparse import ArgumentParser
import os
import torch
import json

parser = ArgumentParser(prog="test_sonic", description="Test a trained model on the Sonic retro gym env")
parser.add_argument("--model-path", default="./trained_models/ppo/Sonic-Genesis-mixed-Train_mean1500_max6k.pt",
                    help="Path to the pytorch agent model file", metavar="MODELPATH")
parser.add_argument("--env-config", default="sonic_config.json",
                    help="Path to the env config json file", metavar="ENVCONFIGFILE")
args = parser.parse_args()

if os.path.exists(args.model_path):
    agent_policy, obs = torch.load(args.model_path)

env_confs = json.load(open(args.env_config, 'r'))
test_env_conf = env_confs['Test']
test_envs = [v for _, v in test_env_conf.items()]
print("test_envs:", test_envs)

# Step 1: Test the agent against 1 env
# Step 2: Test the agent against all the test env
test_env = test_envs[0]
env = make_local_env(game=test_env['game'], state=test_env['level'])
obs = env.reset()
env.render('human')
