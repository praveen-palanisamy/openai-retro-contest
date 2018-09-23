#!/usr/bin/env python

"""
Train an agent on Sonic using PPO2 from OpenAI Baselines.
"""

import tensorflow as tf

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import baselines.ppo2.ppo2 as ppo2
import baselines.ppo2.policies as policies
import gym_remote.exceptions as gre

from sonic_util import make_env, make_local_env, AllowBacktracking
from argparse import ArgumentParser
arg_parser = ArgumentParser("ppo2")
arg_parser.add_argument("--local", action="store_true",
                        help="Use local env and not gym remote env")
args = arg_parser.parse_args()
def main():
    """Run PPO until the environment throws an exception."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config):
        # Take more timesteps than we need to be sure that
        # we stop due to an exception.
        env_maker = make_local_env if args.local else make_env

        ppo2.learn(policy=policies.CnnPolicy,
                   env=DummyVecEnv([env_maker]),
                   nsteps=4096,
                   nminibatches=8,
                   lam=0.95,
                   gamma=0.99,
                   noptepochs=5,
                   log_interval=1,
                   ent_coef=0.01,
                   lr=lambda _: 2e-4,
                   cliprange=lambda _: 0.1,
                   total_timesteps=int(5e7))

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
