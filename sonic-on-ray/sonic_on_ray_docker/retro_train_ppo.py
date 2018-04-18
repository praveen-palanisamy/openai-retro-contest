from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonic_on_ray
import gym_remote.exceptions as gre

import ray
from ray.rllib import ppo
from ray.tune.registry import register_env

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-checkpoint-dir', help='Checkpoint dir', default=None)
    parser.add_argument('--load-checkpoint', help='Path to existing checkpoint created by _save', default=None)
    parser.add_argument('--local', help='Use retro_contest.local.make')
    args = parser.parse_args()

    env_name = 'sonic_env'
    # Note that the hyperparameters have been tuned for sonic, which can be used

    if args.local:
        game = 'SonicTheHedgehog-Genesis'
        state = 'GreenHillZone.Act1'
        register_env(env_name, lambda config: sonic_on_ray.make_local(game, state))

    else:
        register_env(env_name, lambda config: sonic_on_ray.make())

    ray.init()

    config = ppo.DEFAULT_CONFIG.copy()

    config.update({
        'timesteps_per_batch': 40000,
        'min_steps_per_task': 100,
        'num_workers': 32,
        'gamma': 0.99,
        'lambda': 0.95,
        'clip_param': 0.1,
        'num_sgd_iter': 30,
        'sgd_batchsize': 4096,
        'sgd_stepsize': 5e-5,
        'use_gae': True,
        'horizon': 4000,
        'devices': ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3', '/gpu:4', '/gpu:5',
                    '/gpu:6', 'gpu:7'],
        'tf_session_args': {
            'gpu_options': {'allow_growth': True}
        }
    })

    alg = ppo.PPOAgent(config=config, env=env_name)
    print("Created a PSqO object")
    if args.load_checkpoint is not None:
        print("Trying to restore from checkpoint", args.load_checkpoint)
        alg.restore(args.load_checkpoint)
        print("Restored state from checkpoint:", args.load_checkpoint)

    for i in range(10):
        try:
            print("Starting to train")
            result = alg.train()
            print('result = {}'.format(result))

        except gre.GymRemoteError as e:
            print('exception', e)

        #if i % 10 == 0:
        #    checkpoint = alg.save(args.save_checkpoint_dir)
        #    print('checkpoint saved at', checkpoint)
if __name__ == "__main__":
    main()
