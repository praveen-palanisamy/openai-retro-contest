"""
Environments and wrappers for Sonic training.
"""

import gym
import numpy as np
import random

#from baselines.common.atari_wrappers import WarpFrame, FrameStack
#import gym_remote.client as grc

import retro
from gym.spaces import Box
import cv2

from abc import ABC, abstractmethod
from torch.multiprocessing import Pipe, Process


def make_env(stack=True, scale_rew=True):
    """
    Create an environment with some standard wrappers.
    """
    env = grc.RemoteEnv('tmp/sock')
    env = SonicDiscretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    #env = WarpFrame(env)
    # Disabling FrameStack because it currently uses a different Box. (Box(0,255,w,h) compared to Box(0,1,w,h)
    #if stack:
    #    env = FrameStack(env, 4)
    env = CustomWarpFrame(env)
    env = NormalizedEnv(env)
    return env


def make_local_env(game, state, stack=True, scale_rew=False):
    """
    Create an instance of a local Gym environment with some standard wrappers
    """
    env = retro.make(game=game, state=state, scenario='contest')
    env = SonicDiscretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    #env = WarpFrame(env)
    # Disabling FrameStack because it currently uses a different Box. (Box(0,255,w,h) compared to Box(0,1,w,h)
    #if stack:
    #    env = FrameStack(env, 4)
    env = CustomWarpFrame(env)
    env = NormalizedEnv(env)
    env = AllowBacktracking(env)
    return env


def make_env_in_sep_proc(game, state, shared_pipe, parent_pipe, stack=False, scale_rew=False):
    """
    Create an environment instance (remote or local) in a separate proc and return the env object
    :return: The env running in a different proc
    """
    print("make_env_in_sep_proc: Making game=", game, "state=", state)
    parent_pipe.close()

    env = retro.make(game=game, state=state, scenario='contest')
    env = SonicDiscretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    env = CustomWarpFrame(env)
    env = NormalizedEnv(env)
    env = AllowBacktracking(env)
    while True:
        method, data = shared_pipe.recv()
        if method == 'step':
            next_obs, rew, done, info = env.step(data)
            if done:
                next_obs = env.reset()
            shared_pipe.send((next_obs, rew, done, info))

        if method == 'reset':
            obs = env.reset()
            shared_pipe.send(obs)

        if method == 'get_spaces':
            shared_pipe.send((env.observation_space, env.action_space))


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self):
        logger.warn('Render not defined for %s'%self)

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

class SubprocVecSonicEnv(VecEnv):
    def __init__(self, env_confs, num_envs, spaces=None):
        """
        envs: list of Sonic environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = num_envs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = []
        for (worker_conn, parent_conn) in zip(self.work_remotes, self.remotes):
            env_conf = random.sample(env_confs, 1)[0]
            self.ps.append(Process(target=make_env_in_sep_proc, args=(env_conf['game'],
                                                                      env_conf['level'],
                                                                      worker_conn,
                                                                      parent_conn)))
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()

        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, num_envs, observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


class CustomWarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = Box(0.0, 1.0, [1, self.width, self.height], dtype=np.float16)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame.astype(np.float32)
        frame *= (1.0 / 255.0)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = np.reshape(frame, [1, self.width, self.height])
        return frame


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)


class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()


class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.

    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.01


class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info
