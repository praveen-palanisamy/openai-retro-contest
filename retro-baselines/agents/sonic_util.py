"""
Environments and wrappers for Sonic training.
"""

import gym
import numpy as np

from baselines.common.atari_wrappers import WarpFrame, FrameStack
import gym_remote.client as grc

import retro


import inspect
import os
import sys
import traceback

from functools import wraps
from multiprocessing import Process, Queue


class Sentinel:
    pass


def processify(func):
    '''Decorator to run a function as a process.
    Be sure that every argument and the return value
    is *pickable*.
    The created process is joined, so the code does not
    run in parallel.
    '''

    def process_generator_func(q, *args, **kwargs):
        result = None
        error = None
        it = iter(func())
        while error is None and result != Sentinel:
            try:
                result = next(it)
                error = None
            except StopIteration:
                result = Sentinel
                error = None
            except Exception:
                ex_type, ex_value, tb = sys.exc_info()
                error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
                result = None
            q.put((result, error))

    def process_func(q, *args, **kwargs):
        try:
            print("trying")
            result = func(*args, **kwargs)
        except Exception:
            ex_type, ex_value, tb = sys.exc_info()
            error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
            result = None
        else:
            error = None

        q.put((result, error))

    def wrap_func(*args, **kwargs):
        # register original function with different name
        # in sys.modules so it is pickable
        print("wrapping_func")
        process_func.__name__ = func.__name__ + 'processify_func'
        setattr(sys.modules[__name__], process_func.__name__, process_func)

        q = Queue()
        p = Process(target=process_func, args=[q] + list(args), kwargs=kwargs)
        p.start()
        result, error = q.get()
        p.join()

        if error:
            ex_type, ex_value, tb_str = error
            message = '%s (in subprocess)\n%s' % (str(ex_value), tb_str)
            raise ex_type(message)

        return result

    def wrap_generator_func(*args, **kwargs):
        # register original function with different name
        # in sys.modules so it is pickable
        process_generator_func.__name__ = func.__name__ + 'processify_generator_func'
        setattr(sys.modules[__name__], process_generator_func.__name__, process_generator_func)

        q = Queue()
        p = Process(target=process_generator_func, args=[q] + list(args), kwargs=kwargs)
        p.start()

        result = None
        error = None
        while error is None:
            result, error = q.get()
            if result == Sentinel:
                break
            yield result
        p.join()

        if error:
            ex_type, ex_value, tb_str = error
            message = '%s (in subprocess)\n%s' % (str(ex_value), tb_str)
            raise ex_type(message)

    @wraps(func)
    def wrapper(*args, **kwargs):
        if inspect.isgeneratorfunction(func):
            return wrap_generator_func(*args, **kwargs)
        else:
            return wrap_func(*args, **kwargs)
    return wrapper


def make_env(stack=True, scale_rew=True):
    """
    Create an environment with some standard wrappers.
    """
    env = grc.RemoteEnv('tmp/sock')
    env = SonicDiscretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame(env)
    if stack:
        env = FrameStack(env, 4)

    env = AllowBacktracking(env)

    return env



#@processify
def make_local_env(game="SonicTheHedgehog-Genesis", state="GreenHillZone.Act1", stack=True, scale_rew=True):
    """
    Create an instance of a local Gym environment with some standard wrappers
    """
    print("Entering make_local_env")
    env = retro.make(game=game, state=state)
    print("Got env")
    env = SonicDiscretizer(env)
    print("discretized env")
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame(env)
    if stack:
        env = FrameStack(env, 4)
    print("About to return")
    return env

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
