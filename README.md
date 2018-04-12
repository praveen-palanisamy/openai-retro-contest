### Experiments for the retro contest

#### 1. Behavior cloning

The `behavior-cloning` folder contains the `keyboard_control_sonic.py` script to collect demonstrations using the keyboard keys, convert them into the action space for the Sonic env and records the demonstrations in `*.bk2` files which can be played back using the playback script by stepping through the states and actions in the `*.bk2` file. This can be used to 

 *  Populate initial replay buffers or
 *  Directly clone the policy which can then act as the initial policy for an RL agent

The `*.bk2` files can also be used to generate `*.mp4` videos of the screen to see how Sonic plays.

#### 2. Sonic on Ray

The `sonic-on-ray` folder was inherited from `openai/sonic-on-ray` project. The `sonic-on-ray/sonic_on_ray_docker` folder contains `retro_train_ppo.py` which is docker-ready along with the modified version of `sonic-on-ray/sonic_on_ray_docker/sonic_on_ray/sonic_on_ray.py` which uses gym remote env or a retro_contest local environment based on the argument (`--local`) passed to `sonic-on-ray/sonic_on_ray_docker/retro_train_ppo.py`

The `retro_train_ppo.py` also takes the `--save-dir` and `--load-checkpoint` arguments to save to and load from checkpoints.