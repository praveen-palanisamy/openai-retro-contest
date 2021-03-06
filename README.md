## Deep RL Experiments for transfer-learning/retro contest

![openai-retro-contest-praveenp](http://praveenp.com/projects/rsc/imgs/openai-retro-contest-intro.gif)

The experiments are steps towards developing Deep Reinforcement Algorithms that have the ability to generalize to new scenarios based on previous experiences. This is a first step toward making the "test set" different from the "training set" in Reinforcement Learning.

### 0. Intro/writeup/motiv

Sonic is a side-scrolling 2D video game and for the most part, the character moves from left to right to make progress. The agent observes the game-play action from a side-view camera angle.

Each timestep advances the game by 4 frames,

**State/Observation Space**:  "Each observation is the pixels on the screen for the current frame, a shape `[224, 320, 3]` array of uint8 values"

**Action Space:** "Each action is which buttons to hold down until the next frame (a shape `[12]` array of bool values, one for each button on the Genesis controller where invalid button combinations (`Up+Down` or accessing the start menu) are ignored."

Env step() method employs a sticky frame skip. "Like standard frame skip, sticky frame skip applies $n$ actions over $4n$ frames.  However, for each action, we delay it by one frame with robability `0.25`, applying the previous action for that frame instead.  The following diagram shows an example of an action sequence with sticky frame skip:"

![1523572075287](/tmp/1523572075287.png)

" In fact, there are only eight essential button combinations:

`{{},{LEFT},{RIGHT},{LEFT, DOWN},{RIGHT, DOWN},{DOWN},{DOWN, },{B}}`

The UP button is also useful on occasion, but for the most part it can be ignored"

**Collecting rings (100 rings = 1 additional life) is not useful as the episode resets if *a* life is lost. Destroying robots are not necessary to complete the game **

But, having **at least one ring** all/most of the time is a good strategy because that will let Sonic survive when he collides with an enemy or dangerous obstacle. If he is hit without any rings, life is lost, episode ends.

**A good curriculum sequence to start with:** 

*Green Hill Zone* --> *Marble Zone* --> *Spring Yard Zone* --> *Labyrinth Zone* --> *Star Light Zone* --> *Scrap Brain Zone*

====================

### 1. Ideas/Experiments

* Joint training
  - [x] Randomly select levels from a Game on every episode, train & test on validation levels
* Curriculum Learning:
  - [x] Sequentially train on levels on a Game while revisiting the previously trained levels
* Sequential training:
  - [x] Sequentially train on levels in a game

### 2. Description of existing experiments 

- #### Behavior cloning

The `behavior-cloning` folder contains the `keyboard_control_sonic.py` script to collect demonstrations using the keyboard keys, convert them into the action space for the Sonic env and records the demonstrations in `*.bk2` files which can be played back using the playback script by stepping through the states and actions in the `*.bk2` file. This can be used to 

 *  Populate initial replay buffers or
 *  Directly clone the policy which can then act as the initial policy for an RL agent

The `*.bk2` files can also be used to generate `*.mp4` videos of the screen to see how Sonic plays.

- #### Sonic on Ray : Distributed training at scale

The `sonic-on-ray` folder was inherited from `openai/sonic-on-ray` project. The `sonic-on-ray/sonic_on_ray_docker` folder contains `retro_train_ppo.py` which is docker-ready along with the modified version of `sonic-on-ray/sonic_on_ray_docker/sonic_on_ray/sonic_on_ray.py` which uses gym remote env or a retro_contest local environment based on the argument (`--local`) passed to `sonic-on-ray/sonic_on_ray_docker/retro_train_ppo.py`

The `retro_train_ppo.py` also takes the `--save-dir` and `--load-checkpoint` arguments to save to and load from checkpoints.

- #### GA3C Pytorch: GPU Accelerated A3C implementation in Pytorch

 The `experiments/ga3c_pytorch` folder contains necessary code for training and testing the GPU accelerated Asynchronous Advantage Actor-Critic agent implemented in PyTorch.

`sudo docker build -f Dockerfile -t local/retro_ga3c_pytorch:v1 .`

```bash
sudo docker run --runtime nvidia  -v `pwd`/trained_models:/root/compo/trained_models -v `pwd`/tmp:/root/compo/logs  local/retro_ga3c_pytorch:v1
```
To-Do:

- [ ] Document the code usage and implementation details

- #### PPO (+ A2C, ACKTR)

  (Currently on the 	`pytorch_ppo_a2c_acktr` branch based on [this implementation](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr) )

`experiments/pytorch-a2c--ppo-acktr`

Run the training script using:

`python main.py --retro-contest --num-processes 1500  --env-name "Sonic-GHZA1" --algo "ppo" --use-gae --lr 1e-4 --clip-param 0.1 --value-loss-coef 1 --num-mini-batch 16 --num-frames 200000000 --sonic-config-file sonic_GHZA1.json --ppo-epoch 8`



### 3. Utilities

- `utils/generate_train_test_conf.py` : Generates a `sonic_config.json` file from `sonic-train.csv` and `sonic-validation.csv`. This facilitates random/ordered/curriculum-based training scripts.

### 4. Tips & Tricks & Issue resolution 

 1. **Issue:**`RuntimeError: Cannot create multiple emulator instances per process`. This makes it difficult to shuffle the environment/levels for each episode i.e  run episodes from different game levels (Without running one environment instance per actor on separate processes)

    **Reason**: Occurs when you use `retro_contest` for `make`ing the environment.  Both`retro_contest.make(...)` and `retro_contest.local.make(...`) only allows instantiation of one environment. It does not seems to be a restriction imposed for the contest. The current implementation of OpenAI gym-retro does not support this. Therefor, even `retro.make(...)` can launch only one emulator instance per process.

    **Resolution:** One way to overcome this issue is to spawn a new process using multiprocessing to instantiate/make a new environment. I used @processify

2. Don't forget to pass `scenario='contest'` when you are running the gym retro environment locally when training for the contest!
  ```python
  import retro
  env = retro.make(game='SonicTheHedgehog-Genesis',     level='GreenHillZone.Act1', scenario='contest')
  ```
  Setting `scenari='contest'` while making the environment will create the environment with the json configuration that is used for the retro-contest and will provide a reward for change in `x`. Some of the environment wrappers (eg.: AllowBacktracking) are written assuming such a reward from the environment. If the `scenario='contest'` is left out, the reward from the environment will be same as the game's score.

### 5. Citing

If you use the code in this repository in whole or in parts for your work, please cite:

```bibtex
@misc{PraveenpRetroTransferLearning,
  author = {Palanisamy, Praveen},
  title = {OpenAI Retro Contest Experiments},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/praveen-palanisamy/openai-retro-contest}},
}
```
