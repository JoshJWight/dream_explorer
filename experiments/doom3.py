import gym
import gymnasium
from vizdoom import gymnasium_wrapper
import random

#List of environments for this particular experiment
DOOM_ENVS = [
    'VizdoomDefendLine-v0',
    'VizdoomDefendCenter-v0',
]

NOOP = 0
MOVE_LEFT = 1
MOVE_RIGHT = 2
ATTACK = 3
MOVE_FORWARD = 4
MOVE_BACKWARD = 5
TURN_LEFT = 6
TURN_RIGHT = 7

DOOM_ENV_CONTROLS = {
    'VizdoomBasic-v0': [NOOP, ATTACK, MOVE_RIGHT, MOVE_LEFT],
    'VizdoomCorridor-v0': [NOOP, TURN_RIGHT, TURN_LEFT, MOVE_BACKWARD, MOVE_FORWARD, ATTACK, MOVE_RIGHT, MOVE_LEFT],
    'VizdoomDefendCenter-v0': [NOOP, ATTACK, TURN_RIGHT, TURN_LEFT],
    'VizdoomDefendLine-v0': [NOOP, ATTACK, TURN_RIGHT, TURN_LEFT],
    'VizdoomHealthGathering-v0': [NOOP, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT],
    'VizdoomHealthGatheringSupreme-v0': [NOOP, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT],
    'VizdoomMyWayHome-v0': [NOOP, MOVE_RIGHT, MOVE_LEFT, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT],
    'VizdoomPredictPosition-v0': [NOOP, ATTACK, TURN_RIGHT, TURN_LEFT],
    'VizdoomTakeCover-v0': [NOOP, MOVE_RIGHT, MOVE_LEFT],
}

REMAPPING = {}
for env_name, controls in DOOM_ENV_CONTROLS.items():
    REMAPPING[env_name] = {}
    for i in range(8):
        REMAPPING[env_name][i] = 0
    for i, control in enumerate(controls):
        REMAPPING[env_name][control] = i

    
class DoomEnv(gym.Env):
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gymnasium.make(env_name)

    def reset(self):
        obs, _ = self.env.reset()
        return obs['screen']
    
    def step(self, action):
        action = REMAPPING[self.env_name][action]
        obs, reward, done, truncated, info = self.env.step(action)
        return obs['screen'], reward, (done or truncated), info
    
    @property
    def action_space(self):
        #return self.env.action_space
        return gym.spaces.Discrete(8)
    
    @property
    def observation_space(self):
        return self.env.observation_space['screen']

#TODO at this point I can probably refactor several of these into a single class
class MultiDoom(gym.Env):
    def __init__(self, startIdx=-1):
        self.envs = {}
        self.startIdx = startIdx
        self.go_to(startIdx)

    def env_for(self, idx):
        env = DoomEnv(DOOM_ENVS[idx])
        return env

    def go_to(self, idx):
        if idx == -1:
            idx = random.randrange(0, len(DOOM_ENVS))
        if idx not in self.envs:
            self.envs[idx] = self.env_for(idx)
        self.currentIdx = idx
        self.env = self.envs[idx]
        return self.env.reset()

    def reset(self):
        return self.go_to(self.startIdx)
    
    def step(self, action):

        state, reward, done, info = self.env.step(action)

        return state, reward, done, info

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space
##############################################################################################################

def main():

  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['xlarge'])
  config = config.update({
      'logdir': '~/logdir/doom/3',
      'run.train_ratio': 64,
      'run.log_every': 30,  # Seconds
      'batch_size': 16,
      'jax.prealloc': False,
      'encoder.mlp_keys': '$^',
      'decoder.mlp_keys': '$^',
      'encoder.cnn_keys': 'image',
      'decoder.cnn_keys': 'image',
      # 'jax.platform': 'cpu',
  })
  config = embodied.Flags(config).parse()

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
      # embodied.logger.WandBOutput(logdir.name, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ])
  from gym.wrappers import ResizeObservation
  from dreamerv3.embodied.envs import from_gym
  def make_env():
    env = MultiDoom(-1)
    env = ResizeObservation(env, 64)
    env = from_gym.FromGym(env)
    env = dreamerv3.wrap_env(env, config)
    return env
  env = embodied.BatchEnv([make_env() for i in range(16)], parallel=False)

  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  replay = embodied.replay.Uniform(
      config.batch_length, config.replay_size, logdir / 'replay')
  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)
  embodied.run.train(agent, env, replay, logger, args)
  # embodied.run.eval_only(agent, env, logger, args)


if __name__ == '__main__':
  main()