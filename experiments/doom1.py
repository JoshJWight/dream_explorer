import gym
import gymnasium
from vizdoom import gymnasium_wrapper

class DoomEnv(gym.Env):
    def __init__(self):
        self.env = gymnasium.make("VizdoomCorridor-v0")

    def reset(self):
        obs, _ = self.env.reset()
        return obs['screen']
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs['screen'], reward, (done or truncated), info
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def observation_space(self):
        return self.env.observation_space['screen']


def main():

  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['xlarge'])
  config = config.update({
      'logdir': '~/logdir/doom/1',
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
    env = DoomEnv()
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
