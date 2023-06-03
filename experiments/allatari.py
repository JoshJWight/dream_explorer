import gym
import gym.envs.atari
import random

#The list from the agent57 paper
ATARI_GAMES = [
    'alien',
    'amidar',
    'assault',
    'asterix',
    'asteroids',
    'atlantis',
    'bank_heist',
    'battle_zone',
    'beam_rider',
    'berzerk',
    'bowling',
    'boxing',
    'breakout',
    'centipede',
    'chopper_command',
    'crazy_climber',
    #'defender', #disabled because the ROM is busted, TODO: find a working ROM
    'demon_attack',
    'double_dunk',
    'enduro',
    'fishing_derby',
    'freeway',
    'frostbite',
    'gopher',
    'gravitar',
    'hero',
    'ice_hockey',
    'jamesbond',
    'kangaroo',
    'krull',
    'kung_fu_master',
    'montezuma_revenge',
    'ms_pacman',
    'name_this_game',
    'phoenix',
    'pitfall',
    'pong',
    'private_eye',
    'qbert',
    'riverraid',
    'road_runner',
    'robotank',
    'seaquest',
    #'skiing', #has a different action space, even with full_action_space=True
    'solaris',
    'space_invaders',
    'star_gunner',
    #'surround', #might not actually be available from gym?
    'tennis',
    'time_pilot',
    'tutankham',
    'up_n_down',
    'venture',
    'video_pinball',
    'wizard_of_wor',
    'yars_revenge',
    'zaxxon'
]


class AllAtari(gym.Wrapper):
    def __init__(self, startIdx=-1):
        self.envs = {}
        self.startIdx = startIdx
        self.go_to(startIdx)

    def env_for(self, idx):
        env = gym.envs.atari.AtariEnv(
            game=ATARI_GAMES[idx],
            obs_type='image',
            frameskip=1,
            repeat_action_probability=0,
            full_action_space=True
        )
        return env

    def go_to(self, idx):
        if idx == -1:
            idx = random.randrange(0, len(ATARI_GAMES))
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













def main():

  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['xlarge'])
  config = config.update({
      'logdir': '~/logdir/all_atari/1',
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
    env = AllAtari()
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
