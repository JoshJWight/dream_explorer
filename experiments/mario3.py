def main():

    import warnings
    import dreamerv3
    from dreamerv3 import embodied
    import gym
    import copy
    warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['xlarge'])
    config = config.update({
        'logdir': '~/logdir/mariobros/3',
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

    from embodied.envs import from_gym
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
    from gym.wrappers import ResizeObservation
    class MarioScoreReward(gym.Wrapper):
        def __init__(self, env):
            super(MarioScoreReward, self).__init__(env)

        def reset(self):
            state = self.env.reset()
            self.last_info = None
            self.x_record = 0
            self.idle_timer = 0
            self.penalties = 0
            return state

        def step(self, action):
            state, _, done, info = self.env.step(action)
            if self.last_info is None:
                reward = 0
                self.x_record = info['x_pos']
            else:
                reward = (info['score'] - self.last_info['score']) / 100.0
                if info['life'] < self.last_info['life']:
                    reward -= 100.0
                    self.x_record = info['x_pos']
                    self.penalties = 0
                if info['flag_get']:
                    reward += 100.0
                elif done:
                    reward -= 100.0

                if reward != 0:
                    self.idle_timer = 0
                elif info['x_pos'] > self.x_record + 100:
                    self.x_record = info['x_pos']
                    self.idle_timer = 0
                    if self.penalties > 0:
                        reward += 1.0
                        self.penalties -= 1
                elif info['stage'] > self.last_info['stage'] or info['world'] > self.last_info['world']:
                    self.x_record = info['x_pos']
                    self.idle_timer = 0
                else:
                    self.idle_timer += 1
                    if self.idle_timer > info["time"]:
                        reward -= 1.0
                        self.penalties += 1
                        self.idle_timer = 0
            self.last_info = copy.deepcopy(info)
            return state, reward, done, info


    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = ResizeObservation(env, 64)
    env = MarioScoreReward(env)
    env = from_gym.FromGym(env)
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)

    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    replay = embodied.replay.Uniform(
        config.batch_length, config.replay_size, logdir / 'replay')
    args = embodied.Config(
        **config.run, logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length)
    embodied.run.train(agent, env, replay, logger, args)


if __name__ == '__main__':
    main()
