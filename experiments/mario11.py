def main():

    import warnings
    import dreamerv3
    from dreamerv3 import embodied
    import gym
    import copy
    import random
    warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['xlarge'])
    config = config.update({
        'logdir': '~/logdir/mariobros/11',
        'run.train_ratio': 64,
        #'run.train_ratio': 16,
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
    
    class MarioGoRightReward(gym.Wrapper):
        def __init__(self, env):
            super(MarioGoRightReward, self).__init__(env)

        def reset(self):
            state = self.env.reset()
            self.last_info = None
            self.x_record = 0
            return state

        def step(self, action):
            state, _, done, info = self.env.step(action)
            reward = 0
            if self.last_info is not None:
                if self.last_info['flag_get']:
                    self.x_record = 0
                    reward += 1
                if info['x_pos'] > self.x_record and info['y_pos'] == self.last_info['y_pos']:
                    reward += info['x_pos'] - self.x_record
                    self.x_record = info['x_pos']
                if done:
                    reward -= 1
            self.last_info = copy.deepcopy(info)
            return state, reward, done, info

    class SequentialMario(gym.Wrapper):
        def __init__(self, startIdx=-1):
            self.envs = {}
            self.startIdx = startIdx
            self.NUM_LEVELS = 32
            self.go_to(startIdx)

        def env_for(self, idx):
            world = idx // 4 + 1
            stage = idx % 4 + 1
            envName = f'SuperMarioBros-{world}-{stage}-v0'
            env = gym_super_mario_bros.make(envName)
            return env

        def go_to(self, idx):
            if idx == -1:
                idx = random.randrange(0, self.NUM_LEVELS)
            if idx not in self.envs:
                self.envs[idx] = self.env_for(idx)
            self.env = self.envs[idx]
            self.currentIdx = idx
            return self.env.reset()

        def reset(self):
            return self.go_to(self.startIdx)    
        
        def step(self, action):
            state, reward, done, info = self.env.step(action)

            if info['flag_get'] and self.currentIdx+1 < self.NUM_LEVELS:
                self.go_to(self.currentIdx + 1)
                done = False

            return state, reward, done, info

        @property
        def action_space(self):
            return self.env.action_space

        @property
        def observation_space(self):
            return self.env.observation_space

    #env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0')
    env = SequentialMario(-1)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = ResizeObservation(env, 64)
    env = MarioGoRightReward(env)
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
