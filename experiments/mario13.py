import gym
import gym_super_mario_bros
import random
import copy

def registerEnv(id, **kwargs):
    entry_point = 'gym_super_mario_bros:SuperMarioBrosEnv'

    gym.envs.registration.register(
            id=id,
            entry_point=entry_point,
            max_episode_steps=9999999,
            reward_threshold=9999999,
            kwargs=kwargs,
            nondeterministic=True,
    )

def registerMario2():
    #only worlds 1-4 actually work, allegedly
    try:
        for world in range(1, 5):
            for level in range(1, 5):
                envName = f'SuperMarioBros2-{world}-{level}-v0'
                registerEnv(envName, lost_levels=True, rom_mode='vanilla', target=(world, level))
    except:
        pass

class SequentialMario(gym.Wrapper):
    def __init__(self, startIdx=-1):
        registerMario2()
        self.envs = {}
        self.startIdx = startIdx
        self.NUM_LEVELS = 48
        self.go_to(startIdx)

    def env_for(self, idx):
        game = idx // 32 + 1
        gameName = 'SuperMarioBros' if game == 1 else 'SuperMarioBros2' 
        #this logic would need to be changed if we ever got to world 9 of smb2
        world = (idx // 4) % 8 + 1
        stage = idx % 4 + 1

        envName = f'{gameName}-{world}-{stage}-v0'
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

class MarioGoRightReward(gym.Wrapper):
    def __init__(self, env):
        super(MarioGoRightReward, self).__init__(env)

    def reset(self):
        state = self.env.reset()
        self.last_info = None
        self.x_record = 0
        self.steps_same_y = 0
        self.jump_start = 0
        self.jump_max = 0
        self.steps_since_score = 0
        return state

    def step(self, action):
        state, _, done, info = self.env.step(action)
        reward = 0
        if self.last_info is not None and info['x_pos'] < 20000:
            if self.last_info['flag_get']:
                self.x_record = info['x_pos']
                reward += 20

            if info['y_pos'] == self.last_info['y_pos']:
                self.steps_same_y += 1
            else:
                self.steps_same_y = 0

            if info['y_pos'] > self.jump_max:
                self.jump_max = info['y_pos']

            if info['x_pos'] > (self.x_record + 20) and self.steps_same_y >= 3:
                reward += (info['x_pos'] - self.x_record) * (self.jump_max - self.jump_start + 1) / 1000.0
                self.x_record = info['x_pos']
                self.jump_start = info['y_pos']
                self.jump_max = info['y_pos']
                self.steps_since_score = 0
            else:
                self.steps_since_score += 1

            if self.steps_since_score > 500:
                reward -= 1
            if self.steps_since_score > 600:
                done = True

            if done and not info['flag_get']:
                reward = -1
            
            if info['score'] > self.last_info['score']:
                reward +=1
        self.last_info = copy.deepcopy(info)
        return state, reward, done, info

def main():

    import warnings
    import dreamerv3
    from dreamerv3 import embodied
    warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['xlarge'])
    config = config.update({
        'logdir': '~/logdir/mariobros/13',
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
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
    from gym.wrappers import ResizeObservation
    

    #env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0')
    def make_env():
        env = SequentialMario(-1)
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        env = ResizeObservation(env, 64)
        env = MarioGoRightReward(env)
        env = from_gym.FromGym(env)
        env = dreamerv3.wrap_env(env, config)
        return env
    env = embodied.BatchEnv([make_env() for i in range(10)], parallel=False)

    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    replay = embodied.replay.Uniform(
        config.batch_length, config.replay_size, logdir / 'replay')
    args = embodied.Config(
        **config.run, logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length)
    embodied.run.train(agent, env, replay, logger, args)

    

if __name__ == '__main__':
    main()
