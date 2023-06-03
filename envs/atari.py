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


class AllAtari(gym.Env):
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