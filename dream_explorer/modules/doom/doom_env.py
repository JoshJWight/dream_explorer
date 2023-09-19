import gym
import gymnasium
from vizdoom import gymnasium_wrapper
import random

DOOM_ENVS = [
    'VizdoomBasic-v0',
    'VizdoomCorridor-v0',
    'VizdoomDefendCenter-v0',
    'VizdoomDefendLine-v0',
    'VizdoomHealthGathering-v0',
    'VizdoomHealthGatheringSupreme-v0',
    'VizdoomMyWayHome-v0',
    'VizdoomPredictPosition-v0',
    'VizdoomTakeCover-v0',
    #'VizdoomDeathmatch-v0', This one has weird complicated controls, ignore for now
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
        print(f"Action: {action}")
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