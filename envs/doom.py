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