from .. import module
from .atari_env import AllAtari
from gym.wrappers import ResizeObservation
from dreamerv3.embodied.envs import from_gym

class AtariModule(module.Module):
    def __init__(self):
        pass

    def levels(self):
        return []
    
    def create_env(self):
        return self.set_level(-1)

    def set_level(self, level):
        env = AllAtari(level)
        env = ResizeObservation(env, 64)
        env = from_gym.FromGym(env)
        return env

    def size(self):
        return 'xlarge'
    
    def action_keys(self):
        #Full action space
        return [[],
                ["space"],
                ["Up"],
                ["Right"],
                ["Left"],
                ["Down"],
                ["Up", "Right"],
                ["Up", "Left"],
                ["Down", "Right"],
                ["Down", "Left"],
                ["Up", "space"],
                ["Right", "space"],
                ["Left", "space"],
                ["Down", "space"],
                ["Up", "Right", "space"],
                ["Up", "Left", "space"],
                ["Down", "Right", "space"],
                ["Down", "Left", "space"]]