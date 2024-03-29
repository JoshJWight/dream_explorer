from .. import module
from .doom_env import MultiDoom, DOOM_ENVS
from gym.wrappers import ResizeObservation
from dreamerv3.embodied.envs import from_gym

class DoomModule(module.Module):
    def __init__(self):
        pass

    def levels(self):
        return DOOM_ENVS
    
    def create_env(self):
        return self.set_level(-1)

    def set_level(self, level):
        env = MultiDoom(level)
        env = ResizeObservation(env, 64)
        env = from_gym.FromGym(env)
        return env

    def size(self):
        return 'xlarge'
    
    def action_keys(self):
        return [[],
                ["a"],
                ["d"],
                ["space"],
                ["w"],
                ["s"],
                ["Left"],
                ["Right"]]