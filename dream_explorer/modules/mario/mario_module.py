from .. import module
from .mario_env import SequentialMario, MarioGoRightReward
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import ResizeObservation
from dreamerv3.embodied.envs import from_gym


class MarioModule(module.Module):
    def __init__(self):
        pass

    def levels(self):
        lvls = []
        for world in range(1, 9):
            for stage in range(1, 5):
                lvls.append("SMB1 " + str(world) + "-" + str(stage))
        for world in range(1, 5):
            for stage in range(1, 5):
                lvls.append("SMB2 " + str(world) + "-" + str(stage))

        return lvls
    
    def create_env(self):
        return self.set_level(-1)

    def set_level(self, level):
        env = SequentialMario(level)
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        env = MarioGoRightReward(env)
        env = ResizeObservation(env, 64)
        env = from_gym.FromGym(env)
        return env
    
    def size(self):
        return 'xlarge'
    
    def action_keys(self):
        #This is COMPLEX_MOVEMENT but with different keys
        return  [[],
                ["Right"],
                ["Right", "z"],
                ["Right", "x"],
                ["Right", "z", "x"],
                ["z"],
                ["Left"],
                ["Left", "z"],
                ["Left", "x"],
                ["Left", "z", "x"],
                ["Down"],
                ["Up"]]