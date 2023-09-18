import numpy as np
import crafter
from dreamerv3.embodied.envs import from_gym
from dreamerv3.embodied.envs import atari

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import ResizeObservation

from mario import mario_helpers
from envs.atari import AllAtari
from envs.doom import MultiDoom

ACTION_KEYS = {}


def empty_key_map():
    return {
        "Up": False,
        "Down": False,
        "Left": False,
        "Right": False,
        "space": False,
        "Tab": False,
        "r": False,
        "t": False,
        "f": False,
        "p": False,
        "1": False,
        "2": False,
        "3": False,
        "4": False,
        "5": False,
        "6": False,
        "a": False,
        "d": False,
        "w": False,
        "s": False,
        "z": False,
        "x": False,
    }

def env_for_task(task):
    if task == 'pong':
        return pong_env()
    elif task == 'asteroids':
        return asteroids_env()
    elif task == 'mspacman':
        return mspacman_env()
    elif task == 'crafter':
        return crafter_env()
    elif task == 'skiing':
        return skiing_env()
    elif task == 'mario':
        #return mario_env()
        return mario_env_score()
    elif task == 'mario_random':
        return mario_env_random()
    elif task == 'atari':
        return atari_env()
    elif task == 'doom':
        return doom_env()
    else:
        raise NotImplementedError
    
def action_for_task(task, key_map):
    if task in ACTION_KEYS:
        task_keys = ACTION_KEYS[task]
        action = np.zeros(len(ACTION_KEYS[task]))
        #Find the longest key combination that is pressed
        best_index = 0 
        best_len = 0
        for i in range(len(task_keys)):
            if all(key_map[key] for key in task_keys[i]):
                if len(task_keys[i]) > best_len:
                    best_index = i
                    best_len = len(task_keys[i])
        action[best_index] = 1
        return action
    else:
        raise NotImplementedError(f"Task {task} not implemented. Please add it to task_helpers.py")
    
    

def crafter_env():
    env = crafter.Env()  # Replace this with your Gym env.
    env = from_gym.FromGym(env)
    return env

def mario_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = ResizeObservation(env, 64)
    env = from_gym.FromGym(env)
    return env

def mario_env_score():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = mario_helpers.MarioHaltingScoreReward(env)
    env = mario_helpers.StickyMario(env)
    env = ResizeObservation(env, 64)
    env = from_gym.FromGym(env)
    return env

def mario_env_random():
    return mario_env_sequential(-1)

def mario_env_sequential(start_level):
    env = mario_helpers.SequentialMario(start_level)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = mario_helpers.MarioGoRightReward(env)
    env = ResizeObservation(env, 64)
    env = from_gym.FromGym(env)
    return env

def pong_env():
    return atari.Atari("pong", gray=False, actions="needed", size=(64, 64))

def asteroids_env():
    return atari.Atari("asteroids", gray=False, actions="needed", size=(64, 64))

def mspacman_env():
    return atari.Atari("ms_pacman", gray=False, actions="needed", size=(64, 64))

def skiing_env():
    return atari.Atari("skiing", gray=False, actions="needed", size=(64, 64))

def atari_env(game_idx=-1):
    env = AllAtari(game_idx)
    env = ResizeObservation(env, 64)
    env = from_gym.FromGym(env)
    return env

def doom_env(env_idx=-1):
    env = MultiDoom(env_idx)
    env = ResizeObservation(env, 64)
    env = from_gym.FromGym(env)
    return env


ACTION_KEYS["crafter"] = [[],
                            ["a"],
                            ["d"],
                            ["w"],
                            ["s"],
                            ["space"],
                            ["Tab"],
                            ["r"],
                            ["t"],
                            ["f"],
                            ["p"],
                            ["1"],
                            ["2"],
                            ["3"],
                            ["4"],
                            ["5"],
                            ["6"]]

#This is COMPLEX_MOVEMENT but with different keys
ACTION_KEYS["mario"] = [[],
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
ACTION_KEYS["mario_random"] = ACTION_KEYS["mario"]

#Full action space
ACTION_KEYS["atari"] = [[],
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


ACTION_KEYS["pong"] = [[], 
                       ["space"], 
                       ["Up"], 
                       ["Down"], 
                       ["Right"], 
                       ["Left"]]

ACTION_KEYS["asteroids"] = [[], 
                            ["space"], 
                            ["Up"], 
                            ["Right"], 
                            ["Left"], 
                            ["Down"], 
                            ["Up", "Right"], 
                            ["Up", "Left"], 
                            ["Up", "space"], 
                            ["Right", "space"], 
                            ["Left", "space"], 
                            ["Down", "space"], 
                            ["Up", "Right", "space"], 
                            ["Up", "Left", "space"]]

ACTION_KEYS["mspacman"] = [[],
                            ["Up"],
                            ["Right"],
                            ["Left"],
                            ["Down"],
                            ["Up", "Right"],
                            ["Up", "Left"],
                            ["Down", "Right"],
                            ["Down", "Left"]]

ACTION_KEYS["skiing"] = [[],
                         ["Right"],
                         ["Left"]]

ACTION_KEYS["doom"] = [[],
                        ["a"],
                        ["d"],
                        ["space"],
                        ["w"],
                        ["s"],
                        ["Left"],
                        ["Right"]]