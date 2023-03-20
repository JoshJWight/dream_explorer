import numpy as np
import crafter
from dreamerv3.embodied.envs import from_gym
from dreamerv3.embodied.envs import atari

import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import ResizeObservation

import copy

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

class MarioIdlePenaltyReward(gym.Wrapper):
    def __init__(self, env):
        super(MarioIdlePenaltyReward, self).__init__(env)

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

class MarioScoreReward(gym.Wrapper):
    def __init__(self, env):
        super(MarioScoreReward, self).__init__(env)

    def reset(self):
        state = self.env.reset()
        self._current_score = 0
        return state

    def step(self, action):
        state, _, done, info = self.env.step(action)
        reward = (info['score'] - self._current_score) / 100.0
        self._current_score = info['score']
        return state, reward, done, info
    
class MarioHaltingScoreReward(gym.Wrapper):
    def __init__(self, env):
        super(MarioHaltingScoreReward, self).__init__(env)

    def reset(self):
        state = self.env.reset()
        self.last_info = None
        self.idle_timer = 0
        self.x_record = 0
        self.xy_record = 0
        self.death_x = -10000
        return state

    def step(self, action):
        self.idle_timer += 1
        state, _, done, info = self.env.step(action)
        reward = 0
        if self.last_info is not None:
            score_reward = (info['score'] - self.last_info['score']) / 100.0
            if info['x_pos'] > self.death_x:
                reward += score_reward

            if info['x_pos'] > self.x_record:
                self.idle_timer = 0
                self.x_record = info['x_pos']
            elif info['x_pos'] * info['y_pos'] > self.xy_record:
                self.idle_timer = 0
                self.xy_record = info['x_pos'] * info['y_pos']
            elif self.last_info['flag_get']:
                self.x_record = 0
                self.xy_record = 0
                self.idle_timer = 0
                self.death_x = -10000
            elif info['life'] < self.last_info['life']:
                self.x_record = 0
                self.xy_record = 0
                self.idle_timer = 0
                self.death_x = max(self.last_info['x_pos'], self.death_x)

            if score_reward > 0:
                self.idle_timer = 0

            if self.idle_timer >= 600: #10 seconds at 60 fps
                done = True
            
            if done:
                reward -= 1
        
        self.last_info = copy.deepcopy(info)
        return state, reward, done, info
    
class StickyMario(gym.Wrapper):
    def __init__(self, env):
        super(StickyMario, self).__init__(env)
        self.env = env

    def reset(self):
        state = self.env.reset()
        self.last_action = self.env.action_space.sample()
        self.stick_timer = 0
        return state

    def step(self, action):
        if action == 11: #Up
            #If this value is too low it doesn't help exploration enough
            #If it's too high then if the agent is uncertain it will end up sticking forever.
            self.stick_timer = 15
        else:
            self.stick_timer -= 1

        if self.stick_timer > 0:
            action = self.last_action
        else:
            self.last_action = action

        state, reward, done, info = self.env.step(action)
        
        return state, reward, done, info

def mario_env_score():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = MarioHaltingScoreReward(env)
    env = StickyMario(env)
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