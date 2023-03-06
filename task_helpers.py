import numpy as np
import crafter
import dreamerv3
import dreamerv3.embodied as embodied
from embodied.envs import from_gym
from embodied.envs import atari

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
    else:
        raise NotImplementedError
    
def action_for_task(task, key_map):
    if task == 'pong':
        return pong_action(key_map)
    elif task == 'asteroids':
        return asteroids_action(key_map)
    elif task == 'mspacman':
        return mspacman_action(key_map)
    elif task == 'crafter':
        return crafter_action(key_map)
    elif task == 'skiing':
        return skiing_action(key_map)
    else:
        raise NotImplementedError

def crafter_env():
    env = crafter.Env()  # Replace this with your Gym env.
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


def crafter_action(key_map):
    action = np.zeros((17), dtype=np.float32)
    if key_map["a"]:
        action[1] = 1 #LEFT
    elif key_map["d"]:
        action[2] = 1 #RIGHT
    elif key_map["w"]:
        action[3] = 1 #UP
    elif key_map["s"]:
        action[4] = 1 #DOWN
    elif key_map["space"]:
        action[5] = 1 #INTERACT
    elif key_map["Tab"]:
        action[6] = 1 #SLEEP
    elif key_map["r"]:
        action[7] = 1 #PLACE ROCK
    elif key_map["t"]:
        action[8] = 1 #PLACE TABLE
    elif key_map["f"]:
        action[9] = 1 #PLACE FURNACE
    elif key_map["p"]:
        action[10] = 1 #PLACE PLANT
    elif key_map["1"]:
        action[11] = 1 #CRAFT WOOD PICKAXE
    elif key_map["2"]:
        action[12] = 1 #CRAFT STONE PICKAXE
    elif key_map["3"]:
        action[13] = 1 #CRAFT IRON PICKAXE
    elif key_map["4"]:
        action[14] = 1 #CRAFT WOOD SWORD
    elif key_map["5"]:
        action[15] = 1 #CRAFT STONE SWORD
    elif key_map["6"]:
        action[16] = 1 #CRAFT IRON SWORD
    else:
        action[0] = 1 #NOOP

    return action


def pong_action(key_map):
    action = np.zeros((6), dtype=np.float32)
    if key_map["Up"]:
        action[2] = 1 #RIGHT
    elif key_map["Down"]:
        action[3] = 1 #LEFT
    elif key_map["Right"]: 
        action[4] = 1 #RIGHTFIRE
    elif key_map["Left"]:
        action[5] = 1 #LEFTFIRE
    elif key_map["space"]:
        action[1] = 1 #FIRE
    else:
        action[0] = 1 #NOOP
    return action

def asteroids_action(key_map):
    action = np.zeros((14), dtype=np.float32)
    up = key_map["Up"]
    down = key_map["Down"]
    left = key_map["Left"]
    right = key_map["Right"]
    fire = key_map["space"]

    if up and left and fire:
        action[13] = 1 #UPLEFTFIRE
    elif up and right and fire:
        action[12] = 1 #UPRIGHTFIRE
    elif down and fire:
        action[11] = 1 #DOWNFIRE
    elif left and fire:
        action[10] = 1 #LEFTFIRE
    elif right and fire:
        action[9] = 1 #RIGHTFIRE
    elif up and fire:
        action[8] = 1 #UPFIRE
    elif up and left:
        action[7] = 1 #UPLEFT
    elif up and right:
        action[6] = 1 #UPRIGHT
    elif down:
        action[5] = 1 #DOWN
    elif left:
        action[4] = 1 #LEFT
    elif right:
        action[3] = 1 #RIGHT
    elif up:
        action[2] = 1 #UP
    elif fire:
        action[1] = 1 #FIRE
    else:
        action[0] = 1 #NOOP

    return action

def mspacman_action(key_map):
    action = np.zeros((9), dtype=np.float32)
    up = key_map["Up"]
    down = key_map["Down"]
    left = key_map["Left"]
    right = key_map["Right"]
    if down and left:
        action[8] = 1
    elif down and right:
        action[7] = 1
    elif up and left:
        action[6] = 1
    elif up and right:
        action[5] = 1
    elif down:
        action[4] = 1
    elif left:
        action[3] = 1
    elif right:
        action[2] = 1
    elif up:
        action[1] = 1
    else:
        action[0] = 1

    return action

def skiing_action(key_map):
    action = np.zeros((3), dtype=np.float32)
    if key_map["Left"]:
        action[2] = 1
    elif key_map["Right"]:
        action[1] = 1
    else:
        action[0] = 1
    return action