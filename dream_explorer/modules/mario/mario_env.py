import gym
import gym_super_mario_bros
import copy
import random

from .register_mario2 import registerMario2

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

            #TODO slap on a patch to discourage exploitation of repeatable point sources like shells

            if score_reward > 0:
                self.idle_timer = 0

            if self.idle_timer >= 600: #10 seconds at 60 fps
                done = True
            
            if done:
                reward -= 1
        
        self.last_info = copy.deepcopy(info)
        return state, reward, done, info
    
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