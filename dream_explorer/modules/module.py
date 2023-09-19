import numpy as np

#Template for modules
class Module:
    def __init__(self):
        pass

    #Should return the list of options used in level select
    def levels(self):
        return []
    
    #Should return a dreamerv3 env
    def create_env(self):
        pass

    #Should return an env using the given level (which could be the input one after modification)
    #Level is an unsigned integer representing an index into the list returned by levels()
    def set_level(self, level):
        pass

    #Should return dreamerv3 model size for this task
    def size(self):
        return 'xlarge'
    
    #Should return a list of lists of keys associated with each action
    def action_keys(self):
        return []
    



    #Does not need to be overridden
    #Note: this produces a one-hot vector of the action
    def action_for_keys(self, key_map):
        task_keys = self.action_keys()
        action = np.zeros(len(task_keys))
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

#Helper functions
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
        "c": False,
        "Shift_L": False,
        "Z": False,
    }
    
