#Template for modules

class Module:
    def __init__(self):
        pass

    #Should return the list of options used in level select
    def levels(self):
        return []
    
    #Should return an env using the given level (which could be the input one after modification)
    def set_level(self, level, env):
        pass
    
    def env(self):
        pass

    def size(self):
        return 'xlarge'