import gym

def registerEnv(id, **kwargs):
    entry_point = 'gym_super_mario_bros:SuperMarioBrosEnv'

    gym.envs.registration.register(
            id=id,
            entry_point=entry_point,
            max_episode_steps=9999999,
            reward_threshold=9999999,
            kwargs=kwargs,
            nondeterministic=True,
    )

def registerMario2():
    #only worlds 1-4 actually work, allegedly
    try:
        for world in range(1, 5):
            for level in range(1, 5):
                envName = f'SuperMarioBros2-{world}-{level}-v0'
                registerEnv(envName, lost_levels=True, rom_mode='vanilla', target=(world, level))
                #print(f"Registered {envName}")
    except:
        pass