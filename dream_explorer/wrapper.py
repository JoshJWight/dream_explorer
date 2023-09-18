import numpy as np
import warnings
import dreamerv3
from dreamerv3 import embodied
import dreamerv3.ninjax as nj
import task_helpers
import types
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')



class ModelWrapper:
    def __init__(self, logdir, task, config):
        self.task = task
        self.config = config
        logdir = embodied.Path(config.logdir)
        step = embodied.Counter()

        self.env = task_helpers.env_for_task(task)
        self.env = dreamerv3.wrap_env(self.env, config)
        self.env = embodied.BatchEnv([self.env], parallel=False)

        self.agent = dreamerv3.Agent(self.env.obs_space, self.env.act_space, step, config)
        replay = embodied.replay.Uniform(
            config.batch_length, config.replay_size, logdir / 'replay')
        args = embodied.Config(
            **config.run, logdir=config.logdir,
            batch_steps=config.batch_size * config.batch_length)
        
        checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
        checkpoint.step = step
        checkpoint.agent = self.agent
        checkpoint.replay = replay
        checkpoint.load(args.from_checkpoint)
        #End copypasta setup stuff

        #Because of all the jax wrapper stuff, we can't just fiddle with the agent's internals directly
        #Instead we'll inject some new functions into the agent. 

        def env_step(self, obs, state, action):
            if state is None:
                state, _ = self.wm.initial(1)
            obs = self.preprocess(obs)
            embed = self.wm.encoder(obs)
            context, _ = self.wm.rssm.obs_step(
                state, action, embed, obs['is_first'])
            return context, self.decode(context), self.metric_fun(context)

        def imag_step(self, state, action):
            prior = self.wm.rssm.img_step(state, action)
            return prior, self.decode(prior), self.metric_fun(prior)
        
        def imag_step_agent(self, state):
            action = self.get_action(state)
            return self.imag_step(state, action)

        def decode(self, state):
            recon = self.wm.heads['decoder'](state)
            result = {}
            for key in self.wm.heads['decoder'].cnn_shapes.keys():
                result[key] = recon[key].mode()
            return result
        
        def metric_fun(self, state):
            reward = self.wm.heads['reward'](state).mean()
            value = self.task_behavior.ac.critics['extr'].net(state).mean()
            cont = self.wm.heads['cont'](state).mean()
            return reward, value, cont
        
        def get_initial_state(self):
            state, _ = self.wm.initial(1)
            return state
        
        def get_action(self, state):
            action = self.task_behavior.policy(state, None)[0]['action']
            action = action.sample(seed=nj.rng())
            return action


        
        self.agent.agent.env_step = types.MethodType(env_step, self.agent.agent)
        self.agent.agent.imag_step = types.MethodType(imag_step, self.agent.agent)
        self.agent.agent.imag_step_agent = types.MethodType(imag_step_agent, self.agent.agent)
        self.agent.agent.decode = types.MethodType(decode, self.agent.agent)
        self.agent.agent.metric_fun = types.MethodType(metric_fun, self.agent.agent)
        self.agent.agent.get_initial_state = types.MethodType(get_initial_state, self.agent.agent)
        self.agent.agent.get_action = types.MethodType(get_action, self.agent.agent)

        kw = dict(device=self.agent.policy_devices[0])
        self._env_step = nj.pure(self.agent.agent.env_step)
        self._env_step = nj.jit(self._env_step, **kw)
        self._imag_step = nj.pure(self.agent.agent.imag_step)
        self._imag_step = nj.jit(self._imag_step, **kw)
        self._imag_step_agent = nj.pure(self.agent.agent.imag_step_agent)
        self._imag_step_agent = nj.jit(self._imag_step_agent, **kw)
        self._get_initial_state = nj.pure(self.agent.agent.get_initial_state)
        self._get_initial_state = nj.jit(self._get_initial_state, **kw)
        self._get_action = nj.pure(self.agent.agent.get_action)
        self._get_action = nj.jit(self._get_action, **kw)
        self.input_state = None

        self.obs = self.env.step({"action": [task_helpers.action_for_task(self.task, task_helpers.empty_key_map())], "reset": [True]})

        self.steps = 0
        self.use_env = True

        self.env_only = False
        self.agent_policy = False

    def reset(self):
        self.obs = self.env.step({"action": [task_helpers.action_for_task(self.task, task_helpers.empty_key_map())], "reset": [True]})
        self.input_state = None
        self.use_env = True
        self.steps = 0

    def set_level(self, level):
        if self.task == "mario" or self.task == "mario_random":
            self.env = task_helpers.mario_env_sequential(level)
        elif self.task == "atari":
            self.env = task_helpers.atari_env(level)
        elif self.task == "doom":
            self.env = task_helpers.doom_env(level)
        else:
            print("Level setting not supported for this task")
            return
        self.env = dreamerv3.wrap_env(self.env, self.config)
        self.env = embodied.BatchEnv([self.env], parallel=False)
                
    #agent_policy is only available if using the env
    def step(self, action):
        rng = self.agent._next_rngs(self.agent.policy_devices)
        varibs = self.agent.varibs if self.agent.single_device else self.agent.policy_varibs
        

        action = np.array([action])
        self.steps += 1
        if self.steps > 10:
            self.use_env = False
        
        
        if self.input_state is None:
            state_jax, _ = self._get_initial_state(varibs, rng)
        else:
            state_jax = self.agent._convert_inps((self.input_state), self.agent.policy_devices)

        if self.agent_policy:
            action_jax, _ = self._get_action(varibs, rng, state_jax)
            action = self.agent._convert_outs(action_jax, self.agent.policy_devices)
        else:
            action_jax = self.agent._convert_inps(action, self.agent.policy_devices)
            
        self.obs = self.env.step({"action": action, "reset": [False]})
        


        if self.use_env or self.env_only:
            if not self.obs['reward'][0] == 0.0:
                print(f"Reward {self.obs['reward'][0]}")
            obs_jax = self.agent._convert_inps(self.obs, self.agent.policy_devices)
            
            results, _ = self._env_step(varibs, rng, obs_jax, state_jax, action_jax)
        else:
            results, _ = self._imag_step(varibs, rng, state_jax, action_jax)

        self.input_state = self.agent._convert_outs(results[0], self.agent.policy_devices)
        img = self.agent._convert_outs(results[1], self.agent.policy_devices)

        image = img["image"][0]
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)

        reward, value, cont = self.agent._convert_outs(results[2], self.agent.policy_devices)

        metrics = {}
        metrics["EnvReward"] = self.obs['reward'][0]
        metrics["EnvContinue"] = not self.obs['is_terminal'][0]
        metrics["ImagReward"] = reward[0]
        metrics["ImagValue"] = value[0]
        metrics["ImagContinue"] = cont[0]

        env_image = self.obs["image"][0]

        return env_image, image, metrics, action[0]