import numpy as np
import warnings
import dreamerv3
from dreamerv3 import embodied
import dreamerv3.ninjax as nj
import task_helpers
import types
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')



class ModelWrapper:
    def __init__(self, logdir, task, config, env_only=False, agent_policy=False):
        logdir = embodied.Path(config.logdir)
        step = embodied.Counter()
        logger = embodied.Logger(step, [
            embodied.logger.TerminalOutput(),
            embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
            embodied.logger.TensorBoardOutput(logdir),
            # embodied.logger.WandBOutput(logdir.name, config),
            # embodied.logger.MLFlowOutput(logdir.name),
        ])

        self.env = task_helpers.env_for_task(task)
        self.env = dreamerv3.wrap_env(self.env, config)
        self.env = embodied.BatchEnv([self.env], parallel=False)

        self.agent = dreamerv3.Agent(self.env.obs_space, self.env.act_space, step, config)
        replay = embodied.replay.Uniform(
            config.batch_length, config.replay_size, logdir / 'replay')
        args = embodied.Config(
            **config.run, logdir=config.logdir,
            batch_steps=config.batch_size * config.batch_length)
        
        step = logger.step
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
            #latentb = {k: jnp.expand_dims(v, 0) for k, v in context.items()}
            return context, self.decode(context)

        def imag_step(self, state, action):
            prior = self.wm.rssm.img_step(state, action)
            return prior, self.decode(prior)

        def decode(self, state):
            recon = self.wm.heads['decoder'](state)
            result = {}
            for key in self.wm.heads['decoder'].cnn_shapes.keys():
                result[key] = recon[key].mode()
            return result
        
        self.agent.agent.env_step = types.MethodType(env_step, self.agent.agent)
        self.agent.agent.imag_step = types.MethodType(imag_step, self.agent.agent)
        self.agent.agent.decode = types.MethodType(decode, self.agent.agent)

        kw = dict(device=self.agent.policy_devices[0])
        self._env_step = nj.pure(self.agent.agent.env_step)
        self._env_step = nj.jit(self._env_step, **kw)
        self._imag_step = nj.pure(self.agent.agent.imag_step)
        self._imag_step = nj.jit(self._imag_step, **kw)

        self.input_state = None

        self.obs = self.env.step({"action": [task_helpers.action_for_task(task, task_helpers.empty_key_map())], "reset": [True]})

        self.steps = 0
        self.use_env = True

        self.env_only = env_only
        self.agent_policy = agent_policy
                
    #agent_policy is only available if using the env
    def step(self, action):
        action = np.array([action])
        self.steps += 1
        if self.steps > 10:
            self.use_env = False
        
        action_jax = self.agent._convert_inps(action, self.agent.policy_devices)
        if self.input_state is None:
            state_jax = None
        else:
            state_jax = self.agent._convert_inps((self.input_state), self.agent.policy_devices)
        rng = self.agent._next_rngs(self.agent.policy_devices)
        varibs = self.agent.varibs if self.agent.single_device else self.agent.policy_varibs

        if self.use_env or self.env_only:
            if self.agent_policy:
                outputs, self.input_state = self.agent.policy(self.obs, self.input_state, mode="eval")
                action = outputs['action']
            self.obs = self.env.step({"action": action, "reset": [False]})
            if self.env_only: #Just using the wrapper to run the env
                return self.obs["image"][0]
            obs_jax = self.agent._convert_inps(self.obs, self.agent.policy_devices)
            
            results, _ = self._env_step(varibs, rng, obs_jax, state_jax, action_jax)
            
            
            #data = {"action": np.asarray([[action]]), "image": np.asarray([[self.obs["image"]]]), "is_first": np.asarray([[self.obs["is_first"]]])}
            #embed = self.agent.agent.wm.encoder(data)

            #states, _ = self.agent.agent.wm.rssm.observe(embed, data['action'], data['is_first'], self.input_state)
        else:
            #data = {"action": np.asarray([[action]])}
            #states = self.agent.agent.wm.rssm.imagine(data['action'], self.input_state)
            results, _ = self._imag_step(varibs, rng, state_jax, action_jax)

        self.input_state = self.agent._convert_outs(results[0], self.agent.policy_devices)
        img = self.agent._convert_outs(results[1], self.agent.policy_devices)

        #plt.imshow(img["image"][0])

        #self.input_state = {k: v[:, -1] for k, v in states.items()}
        #decoder = self.agent.agent.wm.heads['decoder']
        #recon = decoder(self.agent.agent.wm.rssm.get_feat(states))["image"].mode()

        #recon_np = recon[0][0].numpy()
        #h, w, _ = recon_np.shape
        #stack 3 copies together
        #result = np.zeros((h, w, 3), dtype=np.uint8)
        #for i in range(3):
        #    result[:, :, i] = (recon_np[:, :, 0] + 0.5) * 255

        image = img["image"][0]
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)

        return image