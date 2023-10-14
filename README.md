# Dream Explorer

Dream Explorer is a GUI I made to look inside the world models of trained DreamerV3 agents. It uses the open-source DreamerV3 implemenation from http://github.com/danijar/dreamerv3.

The UI shows the original environment and the world model's reconstruction of it. You can play yourself or let the actor-critic agent play for you, and even detach from the original environment and let the world model extrapolate its own hallucination forever.

This is a personal project and I haven't really polished it for release.

## Basic usage pattern

Before you can observe a DreamerV3 agent, you need to train one first. The files in the `experiments/` directory are the training runs I've done so far.

```
python3 experiments/mario13.py
# Wait a while for training to occur...
python3 main.py --task mario --logdir ~/mariobros/13
```