import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

import dreamerv3
import dreamerv3.embodied as embodied

import wrapper
import ui

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--env_only', action='store_true')
    parser.add_argument('--agent_policy', action='store_true')
    args = parser.parse_args()

    sizes = {}
    for x in ['crafter', 'mspacman', 'pong', 'skiing']:
        sizes[x] = 'medium'
    
    for x in ['mario', 'seaquest']:
        sizes[x] = 'xlarge'
    
    #TODO can we derive the size from the logdir?
    assert(args.task in sizes)

    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs[sizes[args.task]])
    config = config.update({
        'logdir': args.logdir,
        'run.train_ratio': 64,
        'run.log_every': 30,  # Seconds
        'batch_size': 16,
        'jax.prealloc': False,
        'encoder.mlp_keys': '$^',
        'decoder.mlp_keys': '$^',
        'encoder.cnn_keys': 'image',
        'decoder.cnn_keys': 'image',
        # 'jax.platform': 'cpu',
    })
    config = embodied.Flags(config).parse(argv=[])

    mywrapper = wrapper.ModelWrapper(args.logdir, args.task, config, args.env_only, args.agent_policy)


    win = ui.GameWindow(args.task, mywrapper)
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()