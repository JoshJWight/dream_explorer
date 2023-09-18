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
    args = parser.parse_args()

    if args.task is None or args.logdir is None:
        print("Usage: main.py --task <taskname> --logdir <logdir>")
        exit(1)

    sizes = {}
    for x in ['crafter', 'mspacman', 'pong', 'skiing']:
        sizes[x] = 'medium'
    
    for x in ['mario', 'mario_random', 'seaquest', 'atari', 'doom', 'spelunky']:
        sizes[x] = 'xlarge'
    
    #TODO can we derive the size from the logdir?
    assert(args.task in sizes)

    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs[sizes[args.task]])
    config = config.update({
        'logdir': args.logdir,
    })
    config = embodied.Flags(config).parse(argv=[])

    mywrapper = wrapper.ModelWrapper(args.logdir, args.task, config)


    win = ui.GameWindow(args.task, mywrapper)
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()