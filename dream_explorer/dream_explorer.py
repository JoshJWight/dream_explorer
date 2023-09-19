import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

import dreamerv3
import dreamerv3.embodied as embodied

import dream_explorer.wrapper as wrapper
import dream_explorer.ui as ui

def run(module, logdir):

    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs[module.size()])
    config = config.update({
        'logdir': logdir,
    })
    config = embodied.Flags(config).parse(argv=[])

    mywrapper = wrapper.ModelWrapper(module, config)


    win = ui.GameWindow(module, mywrapper)
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()




