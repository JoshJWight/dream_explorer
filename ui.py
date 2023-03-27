import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from gi.repository import GdkPixbuf
from gi.repository import GLib
from gi.repository import GObject
from gi.repository import Gdk

import numpy as np

import wrapper
import task_helpers

def set_image(gtk_image, source_image):
    h, w, c = source_image.shape

    image = GdkPixbuf.Pixbuf.new_from_bytes(GLib.Bytes(source_image.tobytes()), GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * c)
    image = image.scale_simple(500, 500, GdkPixbuf.InterpType.BILINEAR)
    gtk_image.set_from_pixbuf(image)

class GameWindow(Gtk.Window):
    def __init__(self, task, mywrapper):
        Gtk.Window.__init__(self, title="Game Window")
        self.set_default_size(800, 600)
        self.set_border_width(10)
        self.set_position(Gtk.WindowPosition.CENTER)

        self.box = Gtk.Box(spacing=6)
        self.add(self.box)

        self.connect("key-press-event", self.on_key_press_event)
        self.connect("key-release-event", self.on_key_release_event)

        source_image = np.zeros((64, 64, 3), dtype=np.uint8)
        self.env_image = Gtk.Image()
        set_image(self.env_image, source_image)
        self.box.pack_start(self.env_image, True, True, 0)
        self.image = Gtk.Image()
        set_image(self.image, source_image)
        self.box.pack_start(self.image, True, True, 0)

        self.widget_box = Gtk.VBox()
        self.box.pack_start(self.widget_box, True, True, 0)

        #Reset button
        self.button = Gtk.Button(label="Reset")
        self.button.connect("clicked", self.on_reset_clicked)
        self.widget_box.pack_start(self.button, True, True, 0)

        #Framerate label
        self.framerate_label = Gtk.Label(label="Framerate (log 10)")
        self.widget_box.pack_start(self.framerate_label, True, True, 0)

        #Framerate slider
        self.framerate_slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 3, 1)
        self.framerate_slider.set_digits(2)
        self.framerate_slider.set_value(1.7)
        self.widget_box.pack_start(self.framerate_slider, True, True, 0)

        #Agent policy checkbox
        self.agent_policy_checkbox = Gtk.CheckButton(label="Agent Policy")
        self.agent_policy_checkbox.connect("toggled", self.on_agent_policy_toggled)
        self.widget_box.pack_start(self.agent_policy_checkbox, True, True, 0)

        #Observe Environment checkbox
        self.observe_env_checkbox = Gtk.CheckButton(label="Observe Environment")
        self.observe_env_checkbox.connect("toggled", self.on_observe_env_toggled)
        self.widget_box.pack_start(self.observe_env_checkbox, True, True, 0)

        self.playback_box = Gtk.VBox()
        self.box.pack_start(self.playback_box, True, True, 0)

        #Play/pause button
        self.play_button = Gtk.Button(label="Pause/Play")
        self.play_button.connect("clicked", self.on_play_clicked)
        self.playback_box.pack_start(self.play_button, True, True, 0)

        #Step button
        self.step_button = Gtk.Button(label="Step")
        self.step_button.connect("clicked", self.on_step_clicked)
        self.playback_box.pack_start(self.step_button, True, True, 0)

        self.metrics_box = Gtk.VBox()
        self.box.pack_start(self.metrics_box, True, True, 0)
        
        self.metrics_grid = Gtk.Grid()
        self.metrics_box.pack_start(self.metrics_grid, True, True, 0)

        #Reward
        self.reward_label = Gtk.Label(label="Imagined Reward")
        self.metrics_grid.attach(self.reward_label, 0, 0, 1, 1)
        self.reward_value = Gtk.Label(label="0")
        self.metrics_grid.attach(self.reward_value, 1, 0, 1, 1)     
        #Value
        self.value_label = Gtk.Label(label="Critic Value")
        self.metrics_grid.attach(self.value_label, 0, 1, 1, 1)
        self.value_value = Gtk.Label(label="0")
        self.metrics_grid.attach(self.value_value, 1, 1, 1, 1)
        #Continue
        self.continue_label = Gtk.Label(label="Imagined Continue")
        self.metrics_grid.attach(self.continue_label, 0, 2, 1, 1)
        self.continue_value = Gtk.Label(label="0")
        self.metrics_grid.attach(self.continue_value, 1, 2, 1, 1)
        #Total Reward
        self.total_reward_label = Gtk.Label(label="Total Imagined Reward")
        self.metrics_grid.attach(self.total_reward_label, 0, 3, 1, 1)
        self.total_reward_value = Gtk.Label(label="0")
        self.metrics_grid.attach(self.total_reward_value, 1, 3, 1, 1)
        #Environment Reward
        self.env_reward_label = Gtk.Label(label="Environment Reward")
        self.metrics_grid.attach(self.env_reward_label, 0, 4, 1, 1)
        self.env_reward_value = Gtk.Label(label="0")
        self.metrics_grid.attach(self.env_reward_value, 1, 4, 1, 1)
        #Environment Total Reward
        self.env_total_reward_label = Gtk.Label(label="Environment Total Reward")
        self.metrics_grid.attach(self.env_total_reward_label, 0, 5, 1, 1)
        self.env_total_reward_value = Gtk.Label(label="0")
        self.metrics_grid.attach(self.env_total_reward_value, 1, 5, 1, 1)
        #Environment Continue
        self.env_continue_label = Gtk.Label(label="Environment Continue")
        self.metrics_grid.attach(self.env_continue_label, 0, 6, 1, 1)
        self.env_continue_value = Gtk.Label(label="0")
        self.metrics_grid.attach(self.env_continue_value, 1, 6, 1, 1)

        self.control_labels = []
        self.control_label_box = Gtk.VBox()
        self.box.pack_start(self.control_label_box, True, True, 0)

        for x in task_helpers.ACTION_KEYS[task]:
            label = Gtk.Label(label=str(x))
            self.control_label_box.pack_start(label, True, True, 0)
            self.control_labels.append(label)


        self.task = task
        self.play = True

        self.total_reward = 0
        self.total_env_reward = 0
        
        self.wrapper = mywrapper
        GObject.timeout_add(50, self.update)

        self.key_map = task_helpers.empty_key_map()

    def update(self):
        if self.play:
            self.step()
        
        delay = 1000 / (10 ** self.framerate_slider.get_value())

        GObject.timeout_add(delay, self.update)

    def step(self):
        action = task_helpers.action_for_task(self.task, self.key_map)

        env_source_image, source_image, metrics, action = self.wrapper.step(action)
        
        set_image(self.env_image, env_source_image)
        set_image(self.image, source_image)

        self.reward_value.set_text("{:.3f}".format(metrics["ImagReward"]))
        self.value_value.set_text("{:.3f}".format(metrics["ImagValue"]))
        self.continue_value.set_text("{:.3f}".format(float(metrics["ImagContinue"])))
        self.total_reward += metrics["ImagReward"]
        self.total_reward_value.set_text("{:.3f}".format(self.total_reward))
        self.env_reward_value.set_text("{:.3f}".format(metrics["EnvReward"]))
        self.env_continue_value.set_text("{:.3f}".format(float(metrics["EnvContinue"])))
        self.total_env_reward += metrics["EnvReward"]
        self.env_total_reward_value.set_text("{:.3f}".format(self.total_env_reward))

        for i, label in enumerate(self.control_labels):
            if action[i] > 0:
                #set color to yellow
                label.set_markup("<span foreground='green'>%s</span>" % label.get_text())
            else:
                #set color to white
                label.set_markup("<span foreground='black'>%s</span>" % label.get_text())

        if metrics["ImagContinue"] < 0.5 or (self.wrapper.env_only and metrics["EnvContinue"] < 0.5):
            self.play = False

    
    def on_key_press_event(self, widget, event):
        keyname = Gdk.keyval_name(event.keyval)
        self.key_map[keyname] = True
    
    def on_key_release_event(self, widget, event):
        keyname = Gdk.keyval_name(event.keyval)
        self.key_map[keyname] = False

    def on_reset_clicked(self, widget):
        self.play = True
        self.total_reward = 0
        self.total_env_reward = 0
        self.wrapper.reset()

    def on_agent_policy_toggled(self, widget):
        self.wrapper.agent_policy = self.agent_policy_checkbox.get_active()

    def on_observe_env_toggled(self, widget):
        self.wrapper.env_only = self.observe_env_checkbox.get_active()

    def on_play_clicked(self, widget):
        self.play = not self.play

    def on_step_clicked(self, widget):
        self.step()