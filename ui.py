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
        self.image = Gtk.Image()
        image = GdkPixbuf.Pixbuf.new_from_bytes(GLib.Bytes(source_image.tobytes()), GdkPixbuf.Colorspace.RGB, False, 8, 64, 64, 64 * 3)
        image = image.scale_simple(500, 500, GdkPixbuf.InterpType.BILINEAR)
        self.image.set_from_pixbuf(image)
        self.box.pack_start(self.image, True, True, 0)

        self.widget_box = Gtk.VBox()
        self.box.pack_start(self.widget_box, True, True, 0)

        #Reset button
        self.button = Gtk.Button(label="Reset")
        self.button.connect("clicked", self.on_reset_clicked)
        self.widget_box.pack_start(self.button, True, True, 0)

        #Framerate slider
        self.framerate_slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 1, 1000, 1)
        self.framerate_slider.set_value(50)
        self.widget_box.pack_start(self.framerate_slider, True, True, 0)

        #Agent policy checkbox
        self.agent_policy_checkbox = Gtk.CheckButton(label="Agent Policy")
        self.agent_policy_checkbox.connect("toggled", self.on_agent_policy_toggled)
        self.widget_box.pack_start(self.agent_policy_checkbox, True, True, 0)

        self.task = task
        
        self.wrapper = mywrapper
        GObject.timeout_add(50, self.update)

        self.key_map = task_helpers.empty_key_map()

    def update(self):
        action = task_helpers.action_for_task(self.task, self.key_map)
        #print(action)
        source_image = self.wrapper.step(action)
        h, w, c = source_image.shape

        image = GdkPixbuf.Pixbuf.new_from_bytes(GLib.Bytes(source_image.tobytes()), GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * c)
        image = image.scale_simple(500, 500, GdkPixbuf.InterpType.BILINEAR)
        self.image.set_from_pixbuf(image)
        
        delay = 1000 / self.framerate_slider.get_value()

        GObject.timeout_add(delay, self.update)

    
    def on_key_press_event(self, widget, event):
        keyname = Gdk.keyval_name(event.keyval)
        self.key_map[keyname] = True
    
    def on_key_release_event(self, widget, event):
        keyname = Gdk.keyval_name(event.keyval)
        self.key_map[keyname] = False

    def on_reset_clicked(self, widget):
        self.wrapper.reset()

    def on_agent_policy_toggled(self, widget):
        self.wrapper.agent_policy = self.agent_policy_checkbox.get_active()

if __name__ == "__main__":
    print("DON'T RUN THIS FILE DIRECTLY!")