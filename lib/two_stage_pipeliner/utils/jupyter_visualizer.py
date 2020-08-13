from typing import List, Any

import ipywidgets as widgets
from IPython.display import display, clear_output, Image
import matplotlib.pyplot as plt


class JupyterVisualizer:
    def __init__(self,
                 images,
                 images_names,
                 display_fn=lambda x: display(Image(x)),
                 choices: List[Any] = []):
        self.current_image = 0
        self.display_fn = display_fn
        self.out = widgets.Output()
        self.images_dropdown = widgets.Dropdown(
            options=list(zip(images_names, images)),
            description='Image:'
        )
        self.choices = widgets.Dropdown(
            options=choices,
            description='Choice:'
        )
        self.choices.change_options = False
        self.hbox = widgets.HBox(children=(self.images_dropdown, self.choices))
        self.vbox = widgets.VBox(children=(self.hbox, self.out))

    def visualize(self):
        display(self.vbox)

        @self.out.capture()
        def update_out(change):
            with self.out:
                clear_output(wait=True)
                self.display_fn(self.images_dropdown.value)
                plt.show()

        @self.out.capture()
        def on_change(change):
            if not self.choices.change_options:
                update_out(self)

        self.images_dropdown.observe(update_out, names=['value'])
        self.choices.observe(on_change, names=['value'])
        update_out(self)
