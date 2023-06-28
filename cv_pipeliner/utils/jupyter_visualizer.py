from typing import List, Any

import ipywidgets as widgets
from IPython.display import display, clear_output, Image
import matplotlib.pyplot as plt


class JupyterVisualizer:
    def __init__(
        self,
        images,
        images_names,
        display_fn=lambda x: display(Image(x)),
        choices: List[Any] = [],
        choices_description: str = "Choice:",
        choices2: List[Any] = [],
        choices2_description: str = "Choice2:",
    ):
        self.current_image = 0
        self.display_fn = display_fn
        self.out = widgets.Output()
        self.images_dropdown = widgets.Dropdown(options=list(zip(images_names, images)), description="Image:")
        hbox_children = [self.images_dropdown]
        if choices:
            self.choices = widgets.Dropdown(options=choices, description=choices_description)
            self.choices.change_options = False
            hbox_children.append(self.choices)
        else:
            self.choices = None
        if choices2:
            self.choices2 = widgets.Dropdown(options=choices2, description=choices2_description)
            self.choices2.change_options = False
            hbox_children.append(self.choices2)
        else:
            self.choices2 = None
        self.hbox = widgets.HBox(children=tuple(hbox_children))
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

        @self.out.capture()
        def on_change2(change):
            if not self.choices2.change_options:
                update_out(self)

        self.images_dropdown.observe(update_out, names=["value"])
        if self.choices:
            self.choices.observe(on_change, names=["value"])
        if self.choices2:
            self.choices2.observe(on_change2, names=["value"])
        update_out(self)
