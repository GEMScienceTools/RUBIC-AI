"""Provide a dialog for entering building construction epochs.

The dialog allows users to select the number of construction epochs and enter
an identifying value for each epoch using dynamically displayed input fields.
"""

import numpy as np
from PyQt5 import QtWidgets

_DESIGN_WIDTH = 1920
_DESIGN_HEIGHT = 1080
_MAX_EPOCHS = 6
_MIN_EPOCHS = 2


class EpochSelectionDialog(QtWidgets.QDialog):
    """Provide a dialog for selecting and entering construction epochs.

    Parameters
    ----------
    parent : PyQt5.QtWidgets.QWidget, optional
        Parent widget of the dialog.
    main_window : object, optional
        Reference to the main application window.

    Attributes
    ----------
    main_window : object or None
        Reference to the main application window.
    epoch_count_cb : PyQt5.QtWidgets.QComboBox
        Dropdown used to select the number of construction epochs.
    epoch_inputs : list[PyQt5.QtWidgets.QLineEdit]
        Input fields containing the construction-epoch values.
    """

    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window

        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        scale_x = screen_width / _DESIGN_WIDTH
        scale_y = screen_height / _DESIGN_HEIGHT
        scale_factor = np.sqrt(scale_x * scale_y)

        self.setWindowTitle("Epoch of construction values")
        self.resize(int(400 * scale_factor), int(300 * scale_factor))

        self.layout = QtWidgets.QVBoxLayout(self)

        self.layout.addWidget(QtWidgets.QLabel("Select number of epochs:"))
        self.epoch_count_cb = QtWidgets.QComboBox()
        self.epoch_count_cb.addItems(
            [str(index) for index in range(_MIN_EPOCHS, _MAX_EPOCHS + 1)]
        )
        self.epoch_count_cb.currentIndexChanged.connect(self.update_fields)
        self.layout.addWidget(self.epoch_count_cb)

        self.inputs_container = QtWidgets.QWidget()
        self.inputs_layout = QtWidgets.QFormLayout(self.inputs_container)
        self.layout.addWidget(self.inputs_container)

        self.epoch_inputs = []
        for index in range(_MAX_EPOCHS):
            line_edit = QtWidgets.QLineEdit()

            if index == 0:
                line_edit.setPlaceholderText("Y:1990-2000")
            elif index == 1:
                line_edit.setPlaceholderText("Y:>1990 or Y:<1990")

            self.inputs_layout.addRow(f"Epoch {index + 1}:", line_edit)
            self.epoch_inputs.append(line_edit)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.layout.addWidget(self.button_box)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.update_fields()

    def update_fields(self):
        """Update the visibility of the construction-epoch input fields.

        Display the number of fields selected in the dropdown while ensuring
        that at least the first two input fields remain visible.
        """
        count = int(self.epoch_count_cb.currentText())
        visible_count = max(count, _MIN_EPOCHS)

        for index, line_edit in enumerate(self.epoch_inputs):
            is_visible = index < visible_count
            line_edit.setVisible(is_visible)
            self.inputs_layout.labelForField(line_edit).setVisible(is_visible)

    def get_epochs(self):
        """Return the construction-epoch values entered by the user."""
        count = int(self.epoch_count_cb.currentText())
        return [self.epoch_inputs[index].text() for index in range(count)]
