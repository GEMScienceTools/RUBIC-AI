from PyQt5 import QtWidgets, QtCore

class EpochSelectionDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window  # Reference to the main window (GUIInterface)
        
        # Get screen resolution
        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        # Scale the GUI based on resolution
        sf_x = screen_width / 1920
        sf_y = screen_height / 1080

        # Window Title
        self.setWindowTitle("Epoch of construction values")
        self.resize(int(400*sf_x), int(300*sf_y))

        # Main layout
        self.layout = QtWidgets.QVBoxLayout(self)

        # Dropdown to choose number of epochs
        self.layout.addWidget(QtWidgets.QLabel("Select number of epochs:"))
        self.epoch_count_cb = QtWidgets.QComboBox()
        self.epoch_count_cb.addItems([str(i) for i in range(2, 7)])  # Always at least 2
        self.epoch_count_cb.currentIndexChanged.connect(self.update_fields)
        self.layout.addWidget(self.epoch_count_cb)

        # Container for input fields
        self.inputs_container = QtWidgets.QWidget()
        self.inputs_layout = QtWidgets.QFormLayout(self.inputs_container)
        self.layout.addWidget(self.inputs_container)

        # Create input fields for epochs
        self.epoch_inputs = []
        for i in range(6):
            line_edit = QtWidgets.QLineEdit()

            # Set placeholders for first two fields
            if i == 0:
                line_edit.setPlaceholderText("Y:1990-2000")
            elif i == 1:
                line_edit.setPlaceholderText("Y:>1990 or Y:<1990")

            self.inputs_layout.addRow(f"Epoch {i+1}:", line_edit)
            self.epoch_inputs.append(line_edit)

        # OK / Cancel buttons
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.layout.addWidget(self.button_box)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        # Set default visibility
        self.update_fields()

    def update_fields(self):
        count = int(self.epoch_count_cb.currentText())

        # Always show first 2 inputs
        for i, line_edit in enumerate(self.epoch_inputs):
            visible = i < count
            if i < 2:
                visible = True
            line_edit.setVisible(visible)
            self.inputs_layout.labelForField(line_edit).setVisible(visible)

    def get_epochs(self):
        count = int(self.epoch_count_cb.currentText())
        return [self.epoch_inputs[i].text() for i in range(count)]
