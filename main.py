import sys
from PyQt5.QtWidgets import QApplication
from methods.gui_interface import GUIInterface

# Run the graphical user interface by calling all functions.
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GUIInterface()
    window.show()
    
    # Run the GUI event loop
    app.exec_()