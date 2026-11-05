import sys
import math
from urllib.parse import urlencode

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWebEngineWidgets import QWebEngineView


class StreetViewViewer(QtWidgets.QDialog):
    """Interactive Google Street View viewer for RUBIC-AI."""

    def __init__(
        self,
        latitude,
        longitude,
        api_key,
        heading=0,
        pitch=5,
        fov=90,
        parent=None
    ):
        super().__init__(parent)

        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.api_key = api_key
        self.heading = float(heading)
        self.pitch = float(pitch)
        self.fov = float(fov)

        if not (
            math.isfinite(self.latitude)
            and math.isfinite(self.longitude)
            and -90 <= self.latitude <= 90
            and -180 <= self.longitude <= 180
        ):
            raise ValueError("Invalid coordinates.")

        if not self.api_key:
            raise ValueError("Google Maps API key is required.")

        self.pitch = max(-90, min(self.pitch, 90))
        self.fov = max(10, min(self.fov, 100))
        self.heading = self.heading % 360

        self.setWindowTitle("RUBIC-AI | Street View")
        self.resize(950, 750)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.web_view = QWebEngineView(self)
        layout.addWidget(self.web_view)

        self.load_street_view()

    def load_street_view(self):
        """Load interactive Google Street View."""

        params = urlencode({
            "key": self.api_key,
            "location": (
                f"{self.latitude},{self.longitude}"
            ),
            "heading": self.heading,
            "pitch": self.pitch,
            "fov": self.fov,
        })

        url = (
            "https://www.google.com/maps/embed/v1/streetview?"
            + params
        )

        self.web_view.load(QtCore.QUrl(url))


if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)

    GOOGLE_MAPS_API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"

    viewer = StreetViewViewer(
        latitude=40.6405,
        longitude=-8.6538,
        api_key=GOOGLE_MAPS_API_KEY,
        heading=0,
        pitch=5,
        fov=90
    )

    viewer.show()

    sys.exit(app.exec_())
