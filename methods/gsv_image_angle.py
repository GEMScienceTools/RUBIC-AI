from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import numpy as np

class gsv_angle_setting(QtWidgets.QDialog):
    def __init__(self, parent=None, main_window=None, gui_methods=None):
        super().__init__(parent)
        self.main_window = main_window
        self.gui_methods = gui_methods
        
        """Get screen resolution to adapt to different screen sizes"""
        # Get screen resolution
        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        #
        DESIGN_WIDTH = 1920
        DESIGN_HEIGHT = 1080
        DESIGN_DPI = 96 * 1.25  # 125% Windows baseline -> 120 DPI
        
        # Scale the GUI based on resolution
        sf_x = screen_width / DESIGN_WIDTH
        sf_y = screen_height / DESIGN_HEIGHT
        sf_factor = np.sqrt(sf_x * sf_y)

        # DPI-based scale
        # Get a reliable DPI value
        if sys.platform.startswith("win"):
            # Windows: use ctypes to get real DPI
            import ctypes
            LOGPIXELSX = 88
            hdc = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, LOGPIXELSX)
            ctypes.windll.user32.ReleaseDC(0, hdc)
        else:
            # macOS / Linux: start with logical DPI
            dpi = screen.logicalDotsPerInch()
            # If logical DPI looks weird, fallback to physical
            if dpi < 60 or dpi > 200:
                dpi = screen.physicalDotsPerInch()

        # Normalize to your design environment (Windows @ 125% = 120 DPI)
        # If dpi == 120 => scale_dpi = 1 (your original machine)
        scale_dpi = DESIGN_DPI / dpi
        
        # For geometry: mainly resolution-based
        sf_x = sf_factor
        sf_y = sf_factor
        # Scale the GUI based on resolution
        sf_font = sf_factor * scale_dpi
        
        self.setObjectName("GSV_angle_setting")
        self.resize(int(680 * sf_x), int(900 * sf_y))
        self.setWindowTitle("GSV image angle Setting")

        # === UI Elements Start ===
        self.gsv_angle_frame = QtWidgets.QWidget(self)
        self.gsv_angle_frame.setObjectName("gsv_angle_frame")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.gsv_angle_frame)

        self.save_button_gsv = QtWidgets.QPushButton(self.gsv_angle_frame)
        self.save_button_gsv.setGeometry(QtCore.QRect(int(240 * sf_x), int(830 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.save_button_gsv.setFont(font)
        self.save_button_gsv.setObjectName("save_button_gsv")
        self.save_button_gsv.setText("Save and continue")
        self.save_button_gsv.clicked.connect(self.save_and_close)
                                         
        self.backg_1 = QtWidgets.QLabel(self.gsv_angle_frame)
        self.backg_1.setGeometry(QtCore.QRect(int(10 * sf_x), int(10 * sf_y), int(631 * sf_x), int(811 * sf_y)))
        self.backg_1.setStyleSheet("background-color: rgb(255, 215, 255);")
        self.backg_1.setText("")
        self.backg_1.setObjectName("backg_1")
        
        self.gsv_label = QtWidgets.QLabel(self.gsv_angle_frame)
        self.gsv_label.setGeometry(QtCore.QRect(int(230 * sf_x), int(10 * sf_y), int(241 * sf_x), int(30 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(12 * sf_font))
        font.setBold(True)
        font.setItalic(True)
        font.setUnderline(True)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.gsv_label.setFont(font)
        self.gsv_label.setObjectName("gsv_label")
        self.gsv_label.setText("GSV image parameters")
        
        self.pitch_label = QtWidgets.QLabel(self.gsv_angle_frame)
        self.pitch_label.setGeometry(QtCore.QRect(int(20 * sf_x), int(39 * sf_y), int(51 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.pitch_label.setFont(font)
        self.pitch_label.setObjectName("pitch_label")
        self.pitch_label.setText("Pitch:")
        
        self.pitch_img = QtWidgets.QLabel(self.gsv_angle_frame)
        self.pitch_img.setGeometry(QtCore.QRect(int(20 * sf_x), int(170 * sf_y), int(611 * sf_x), int(361 * sf_y)))
        self.pitch_img.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.pitch_img.setObjectName("pitch_img")
        self.pitch_img.setPixmap(QtGui.QPixmap("help_img/pitch_angle.png"))
        self.pitch_img.setScaledContents(True)
        
        self.pitch_descrip = QtWidgets.QTextBrowser(self.gsv_angle_frame)
        self.pitch_descrip.setGeometry(QtCore.QRect(int(20 * sf_x), int(80 * sf_y), int(611 * sf_x), int(81 * sf_y)))
        self.pitch_descrip.setObjectName("pitch_descrip")

        # Scalable font size
        fs = int(10 * sf_font)

        self.pitch_descrip.setHtml(f"""
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">
<html><head><meta name="qrichtext" content="1" /><style type="text/css">
p, li {{ white-space: pre-wrap; }}
</style></head>
<body style=" font-family:'MS Shell Dlg 2'; font-size:{fs}pt; font-weight:400; font-style:normal;">
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;" align="justify">
<span style=" font-size:{fs}pt;">The </span>
<span style=" font-size:{fs}pt; font-weight:600;">pitch angle</span>
<span style=" font-size:{fs}pt;"> defines the </span>
<span style=" font-size:{fs}pt; font-weight:600;">vertical inclination of the camera</span>
<span style=" font-size:{fs}pt;">, allowing a full view of tall buildings. This angle is limited between </span>
<span style=" font-size:{fs}pt; font-weight:600;">0° and 60°</span>
<span style=" font-size:{fs}pt;">, with </span>
<span style=" font-size:{fs}pt; font-weight:600;">5° as the default value.</span>
</p>
</body></html>
""")

        self.pitch_value = QtWidgets.QSpinBox(self.gsv_angle_frame)
        self.pitch_value.setGeometry(QtCore.QRect(int(80 * sf_x), int(40 * sf_y), int(51 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.pitch_value.setFont(font)
        self.pitch_value.setMaximum(60)
        self.pitch_value.setProperty("value", 5)
        self.pitch_value.setObjectName("pitch_value")
        
        self.heading_label = QtWidgets.QLabel(self.gsv_angle_frame)
        self.heading_label.setGeometry(QtCore.QRect(int(20 * sf_x), int(539 * sf_y), int(81 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.heading_label.setFont(font)
        self.heading_label.setObjectName("heading_label")
        self.heading_label.setText("Heading:")
        
        self.heading_value = QtWidgets.QSpinBox(self.gsv_angle_frame)
        self.heading_value.setGeometry(QtCore.QRect(int(110 * sf_x), int(540 * sf_y), int(61 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.heading_value.setFont(font)
        self.heading_value.setMinimum(-180)
        self.heading_value.setMaximum(180)
        self.heading_value.setProperty("value", 0)
        self.heading_value.setObjectName("heading_value")
        
        self.heading_descrip = QtWidgets.QTextBrowser(self.gsv_angle_frame)
        self.heading_descrip.setGeometry(QtCore.QRect(int(20 * sf_x), int(580 * sf_y), int(611 * sf_x), int(81 * sf_y)))
        self.heading_descrip.setObjectName("heading_descrip")
        # Scalable font size
        fs = int(10 * sf_font)

        self.heading_descrip.setHtml(f"""
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">
<html><head><meta name="qrichtext" content="1" /><style type="text/css">
p, li {{ white-space: pre-wrap; }}
</style></head>
<body style=" font-family:'MS Shell Dlg 2'; font-size:{fs}pt; font-weight:400; font-style:normal;">
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;" align="justify">
<span style="font-size:{fs}pt;">The </span>
<span style="font-size:{fs}pt; font-weight:600;">horizontal camera angle</span>
<span style="font-size:{fs}pt;"> controls the direction of the camera, even allowing the view of a building on the opposite side of the street (180°). This angle is limited between </span>
<span style="font-size:{fs}pt; font-weight:600;">-180° and 180°</span>
<span style="font-size:{fs}pt;">, with </span>
<span style="font-size:{fs}pt; font-weight:600;">0° as the default value.</span>
</p>
</body></html>
""")
        
        self.fov_label = QtWidgets.QLabel(self.gsv_angle_frame)
        self.fov_label.setGeometry(QtCore.QRect(int(20 * sf_x), int(670 * sf_y), int(81 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.fov_label.setFont(font)
        self.fov_label.setObjectName("fov_label")
        self.fov_label.setText("FOV:")
        
        self.fov_value = QtWidgets.QSpinBox(self.gsv_angle_frame)
        self.fov_value.setGeometry(QtCore.QRect(int(70 * sf_x), int(670 * sf_y), int(61 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.fov_value.setFont(font)
        self.fov_value.setMinimum(1)
        self.fov_value.setMaximum(120)
        self.fov_value.setProperty("value", 120)
        self.fov_value.setObjectName("fov_value")
        
        self.fov_descrip = QtWidgets.QTextBrowser(self.gsv_angle_frame)
        self.fov_descrip.setGeometry(QtCore.QRect(int(20 * sf_x), int(710 * sf_y), int(611 * sf_x), int(101 * sf_y)))
        self.fov_descrip.setObjectName("fov_descrip")
        self.fov_descrip.setHtml(f"""
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">
<html><head><meta name="qrichtext" content="1" /><style type="text/css">
p, li {{ white-space: pre-wrap; }}
</style></head>
<body style=" font-family:'MS Shell Dlg 2'; font-size:{fs}pt; font-weight:400; font-style:normal;">
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;" align="justify">
<span style="font-size:{fs}pt;">The </span>
<span style="font-size:{fs}pt; font-weight:600;">field of view</span>
<span style="font-size:{fs}pt;"> controls the zoom of the camera. Smaller values </span>
<span style="font-size:{fs}pt; font-weight:600;">zoom in</span>
<span style="font-size:{fs}pt;">, while larger values </span>
<span style="font-size:{fs}pt; font-weight:600;">zoom out</span>
<span style="font-size:{fs}pt;">. By default, this value is set to </span>
<span style="font-size:{fs}pt; font-weight:600;">120</span>
<span style="font-size:{fs}pt;">, which is the maximum allowed. This is useful for creating a natural zoom effect without losing too much resolution for distant building images.</span>
</p>
</body></html>
""")

    def save_and_close(self):
        self.accept()
