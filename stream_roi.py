#mjpeg stream temp not alligned
import sys
import os
import signal
import logging
import threading
import numpy as np
import cv2 as cv
from flask import Flask, Response
from senxor.mi48 import MI48, format_header, format_framestats
from senxor.utils import data_to_frame, remap, cv_filter, RollingAverageFilter, connect_senxor

# Enable logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class ThermalCamera:
    def __init__(self, roi=(0, 0, 61, 61), com_port=None):
        """
        Initializes the thermal camera with a given ROI and optional COM port.
        Runs in a separate thread.
        """
        self.roi = roi  # (x1, y1, x2, y2)
        self.com_port = com_port    
        self.running = True
        self.latest_frame = None
        self.lock = threading.Lock()
        
        # Connect to the MI48 camera
        self.mi48, self.connected_port, _ = connect_senxor(src=self.com_port) if self.com_port else connect_senxor()

        logger.info(f"Camera initialized on {self.connected_port}")

        # Set camera settings
        self.mi48.set_fps(15)
        self.mi48.disable_filter(f1=True, f2=True, f3=True)
        self.mi48.set_filter_1(85)
        self.mi48.enable_filter(f1=True, f2=False, f3=False, f3_ks_5=False)
        self.mi48.set_offset_corr(0.0)
        self.mi48.set_sens_factor(100)
        
        # Start streaming
        self.mi48.start(stream=True, with_header=True)

        # Filters for stability
        self.dminav = RollingAverageFilter(N=10)
        self.dmaxav = RollingAverageFilter(N=10)

        # Start thread
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        """Runs the camera processing loop asynchronously."""
        while self.running:
            self.process_frame()

    def process_frame(self):
        """Processes a frame: crops ROI, calculates temperatures, overlays text."""
        data, header = self.mi48.read()
        if data is None:
            logger.error("No data received from the camera.")
            return

        # Calculate min/max temperatures
        min_temp = self.dminav(data.min())
        max_temp = self.dmaxav(data.max())
 
        # Convert raw data to an image frame
        frame = data_to_frame(data, (80, 62), hflip= True ) #hflip verticle flip

        # Manually flip the frame along the vertical axis
        frame = cv.flip(frame, 1)  # Flip along the Y-axis

        # Rotate 90 degrees clockwise
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)

        frame = np.clip(frame, min_temp, max_temp)

        # Apply filters
        filt_frame = cv_filter(remap(frame), {'blur_ks': 3, 'd': 5, 'sigmaColor': 27, 'sigmaSpace': 27}, 
                               use_median=True, use_bilat=True, use_nlm=False)  

        # Crop to ROI
        x1, y1, x2, y2 = self.roi
        roi_frame = filt_frame[y1:y2, x1:x2]

        #  Apply thermal color mapping
        roi_frame = cv.applyColorMap(roi_frame, cv.COLORMAP_INFERNO)  # Use thermal color mapping


        # Resize the frame to make it larger
        roi_frame = cv.resize(roi_frame, (600, 600), interpolation=cv.INTER_LINEAR)


        # Calculate section temperatures
        temps = self.calculate_temperatures(frame, x1, y1, x2, y2)

        # Overlay text on the image
        self.overlay_text(roi_frame, temps)

        # Store the latest frame for streaming
        with self.lock:
            self.latest_frame = roi_frame


    def calculate_temperatures(self, frame, x1, y1, x2, y2):
        """Calculates the average temperatures for 5 sections: Top, Bottom, Left, Right, Center."""
        w, h = x2 - x1, y2 - y1
        section_w, section_h = w // 3, h // 3

        sections = {
            "Top": frame[y1:y1+section_h, x1:x2],
            "Bottom": frame[y2-section_h:y2, x1:x2],
            "Left": frame[y1:y2, x1:x1+section_w],
            "Right": frame[y1:y2, x2-section_w:x2],
            "Center": frame[y1+section_h:y2-section_h, x1+section_w:x2-section_w]
        }

        return {name: np.mean(region) for name, region in sections.items()}

    def overlay_text(self, frame, temps):
        """Overlays temperature values on the image."""
        positions = {
            "Top": (10, 15), # tept display values
            "Bottom": (10, frame.shape[0] - 10),
            "Left": (5, frame.shape[0] // 2),
            "Right": (frame.shape[1] - 40, frame.shape[0] // 2),
            "Center": (frame.shape[1] // 2 - 20, frame.shape[0] // 2)
        }

        for section, temp in temps.items():
            x, y = positions[section]
            cv.putText(frame, f"{temp:.2f}C", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def start_stream(self):
        """Starts an MJPEG stream."""
        app = Flask(__name__)

        @app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

        app.run(host="0.0.0.0", port=5000, threaded=True)

    def generate_frames(self):
        """Generates frames for the MJPEG stream."""
        while self.running:
            with self.lock:
                if self.latest_frame is None:
                    continue
                _, buffer = cv.imencode('.jpg', self.latest_frame)
                frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def get_temperatures(self):
        """Returns the 5 temperature values."""
        return self.calculate_temperatures(self.latest_frame, *self.roi) if self.latest_frame is not None else {}

    def stop(self):
        """Stops the camera."""
        self.running = False
        self.mi48.stop()
        cv.destroyAllWindows()


# **Main Execution**
if __name__ == "__main__":
    # Define ROI (adjust as needed)
    roi = (0, 0, 100, 100)  # Large square ROI

    # Initialize thermal camera
    cam = ThermalCamera(roi=roi, com_port=None)

    # Start MJPEG streaming
    cam.start_stream()
