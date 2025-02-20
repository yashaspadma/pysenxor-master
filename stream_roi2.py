#mjpeg feed, 


import sys
import os
import signal
import logging
import threading                        # Runs the camera processing in a separate background thread
import numpy as np                      # calculations
import cv2 as cv                        # img proc grid, txts
from flask import Flask, Response
from senxor.mi48 import MI48, format_header, format_framestats   # Connects and communicates with the MI48 thermal camera
from senxor.utils import data_to_frame, remap, cv_filter, RollingAverageFilter, connect_senxor

# Enable logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

#camera connection, image processing, and video streaming
class ThermalCamera:            
    def __init__(self, roi=(0, 0, 61, 61), com_port=None):        # Uses the default (0, 0, 61, 61) unless overridden
        """
        Initializes the thermal camera with a given ROI and optional COM port.
        Runs in a separate thread.
        """
        self.roi = roi  # (x1, y1, x2, y2)       #cam FOV crop
        self.com_port = com_port                 #cam com
        self.running = True
        self.latest_frame = None
        self.lock = threading.Lock()

        self.temps = {"Top": 0, "Bottom": 0, "Left": 0, "Right": 0, "Center": 0}  #to store the avg values to access on main

        
        # Connect to the MI48 camera detectsauto
        self.mi48, self.connected_port, _ = connect_senxor(src=self.com_port) if self.com_port else connect_senxor()

        logger.info(f"Camera initialized on {self.connected_port}")

        # Set camera settings
        self.mi48.set_fps(25)                                             # Set Frames Per Second (FPS)  15-->25
        self.mi48.disable_filter(f1=True, f2=True, f3=True)               # Disable noise filters
        self.mi48.set_filter_1(85)                                        # Adjust internal filter settings
        self.mi48.enable_filter(f1=True, f2=False, f3=False, f3_ks_5=False)
        self.mi48.set_offset_corr(0.0)                                    # Set offset correction
        self.mi48.set_sens_factor(100)                                    # Adjust sensitivity
        
        # Start streaming
        self.mi48.start(stream=True, with_header=True)

        # Filters for stability
        self.dminav = RollingAverageFilter(N=10)
        self.dmaxav = RollingAverageFilter(N=10)

        # Start thread
        self.thread = threading.Thread(target=self.run, daemon=True)     #Runs the process_frame() function continuously in a separate thread.
        self.thread.start()

    def run(self):
        """Runs the camera processing loop asynchronously."""
        while self.running:
            self.process_frame()

    def process_frame(self):
        """Processes a frame: crops ROI, calculates temperatures, overlays grid and text."""
        data, header = self.mi48.read()
        if data is None:
            logger.error("No data received from the camera.")
            return

        # Calculate min/max temperatures
        min_temp = self.dminav(data.min())
        max_temp = self.dmaxav(data.max())

        # Convert raw data to an image frame
        frame = data_to_frame(data, (80, 62), hflip=True)  #hflip verticle flip
        frame = np.clip(frame, min_temp, max_temp)

        # Manually flip the frame along the vertical axis
        frame = cv.flip(frame, 1)  # Flip along the Y-axis

        # Rotate 90 degrees clockwise
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)


        # Apply filters
        filt_frame = cv_filter(remap(frame), {'blur_ks': 3, 'd': 5, 'sigmaColor': 27, 'sigmaSpace': 27},  #Remaps temperature values for visualization
                               use_median=True, use_bilat=True, use_nlm=False)                            #Applies smoothing filters to reduce noise.

        # Crop to ROI
        x1, y1, x2, y2 = self.roi
        roi_frame = filt_frame[y1:y2, x1:x2]

        #  Apply thermal color mapping
        roi_frame = cv.applyColorMap(roi_frame, cv.COLORMAP_INFERNO)  #cv.COLORMAP_INFERNO)  

        #  Draw the 3×3 grid
        self.draw_grid(roi_frame)

        # Resize the frame to make it larger
        roi_frame = cv.resize(roi_frame, (600, 600), interpolation=cv.INTER_LINEAR)

        # Calculate section temperatures
        temps = self.calculate_temperatures(frame, x1, y1, x2, y2)

        # Overlay text on the image
        self.overlay_text(roi_frame, temps)

        # Store the latest frame for streaming
        with self.lock:
            self.latest_frame = roi_frame

    def draw_grid(self, frame):
        """Draws a 3×3 grid overlay on the thermal feed."""
        h, w = frame.shape[:2]  # Get frame dimensions
        step_w, step_h = w // 3, h // 3  # Divide width and height into 3 sections to get 3x3 grid

        # Dotted grid parameters
        dot_length = 6 # Length of each dot segment
        dot_gap = 12  # Gap between dots

        # Draw vertical lines
        for i in range(1, 3):
            x = i * step_w  
            for y in range(0, h, dot_length + dot_gap):  # Draw dots along vertical line
                cv.line(frame, (x, y), (x, min(y + dot_length, h)), (255, 255, 255), 1)

        # Draw horizontal lines
        for i in range(1, 3):
            y = i * step_h
            for x in range(0, w, dot_length + dot_gap):  # Draw dots along horizontal line
                cv.line(frame, (x, y), (min(x + dot_length, w), y), (255, 255, 255), 1)
        
        # Blend overlay with original frame to reduce visibility          ------------------------------------
        #alpha = 5 # Adjust transparency (lower = more transparent)
        #cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)        #-------------------------------------

    def calculate_temperatures(self, frame, x1, y1, x2, y2):
        """Calculates the average temperatures for 5 sections: Top, Bottom, Left, Right, Center."""
        w, h = x2 - x1, y2 - y1
        section_w, section_h = w // 3, h // 3  # Divide into 3x3 grid

        # Define sections
        sections = {
            "Top": frame[y1:y1+section_h, x1:x2],  # Top 3 squares
            "Bottom": frame[y2-section_h:y2, x1:x2],  # Bottom 3 squares
            "Left": frame[y1:y2, x1:x1+section_w],  # Left 3 squares
            "Right": frame[y1:y2, x2-section_w:x2],  # Right 3 squares
            "Center": frame[y1+section_h:y2-section_h, x1+section_w:x2-section_w]  # Center square
        }

        # Calculate average temperature of each section
        self.temps = {name: np.mean(region) for name, region in sections.items()}
        return self.temps

    def get_avg_temperatures(self):
        """Returns the latest average temperatures for the 5 zones."""
        return self.temps

    def overlay_text(self, frame, temps):
        """Overlays temperature values on the image."""
        h, w = frame.shape[:2]
        section_w, section_h = w // 3, h // 3  # Grid size

        # Set positions to display average temperature
        positions = {
            "Top": (w // 2 - 50, section_h // 2),
            "Bottom": (w // 2 - 50, h - section_h // 2),
            "Left": (section_w // 4, h // 2),
            "Right": (w - section_w // 2 - 50, h // 2),
            "Center": (w // 2 - 50, h // 2)
        }

        # Overlay text for each section
        for section, temp in temps.items():
            x, y = positions[section]
            cv.putText(frame, f"{temp:.2f}C", (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)  #avg temp text

    def start_stream(self):
        """Starts an MJPEG stream."""
        app = Flask(__name__)       # Creates a Flask web server to stream the processed video

        @app.route('/video_feed')   #make sure to add this after the ip ad
        def video_feed():
            return Response(self.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

        app.run(host="0.0.0.0", port=5000, threaded=True)
                                   # Run Flask in a separate thread
       # threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False), daemon=True).start()

    def generate_frames(self):
        """Generates frames for the MJPEG stream."""
        while self.running:
            with self.lock:
                if self.latest_frame is None:
                    continue
                
                _, buffer = cv.imencode('.jpg', self.latest_frame)   #Encodes the processed image as a JPEG for fast streaming
                frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def stop(self):
        """Stops the camera."""
        self.running = False
        self.mi48.stop()
        cv.destroyAllWindows()


# **Main Execution**
if __name__ == "__main__":
    roi = (0, 0, 61, 61)   #(0 10 61 71)                       # Large square ROI change for the camera fov crop
    cam = ThermalCamera(roi=roi, com_port=None)   # Overrides the default value
    cam.start_stream()
