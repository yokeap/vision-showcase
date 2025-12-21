#!/usr/bin/env python3
"""
Real-Time Measurement System with Basler GigE Camera
====================================================

Live measurement using Basler a2A1920-51gcBAS camera with HSV detection.
Press 'q' to quit, 's' to save snapshot, 'c' to recalibrate.

Author: Siwakorn - Kasetsart University
Hardware: Basler a2A1920-51gcBAS on Recomputer J4012
"""

import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime

try:
    from pypylon import pylon
except ImportError:
    print("ERROR: pypylon not installed!")
    print("Install with: pip3 install pypylon")
    exit(1)


class RealtimeMeasurement:
    """Real-time measurement system using Basler camera."""
    
    def __init__(self, calibration_path: str = "calibration.json"):
        """
        Initialize real-time measurement system.
        
        Args:
            calibration_path: Path to calibration JSON file
        """
        self.calibration = self.load_calibration(calibration_path)
        self.pixels_per_mm = self.calibration['pixels_per_mm']
        self.mm_per_pixel = self.calibration['mm_per_pixel']
        
        # Detection parameters (adjustable)
        self.lower_blue = np.array([90, 50, 50])
        self.upper_blue = np.array([130, 255, 255])
        self.min_area = 500
        
        # Display parameters
        self.show_mask = False
        self.freeze_frame = False
        self.frozen_image = None
        self.show_raw = False  # Toggle to show raw vs filtered values
        
        # Measurement stabilization (moving average filter)
        self.filter_size = 5  # Number of frames to average (adjustable 1-20)
        self.width_buffer = []
        self.height_buffer = []
        self.angle_buffer = []
        
        print("=" * 70)
        print("  REAL-TIME MEASUREMENT SYSTEM")
        print("  Basler GigE Camera + HSV Detection")
        print("=" * 70)
        print(f"\nCalibration: {self.pixels_per_mm:.4f} pixels/mm")
        print(f"Camera: Basler a2A1920-51gcBAS")
        print(f"Stabilization: {self.filter_size}-frame moving average")
        print("=" * 70)
    
    def load_calibration(self, path: str) -> dict:
        """Load calibration from JSON file."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")
        
        with open(path, 'r') as f:
            calibration = json.load(f)
        
        return calibration
    
    def setup_camera(self):
        """
        Setup Basler GigE camera with optimal settings.
        
        Returns:
            Camera object ready for capture
        """
        print("\n🎥 Initializing Basler camera...")
        
        # Create camera instance
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()
        
        if len(devices) == 0:
            raise RuntimeError("No Basler camera found!")
        
        print(f"   Found {len(devices)} camera(s)")
        
        # Use first camera
        camera = pylon.InstantCamera(tlFactory.CreateDevice(devices[0]))
        camera.Open()
        
        # Print camera info
        print(f"   Model: {camera.GetDeviceInfo().GetModelName()}")
        print(f"   Serial: {camera.GetDeviceInfo().GetSerialNumber()}")
        
        # Configure camera settings
        camera.PixelFormat.SetValue("RGB8")
        
        # Set acquisition mode to continuous
        camera.AcquisitionMode.SetValue("Continuous")
        
        # Optional: Set frame rate (adjust as needed)
        # camera.AcquisitionFrameRateEnable.SetValue(True)
        # camera.AcquisitionFrameRate.SetValue(30.0)
        
        # Optional: Set exposure (adjust for your lighting)
        # camera.ExposureTime.SetValue(10000)  # microseconds
        
        print("✓ Camera configured and ready!")
        
        return camera
    
    def detect_object_hsv(self, image: np.ndarray) -> tuple:
        """
        Detect object using HSV color thresholding.
        
        Args:
            image: Input BGR image
            
        Returns:
            Tuple of (contours list, binary mask)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for blue color
        blue_mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        
        # Invert mask (we want NON-blue areas)
        object_mask = cv2.bitwise_not(blue_mask)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        valid_contours = [c for c in contours if cv2.contourArea(c) > self.min_area]
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
        
        return valid_contours, object_mask
    
    def measure_object(self, contour: np.ndarray) -> dict:
        """Measure object dimensions from contour using rotated bounding box."""
        
        # Get axis-aligned bounding rectangle (for reference)
        x, y, w_px, h_px = cv2.boundingRect(contour)
        
        # Get minimum area rectangle (rotated bounding box)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Extract rotated rectangle parameters
        (cx, cy), (w_rot, h_rot), angle = rect
        
        # Ensure width > height (swap if needed)
        if w_rot < h_rot:
            w_rot, h_rot = h_rot, w_rot
            angle += 90
        
        # Normalize angle to [-90, 90]
        while angle > 90:
            angle -= 180
        while angle < -90:
            angle += 180
        
        # Convert rotated dimensions to millimeters
        width_mm = w_rot * self.mm_per_pixel
        height_mm = h_rot * self.mm_per_pixel
        
        # Calculate center (use rotated rect center)
        center_x = int(cx)
        center_y = int(cy)
        
        # Calculate area
        area_px = cv2.contourArea(contour)
        area_mm2 = area_px * (self.mm_per_pixel ** 2)
        
        measurement = {
            'bbox': (x, y, w_px, h_px),  # Axis-aligned (for reference)
            'rotated_box': box,           # 4 corner points of rotated rectangle
            'width_mm': width_mm,         # True width (longer side)
            'height_mm': height_mm,       # True height (shorter side)
            'width_px': w_rot,
            'height_px': h_rot,
            'center': (center_x, center_y),
            'angle': angle,               # Rotation angle in degrees
            'area_mm2': area_mm2
        }
        
        return measurement
    
    def stabilize_measurement(self, measurement: dict) -> dict:
        """
        Stabilize measurements using moving average filter.
        
        Args:
            measurement: Raw measurement dict
            
        Returns:
            Stabilized measurement dict
        """
        if measurement is None:
            # Clear buffers if no object detected
            self.width_buffer.clear()
            self.height_buffer.clear()
            self.angle_buffer.clear()
            return None
        
        # Add current measurements to buffers
        self.width_buffer.append(measurement['width_mm'])
        self.height_buffer.append(measurement['height_mm'])
        self.angle_buffer.append(measurement['angle'])
        
        # Keep buffer size limited
        if len(self.width_buffer) > self.filter_size:
            self.width_buffer.pop(0)
            self.height_buffer.pop(0)
            self.angle_buffer.pop(0)
        
        # Calculate moving average
        stabilized = measurement.copy()
        stabilized['width_mm'] = np.mean(self.width_buffer)
        stabilized['height_mm'] = np.mean(self.height_buffer)
        stabilized['angle'] = np.mean(self.angle_buffer)
        
        # Also store raw values for comparison
        stabilized['width_mm_raw'] = measurement['width_mm']
        stabilized['height_mm_raw'] = measurement['height_mm']
        stabilized['angle_raw'] = measurement['angle']
        
        return stabilized
    
    def draw_overlay(self, image: np.ndarray, measurement: dict = None) -> np.ndarray:
        """
        Draw measurement overlay on image.
        
        Args:
            image: Input image
            measurement: Measurement dict (None if no object detected)
            
        Returns:
            Image with overlay
        """
        overlay = image.copy()
        h, w = overlay.shape[:2]
        
        # Draw status bar at top
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Title
        cv2.putText(overlay, "REAL-TIME MEASUREMENT", (10, 30),
                   font, 0.8, (0, 255, 0), 2)
        
        # FPS and status
        status_text = "FROZEN" if self.freeze_frame else "LIVE"
        status_color = (0, 165, 255) if self.freeze_frame else (0, 255, 0)
        cv2.putText(overlay, status_text, (10, 60),
                   font, 0.6, status_color, 2)
        
        # If object detected, draw measurements
        if measurement is not None:
            x, y, w, h = measurement['bbox']
            rotated_box = measurement['rotated_box']
            width_mm = measurement['width_mm']
            height_mm = measurement['height_mm']
            cx, cy = measurement['center']
            angle = measurement['angle']
            
            # Colors
            COLOR_BOX = (0, 255, 0)
            COLOR_TEXT = (255, 255, 255)
            COLOR_DIM = (255, 0, 0)
            
            # Draw ROTATED bounding box (the one we actually measure)
            cv2.drawContours(overlay, [rotated_box], 0, COLOR_BOX, 3)
            
            # Draw center crosshair
            cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)
            cv2.line(overlay, (cx - 15, cy), (cx + 15, cy), (0, 0, 255), 2)
            cv2.line(overlay, (cx, cy - 15), (cx, cy + 15), (0, 0, 255), 2)
            
            # Calculate rotated box edges for dimension lines
            # Get the 4 corners of rotated box
            pt1, pt2, pt3, pt4 = rotated_box
            
            # Find top edge (for width dimension)
            # Top edge is the edge closest to top of image
            edges = [
                (pt1, pt2, (pt1[1] + pt2[1]) / 2),
                (pt2, pt3, (pt2[1] + pt3[1]) / 2),
                (pt3, pt4, (pt3[1] + pt4[1]) / 2),
                (pt4, pt1, (pt4[1] + pt1[1]) / 2)
            ]
            top_edge = min(edges, key=lambda e: e[2])
            
            # Draw width dimension on top edge
            offset = 40
            p1, p2 = top_edge[0], top_edge[1]
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2 - offset
            
            # Width line
            cv2.line(overlay, tuple(p1), tuple(p2), COLOR_DIM, 2)
            
            # Width text
            width_text = f"W: {width_mm:.2f} mm"
            text_size = cv2.getTextSize(width_text, font, 0.7, 2)[0]
            text_x = mid_x - text_size[0] // 2
            text_y = mid_y
            cv2.rectangle(overlay, (text_x - 5, text_y - text_size[1] - 5),
                         (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
            cv2.putText(overlay, width_text, (text_x, text_y),
                       font, 0.7, COLOR_TEXT, 2)
            
            # Find right edge (for height dimension)
            edges_x = [
                (pt1, pt2, (pt1[0] + pt2[0]) / 2),
                (pt2, pt3, (pt2[0] + pt3[0]) / 2),
                (pt3, pt4, (pt3[0] + pt4[0]) / 2),
                (pt4, pt1, (pt4[0] + pt1[0]) / 2)
            ]
            right_edge = max(edges_x, key=lambda e: e[2])
            
            # Height text on right edge
            p1, p2 = right_edge[0], right_edge[1]
            mid_x = (p1[0] + p2[0]) // 2 + offset
            mid_y = (p1[1] + p2[1]) // 2
            
            # Height line
            cv2.line(overlay, tuple(p1), tuple(p2), COLOR_DIM, 2)
            
            height_text = f"H: {height_mm:.2f} mm"
            text_size = cv2.getTextSize(height_text, font, 0.7, 2)[0]
            text_x = mid_x
            text_y = mid_y + text_size[1] // 2
            cv2.rectangle(overlay, (text_x - 5, text_y - text_size[1] - 5),
                         (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
            cv2.putText(overlay, height_text, (text_x, text_y),
                       font, 0.7, COLOR_TEXT, 2)
            
            # Measurement panel
            panel_y = 100
            panel_height = 130 if hasattr(measurement, 'width_mm_raw') else 110
            cv2.rectangle(overlay, (10, panel_y), (380, panel_y + panel_height), (0, 0, 0), -1)
            
            # Title with filter status
            filter_status = f"FILTERED ({self.filter_size} frames)" if not self.show_raw else "RAW (unfiltered)"
            title_color = (0, 255, 0) if not self.show_raw else (0, 165, 255)
            cv2.putText(overlay, filter_status, (20, panel_y + 25),
                       font, 0.6, title_color, 1)
            
            # Get values to display (raw or filtered)
            if self.show_raw and 'width_mm_raw' in measurement:
                w_display = measurement['width_mm_raw']
                h_display = measurement['height_mm_raw']
                a_display = measurement['angle_raw']
            else:
                w_display = width_mm
                h_display = height_mm
                a_display = angle
            
            cv2.putText(overlay, f"Width:  {w_display:6.2f} mm", (20, panel_y + 50),
                       font, 0.5, COLOR_TEXT, 1)
            cv2.putText(overlay, f"Height: {h_display:6.2f} mm", (20, panel_y + 70),
                       font, 0.5, COLOR_TEXT, 1)
            cv2.putText(overlay, f"Angle:  {a_display:6.1f} deg", (20, panel_y + 90),
                       font, 0.5, COLOR_TEXT, 1)
            
            # Show noise level if filtered
            if not self.show_raw and 'width_mm_raw' in measurement:
                noise_w = abs(measurement['width_mm_raw'] - width_mm)
                noise_h = abs(measurement['height_mm_raw'] - height_mm)
                cv2.putText(overlay, f"Noise:  W:{noise_w:.2f} H:{noise_h:.2f}", 
                           (20, panel_y + 110), font, 0.4, (100, 100, 100), 1)
        else:
            # No object detected
            cv2.putText(overlay, "No object detected", (w // 2 - 150, h // 2),
                       font, 1.0, (0, 0, 255), 2)
        
        # Help text at bottom
        help_y = h - 100
        cv2.rectangle(overlay, (0, help_y), (w, h), (0, 0, 0), -1)
        cv2.putText(overlay, "Controls: [Q]uit  [S]ave  [F]reeze  [M]ask  [R]aw  [+/-]Filter", 
                   (10, help_y + 25), font, 0.5, (200, 200, 200), 1)
        cv2.putText(overlay, f"Scale: {self.pixels_per_mm:.2f} px/mm  |  Filter: {self.filter_size} frames", 
                   (10, help_y + 50), font, 0.5, (200, 200, 200), 1)
        cv2.putText(overlay, "Tip: Increase filter for stability, decrease for responsiveness", 
                   (10, help_y + 75), font, 0.4, (150, 150, 150), 1)
        
        return overlay
    
    def draw_overlay_multi(self, image: np.ndarray, measurements: list) -> np.ndarray:
        """
        Draw measurement overlay for multiple objects.
        
        Args:
            image: Input image
            measurements: List of measurement dicts
            
        Returns:
            Image with overlay
        """
        overlay = image.copy()
        h, w = overlay.shape[:2]
        
        # Draw status bar at top
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Title
        cv2.putText(overlay, "REAL-TIME MEASUREMENT - MULTI OBJECT", (10, 30),
                   font, 0.8, (0, 255, 0), 2)
        
        # Object count and status
        obj_count = len(measurements) if measurements else 0
        status_text = f"{obj_count} object(s)" + (" - FROZEN" if self.freeze_frame else " - LIVE")
        status_color = (0, 165, 255) if self.freeze_frame else (0, 255, 0)
        cv2.putText(overlay, status_text, (10, 60),
                   font, 0.6, status_color, 2)
        
        # Define colors for different objects
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 165, 255),  # Orange
            (255, 255, 0),  # Cyan
            (128, 0, 255),  # Purple
            (0, 255, 128),  # Spring Green
            (255, 128, 0),  # Sky Blue
            (128, 255, 0),  # Lime
        ]
        
        # Draw each object
        if measurements:
            for idx, measurement in enumerate(measurements):
                color = colors[idx % len(colors)]
                
                rotated_box = measurement['rotated_box']
                width_mm = measurement['width_mm']
                height_mm = measurement['height_mm']
                cx, cy = measurement['center']
                angle = measurement['angle']
                obj_id = measurement.get('id', idx + 1)
                
                # Draw rotated bounding box
                cv2.drawContours(overlay, [rotated_box], 0, color, 2)
                
                # Draw object ID at center
                cv2.circle(overlay, (cx, cy), 15, color, -1)
                cv2.putText(overlay, str(obj_id), (cx - 7, cy + 7),
                           font, 0.6, (255, 255, 255), 2)
                
                # Draw measurement text near object
                text_y_offset = -25
                text_x = cx - 60
                text_y = cy + text_y_offset
                
                # Background for text
                text = f"#{obj_id}: {width_mm:.1f}x{height_mm:.1f}mm"
                text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
                cv2.rectangle(overlay, 
                             (text_x - 3, text_y - text_size[1] - 3),
                             (text_x + text_size[0] + 3, text_y + 3),
                             (0, 0, 0), -1)
                cv2.putText(overlay, text, (text_x, text_y),
                           font, 0.5, color, 1)
        else:
            # No objects detected
            cv2.putText(overlay, "No objects detected", (w // 2 - 150, h // 2),
                       font, 1.0, (0, 0, 255), 2)
        
        # Measurement table on left side
        if measurements:
            table_x = 10
            table_y = 100
            table_width = 400
            row_height = 30
            table_height = min(len(measurements) * row_height + 60, h - 250)
            
            cv2.rectangle(overlay, (table_x, table_y), 
                         (table_x + table_width, table_y + table_height), 
                         (0, 0, 0), -1)
            
            # Table header
            filter_status = f"FILTERED ({self.filter_size})" if not self.show_raw else "RAW"
            cv2.putText(overlay, f"MEASUREMENTS - {filter_status}", 
                       (table_x + 10, table_y + 25),
                       font, 0.5, (0, 255, 0), 1)
            
            # Column headers
            header_y = table_y + 50
            cv2.putText(overlay, "ID", (table_x + 10, header_y),
                       font, 0.4, (200, 200, 200), 1)
            cv2.putText(overlay, "Width(mm)", (table_x + 50, header_y),
                       font, 0.4, (200, 200, 200), 1)
            cv2.putText(overlay, "Height(mm)", (table_x + 150, header_y),
                       font, 0.4, (200, 200, 200), 1)
            cv2.putText(overlay, "Angle", (table_x + 260, header_y),
                       font, 0.4, (200, 200, 200), 1)
            cv2.putText(overlay, "Area", (table_x + 330, header_y),
                       font, 0.4, (200, 200, 200), 1)
            
            # Draw separator line
            cv2.line(overlay, (table_x + 5, header_y + 5), 
                    (table_x + table_width - 5, header_y + 5), 
                    (100, 100, 100), 1)
            
            # Data rows
            for idx, measurement in enumerate(measurements[:8]):  # Limit to 8 visible
                row_y = header_y + 25 + idx * row_height
                color = colors[idx % len(colors)]
                obj_id = measurement.get('id', idx + 1)
                
                # Get display values
                if self.show_raw and 'width_mm_raw' in measurement:
                    w_val = measurement['width_mm_raw']
                    h_val = measurement['height_mm_raw']
                    a_val = measurement['angle_raw']
                else:
                    w_val = measurement['width_mm']
                    h_val = measurement['height_mm']
                    a_val = measurement['angle']
                
                area_val = measurement['area_mm2']
                
                cv2.putText(overlay, f"{obj_id}", (table_x + 15, row_y),
                           font, 0.45, color, 1)
                cv2.putText(overlay, f"{w_val:6.2f}", (table_x + 60, row_y),
                           font, 0.45, (255, 255, 255), 1)
                cv2.putText(overlay, f"{h_val:6.2f}", (table_x + 160, row_y),
                           font, 0.45, (255, 255, 255), 1)
                cv2.putText(overlay, f"{a_val:5.1f}", (table_x + 265, row_y),
                           font, 0.45, (255, 255, 255), 1)
                cv2.putText(overlay, f"{area_val:4.0f}", (table_x + 330, row_y),
                           font, 0.45, (255, 255, 255), 1)
            
            # If more than 8 objects, show indicator
            if len(measurements) > 8:
                more_y = header_y + 25 + 8 * row_height
                cv2.putText(overlay, f"... and {len(measurements) - 8} more", 
                           (table_x + 10, more_y),
                           font, 0.4, (150, 150, 150), 1)
        
        # Help text at bottom
        help_y = h - 100
        cv2.rectangle(overlay, (0, help_y), (w, h), (0, 0, 0), -1)
        cv2.putText(overlay, "Controls: [Q]uit  [S]ave  [F]reeze  [M]ask  [R]aw  [+/-]Filter", 
                   (10, help_y + 25), font, 0.5, (200, 200, 200), 1)
        cv2.putText(overlay, f"Scale: {self.pixels_per_mm:.2f} px/mm  |  Filter: {self.filter_size} frames  |  Objects: {obj_count}", 
                   (10, help_y + 50), font, 0.5, (200, 200, 200), 1)
        cv2.putText(overlay, "Multi-object detection enabled (max 10 objects)", 
                   (10, help_y + 75), font, 0.4, (150, 150, 150), 1)
        
        return overlay
    
    def run(self):
        """Main real-time measurement loop."""
        
        # Setup camera
        camera = self.setup_camera()
        
        # Start grabbing
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        
        # Converter for BGR format
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        
        print("\n" + "=" * 70)
        print("  LIVE MEASUREMENT STARTED")
        print("=" * 70)
        print("\nControls:")
        print("  Q      - Quit")
        print("  S      - Save snapshot")
        print("  F      - Freeze/Unfreeze frame")
        print("  M      - Toggle mask view")
        print("  R      - Toggle Raw/Filtered display")
        print("  + / -  - Increase/Decrease filter strength")
        print("\nFilter Settings:")
        print(f"  Current: {self.filter_size} frames (adjustable 1-20)")
        print("  Higher = More stable but slower response")
        print("  Lower  = Faster response but more noise")
        print("=" * 70)
        
        try:
            while camera.IsGrabbing():
                # Grab frame
                grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                
                if grabResult.GrabSucceeded():
                    # Convert to BGR
                    image = converter.Convert(grabResult)
                    img = image.GetArray()
                    
                    # Use frozen frame if freeze is active
                    if self.freeze_frame and self.frozen_image is not None:
                        img = self.frozen_image
                    
                    # Detect object
                    contours, mask = self.detect_object_hsv(img)
                    
                    # Measure ALL objects found (not just the largest)
                    measurements = []
                    if len(contours) > 0:
                        # Limit to reasonable number of objects (e.g., 10)
                        max_objects = 10
                        for i, contour in enumerate(contours[:max_objects]):
                            raw_measurement = self.measure_object(contour)
                            # Add object ID
                            raw_measurement['id'] = i + 1
                            measurements.append(raw_measurement)
                        
                        # Apply stabilization filter to each object
                        # For simplicity, only filter the first object for now
                        # (you can expand this to track multiple objects)
                        if len(measurements) > 0:
                            stabilized = self.stabilize_measurement(measurements[0])
                            if stabilized:
                                measurements[0] = stabilized
                    else:
                        # Clear filter buffers if no object
                        self.stabilize_measurement(None)
                    
                    # Draw overlay
                    display = self.draw_overlay_multi(img, measurements)
                    
                    # Show result
                    cv2.imshow("Real-Time Measurement", display)
                    
                    # Show mask if enabled
                    if self.show_mask:
                        cv2.imshow("Detection Mask", mask)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q') or key == ord('Q'):
                        print("\n✓ Quitting...")
                        break
                    
                    elif key == ord('s') or key == ord('S'):
                        # Save snapshot
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"snapshot_{timestamp}.png"
                        cv2.imwrite(filename, display)
                        print(f"✓ Saved: {filename}")
                    
                    elif key == ord('f') or key == ord('F'):
                        # Toggle freeze
                        self.freeze_frame = not self.freeze_frame
                        if self.freeze_frame:
                            self.frozen_image = img.copy()
                            print("⏸  Frame FROZEN")
                        else:
                            self.frozen_image = None
                            print("▶  Frame LIVE")
                    
                    elif key == ord('m') or key == ord('M'):
                        # Toggle mask view
                        self.show_mask = not self.show_mask
                        if not self.show_mask:
                            cv2.destroyWindow("Detection Mask")
                    
                    elif key == ord('r') or key == ord('R'):
                        # Toggle raw/filtered display
                        self.show_raw = not self.show_raw
                        mode = "RAW (unfiltered)" if self.show_raw else f"FILTERED ({self.filter_size} frames)"
                        print(f"📊 Display mode: {mode}")
                    
                    elif key == ord('+') or key == ord('='):
                        # Increase filter size
                        if self.filter_size < 20:
                            self.filter_size += 1
                            print(f"🔧 Filter size: {self.filter_size} frames (more stable)")
                    
                    elif key == ord('-') or key == ord('_'):
                        # Decrease filter size
                        if self.filter_size > 1:
                            self.filter_size -= 1
                            print(f"🔧 Filter size: {self.filter_size} frames (more responsive)")
                            if self.filter_size == 1:
                                print("   (Filter disabled - showing real-time values)")
                
                grabResult.Release()
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        finally:
            # Cleanup
            camera.StopGrabbing()
            camera.Close()
            cv2.destroyAllWindows()
            print("\n✓ Camera closed")
            print("✓ Program terminated")


def main():
    """Main entry point."""
    
    CALIBRATION_FILE = "calibration.json"
    
    print("\n" + "=" * 70)
    print("  BASLER GIGE REAL-TIME MEASUREMENT SYSTEM")
    print("=" * 70)
    print("\nHardware: Basler a2A1920-51gcBAS")
    print("Platform: Recomputer J4012 (Jetson)")
    print("Method: HSV color thresholding + Real-time processing")
    print("=" * 70)
    
    try:
        # Create measurement system
        measurer = RealtimeMeasurement(CALIBRATION_FILE)
        
        # Run live measurement
        measurer.run()
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  • calibration.json (run calibrator.py first)")
    
    except RuntimeError as e:
        print(f"\n❌ Camera Error: {e}")
        print("\nTroubleshooting:")
        print("  • Is the Basler camera connected?")
        print("  • Check network settings (GigE camera)")
        print("  • Try: ping <camera_ip>")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()