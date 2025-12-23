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
        
        # Ground truth for error calculation (T-nut actual dimensions)
        self.GROUND_TRUTH_WIDTH = 15.7   # mm
        self.GROUND_TRUTH_HEIGHT = 15.8  # mm
        
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
        
        # Multi-object tracking - dictionary of buffers per object ID
        self.object_trackers = {}  # {object_id: {'width': [], 'height': [], 'angle': [], 'last_seen': frame_num}}
        self.frame_count = 0
        self.max_missing_frames = 10  # Remove tracker if object missing for this many frames
        
        # Camera info (will be set when camera is initialized)
        self.camera_model = ""
        self.camera_serial = ""
        
        print("=" * 70)
        print("  REAL-TIME MEASUREMENT SYSTEM")
        print("  Basler GigE Camera + HSV Detection")
        print("=" * 70)
        print(f"\nCalibration: {self.pixels_per_mm:.4f} pixels/mm")
        print(f"Camera: Basler a2A1920-51gcBAS")
        print(f"Stabilization: {self.filter_size}-frame moving average")
        print(f"Ground Truth: Width={self.GROUND_TRUTH_WIDTH}mm, Height={self.GROUND_TRUTH_HEIGHT}mm")
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
            Camera object ready for capture, model name, serial number
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
        
        # Get camera info
        camera_model = camera.GetDeviceInfo().GetModelName()
        camera_serial = camera.GetDeviceInfo().GetSerialNumber()
        
        # Print camera info
        print(f"   Model: {camera_model}")
        print(f"   Serial: {camera_serial}")
        
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
        
        # Store camera info in class
        self.camera_model = camera_model
        self.camera_serial = camera_serial
        
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
        
        # Calculate errors against ground truth
        width_error = abs(width_mm - self.GROUND_TRUTH_WIDTH)
        height_error = abs(height_mm - self.GROUND_TRUTH_HEIGHT)
        
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
            'width_error': width_error,   # Absolute error vs ground truth
            'height_error': height_error, # Absolute error vs ground truth
            'width_px': w_rot,
            'height_px': h_rot,
            'center': (center_x, center_y),
            'angle': angle,               # Rotation angle in degrees
            'area_mm2': area_mm2
        }
        
        return measurement
    
    def assign_object_ids(self, measurements: list) -> list:
        """
        Assign persistent IDs to objects based on position tracking.
        Simple nearest-neighbor matching between frames.
        
        Args:
            measurements: List of measurement dicts
            
        Returns:
            List of measurements with persistent IDs assigned
        """
        if not measurements:
            return []
        
        # Get current object centers
        current_centers = [(m['center'][0], m['center'][1]) for m in measurements]
        
        # Match with existing trackers
        used_ids = set()
        assigned_measurements = []
        
        for m_idx, measurement in enumerate(measurements):
            cx, cy = measurement['center']
            
            # Find closest existing tracker
            best_id = None
            best_dist = float('inf')
            
            for obj_id, tracker in self.object_trackers.items():
                if obj_id in used_ids:
                    continue
                
                # Get last known position
                if 'last_center' in tracker:
                    last_cx, last_cy = tracker['last_center']
                    dist = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                    
                    # If within reasonable distance (e.g., 100 pixels), consider it same object
                    if dist < 100 and dist < best_dist:
                        best_dist = dist
                        best_id = obj_id
            
            # Assign ID
            if best_id is not None:
                measurement['id'] = best_id
                used_ids.add(best_id)
            else:
                # Create new ID
                new_id = max(self.object_trackers.keys()) + 1 if self.object_trackers else 1
                measurement['id'] = new_id
                used_ids.add(new_id)
            
            assigned_measurements.append(measurement)
        
        return assigned_measurements
    
    def stabilize_measurements(self, measurements: list) -> list:
        """
        Stabilize measurements for multiple objects using per-object filters.
        
        Args:
            measurements: List of raw measurement dicts with IDs
            
        Returns:
            List of stabilized measurement dicts
        """
        self.frame_count += 1
        
        if not measurements:
            # Clean up old trackers (objects that haven't been seen recently)
            to_remove = []
            for obj_id, tracker in self.object_trackers.items():
                if self.frame_count - tracker['last_seen'] > self.max_missing_frames:
                    to_remove.append(obj_id)
            
            for obj_id in to_remove:
                del self.object_trackers[obj_id]
            
            return []
        
        stabilized_measurements = []
        
        for measurement in measurements:
            obj_id = measurement['id']
            
            # Create tracker if doesn't exist
            if obj_id not in self.object_trackers:
                self.object_trackers[obj_id] = {
                    'width': [],
                    'height': [],
                    'angle': [],
                    'width_error': [],
                    'height_error': [],
                    'last_seen': self.frame_count,
                    'last_center': measurement['center']
                }
            
            tracker = self.object_trackers[obj_id]
            
            # Update tracker
            tracker['width'].append(measurement['width_mm'])
            tracker['height'].append(measurement['height_mm'])
            tracker['angle'].append(measurement['angle'])
            tracker['width_error'].append(measurement['width_error'])
            tracker['height_error'].append(measurement['height_error'])
            tracker['last_seen'] = self.frame_count
            tracker['last_center'] = measurement['center']
            
            # Keep buffer size limited
            if len(tracker['width']) > self.filter_size:
                tracker['width'].pop(0)
                tracker['height'].pop(0)
                tracker['angle'].pop(0)
                tracker['width_error'].pop(0)
                tracker['height_error'].pop(0)
            
            # Calculate moving average
            stabilized = measurement.copy()
            stabilized['width_mm'] = np.mean(tracker['width'])
            stabilized['height_mm'] = np.mean(tracker['height'])
            stabilized['angle'] = np.mean(tracker['angle'])
            stabilized['width_error'] = np.mean(tracker['width_error'])
            stabilized['height_error'] = np.mean(tracker['height_error'])
            
            # Store raw values for comparison
            stabilized['width_mm_raw'] = measurement['width_mm']
            stabilized['height_mm_raw'] = measurement['height_mm']
            stabilized['angle_raw'] = measurement['angle']
            stabilized['width_error_raw'] = measurement['width_error']
            stabilized['height_error_raw'] = measurement['height_error']
            
            stabilized_measurements.append(stabilized)
        
        return stabilized_measurements
    
    def stabilize_measurement(self, measurement: dict) -> dict:
        """
        Legacy single-object stabilization (kept for compatibility).
        Now just wraps the multi-object version.
        """
        if measurement is None:
            return None
        
        results = self.stabilize_measurements([measurement])
        return results[0] if results else None
    
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
        
        # Draw status bar at top (taller for camera info)
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Title - emphasize scaling method
        cv2.putText(overlay, "SCALING-BASED MEASUREMENT", (10, 28),
                   font, 0.8, (0, 255, 255), 2)
        
        # Camera info - emphasize GigE Industrial
        camera_text = f"Camera: {self.camera_model} (GigE Industrial)"
        cv2.putText(overlay, camera_text, (10, 55),
                   font, 0.55, (100, 255, 100), 1)
        
        # Object count and status
        obj_count = len(measurements) if measurements else 0
        filter_mode = "RAW" if self.show_raw else f"FILTERED ({self.filter_size}x)"
        status_text = f"Objects: {obj_count}  |  Mode: {filter_mode}"
        status_color = (0, 165, 255) if self.freeze_frame else (200, 200, 200)
        cv2.putText(overlay, status_text, (10, 80),
                   font, 0.5, status_color, 1)
        
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
            table_y = 110
            table_width = 510
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
            cv2.putText(overlay, "Width(mm)", (table_x + 40, header_y),
                       font, 0.4, (200, 200, 200), 1)
            cv2.putText(overlay, "Err (%)", (table_x + 125, header_y),
                       font, 0.4, (255, 165, 0), 1)  # Orange for error
            cv2.putText(overlay, "Height(mm)", (table_x + 180, header_y),
                       font, 0.4, (200, 200, 200), 1)
            cv2.putText(overlay, "Err (%)", (table_x + 260, header_y),
                       font, 0.4, (255, 165, 0), 1)  # Orange for error
            cv2.putText(overlay, "Angle", (table_x + 315, header_y),
                       font, 0.4, (200, 200, 200), 1)
            cv2.putText(overlay, "Area", (table_x + 365, header_y),
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
                    w_err = measurement.get('width_error_raw', 0)
                    h_err = measurement.get('height_error_raw', 0)
                else:
                    w_val = measurement['width_mm']
                    h_val = measurement['height_mm']
                    a_val = measurement['angle']
                    w_err = measurement.get('width_error', 0)
                    h_err = measurement.get('height_error', 0)
                
                area_val = measurement['area_mm2']
                
                cv2.putText(overlay, f"{obj_id}", (table_x + 15, row_y),
                           font, 0.45, color, 1)
                cv2.putText(overlay, f"{w_val:6.2f}", (table_x + 45, row_y),
                           font, 0.45, (255, 255, 255), 1)
                cv2.putText(overlay, f"{w_err:.2f}", (table_x + 125, row_y),
                           font, 0.45, (0, 165, 255), 1)  # Orange for width error
                cv2.putText(overlay, f"{h_val:6.2f}", (table_x + 175, row_y),
                           font, 0.45, (255, 255, 255), 1)
                cv2.putText(overlay, f"{h_err:.2f}", (table_x + 260, row_y),
                           font, 0.45, (0, 165, 255), 1)  # Orange for height error
                cv2.putText(overlay, f"{a_val:5.1f}", (table_x + 310, row_y),
                           font, 0.45, (255, 255, 255), 1)
                cv2.putText(overlay, f"{area_val:4.0f}", (table_x + 370, row_y),
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
        cv2.putText(overlay, f"Ground Truth: W={self.GROUND_TRUTH_WIDTH}mm H={self.GROUND_TRUTH_HEIGHT}mm  |  Error shown in orange", 
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
                    
                    # Measure ALL objects found
                    measurements = []
                    if len(contours) > 0:
                        # Limit to reasonable number of objects
                        max_objects = 10
                        for i, contour in enumerate(contours[:max_objects]):
                            raw_measurement = self.measure_object(contour)
                            measurements.append(raw_measurement)
                        
                        # Assign persistent IDs based on position tracking
                        measurements = self.assign_object_ids(measurements)
                        
                        # Apply stabilization filter to ALL objects
                        measurements = self.stabilize_measurements(measurements)
                    else:
                        # No objects - clean up old trackers
                        self.stabilize_measurements([])
                    
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