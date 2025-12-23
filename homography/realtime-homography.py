import cv2
import numpy as np
import json
from pypylon import pylon
from datetime import datetime
import time

class TnutMeasurementSystem:
    def __init__(self, homography_file='homography_calibration.json'):
        """Initialize measurement system"""
        self.H = self.load_homography(homography_file)
        self.measurements_history = []
        self.fps = 0
        self.last_time = time.time()
        
        # Measurement filters
        self.min_area_pixels = 100
        self.max_area_pixels = 50000
        
        # Ground truth for error calculation (T-nut actual dimensions)
        self.GROUND_TRUTH_WIDTH = 15.7   # mm
        self.GROUND_TRUTH_HEIGHT = 15.8  # mm
        
        # Colors (BGR)
        self.COLOR_PASS = (0, 255, 0)      # Green
        self.COLOR_FAIL = (0, 0, 255)      # Red
        self.COLOR_TEXT = (0, 255, 255)    # Yellow
        self.COLOR_BBOX = (255, 0, 0)      # Blue
        self.COLOR_ERROR = (0, 165, 255)   # Orange
        
        # Measurement stabilization (moving average filter)
        self.filter_size = 5  # Number of frames to average (adjustable 1-20)
        self.show_raw = False  # Toggle to show raw vs filtered values
        
        # Multi-object tracking - dictionary of buffers per object ID
        self.object_trackers = {}  # {object_id: {'width': [], 'height': [], 'area': [], 'last_seen': frame_num}}
        self.frame_count = 0
        self.max_missing_frames = 10  # Remove tracker if object missing for this many frames
        
        # Camera info (will be set later)
        self.camera_model = ""
        self.camera_serial = ""
        
        print(f"Stabilization: {self.filter_size}-frame moving average")
        print(f"Ground Truth: Width={self.GROUND_TRUTH_WIDTH}mm, Height={self.GROUND_TRUTH_HEIGHT}mm")
        
    def load_homography(self, json_file):
        """Load homography matrix from JSON file"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        H = np.array(data['homography_matrix'])
        print(f"\n📐 Loaded homography matrix from '{json_file}'")
        print(f"   Marker size: {data['marker_size_mm']} mm")
        print(f"   Reprojection error: {data['reprojection_error_mean']:.6f} pixels")
        
        return H
    
    def pixel_to_world(self, pixel_points):
        """Convert pixel coordinates to world coordinates (mm)"""
        pixel_points = np.array(pixel_points)
        if pixel_points.ndim == 1:
            pixel_points = pixel_points.reshape(1, -1)
        
        ones = np.ones((len(pixel_points), 1))
        pixel_homogeneous = np.hstack([pixel_points, ones])
        
        H_inv = np.linalg.inv(self.H)
        world_homogeneous = (H_inv @ pixel_homogeneous.T).T
        world_points = world_homogeneous[:, :2] / world_homogeneous[:, 2:]
        
        return world_points
    
    def preprocess_image(self, image):
        """Remove blue background and isolate objects"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Blue background threshold
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        object_mask = cv2.bitwise_not(blue_mask)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
        
        return object_mask
    
    def find_all_objects(self, mask):
        """Find all object contours"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area_pixels < area < self.max_area_pixels:
                valid_contours.append(contour)
        
        return valid_contours
    
    def measure_object(self, contour):
        """Measure single object dimensions"""
        # Get rotated bounding box
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        
        # Convert to world coordinates
        world_corners = self.pixel_to_world(box)
        
        # Calculate dimensions
        distances = []
        for i in range(4):
            j = (i + 1) % 4
            dist = np.linalg.norm(world_corners[i] - world_corners[j])
            distances.append(dist)
        
        distances.sort()
        width_mm = distances[0]
        height_mm = distances[2]
        
        # Calculate errors against ground truth
        width_error = abs(width_mm - self.GROUND_TRUTH_WIDTH)
        height_error = abs(height_mm - self.GROUND_TRUTH_HEIGHT)
        
        # Calculate area
        contour_points = contour.reshape(-1, 2)
        world_contour = self.pixel_to_world(contour_points)
        
        area_mm2 = 0
        for i in range(len(world_contour)):
            j = (i + 1) % len(world_contour)
            area_mm2 += world_contour[i][0] * world_contour[j][1]
            area_mm2 -= world_contour[j][0] * world_contour[i][1]
        area_mm2 = abs(area_mm2) / 2.0
        
        # Get center
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        
        return {
            'width_mm': width_mm,
            'height_mm': height_mm,
            'width_error': width_error,
            'height_error': height_error,
            'area_mm2': area_mm2,
            'center_pixel': (cx, cy),
            'box': box.astype(int),
            'pass': self.check_pass_fail(width_mm, height_mm),
            'object_id': None  # Will be assigned later
        }
    
    def assign_object_ids(self, measurements):
        """
        Assign persistent IDs to objects based on position tracking.
        Simple nearest-neighbor matching.
        """
        self.frame_count += 1
        
        # Mark all existing trackers as not seen this frame
        for obj_id in self.object_trackers:
            if 'last_seen' not in self.object_trackers[obj_id]:
                self.object_trackers[obj_id]['last_seen'] = self.frame_count - 1
        
        # Match measurements to existing trackers
        unmatched_measurements = []
        for meas in measurements:
            cx, cy = meas['center_pixel']
            
            # Find nearest tracker within threshold
            best_match = None
            best_dist = 100  # Max pixel distance threshold
            
            for obj_id, tracker in self.object_trackers.items():
                if 'center' in tracker:
                    old_cx, old_cy = tracker['center']
                    dist = np.sqrt((cx - old_cx)**2 + (cy - old_cy)**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = obj_id
            
            if best_match is not None:
                # Match found - update tracker
                meas['object_id'] = best_match
                self.object_trackers[best_match]['center'] = (cx, cy)
                self.object_trackers[best_match]['last_seen'] = self.frame_count
            else:
                # No match - this is a new object
                unmatched_measurements.append(meas)
        
        # Assign new IDs to unmatched measurements
        for meas in unmatched_measurements:
            # Find next available ID
            used_ids = set(self.object_trackers.keys())
            new_id = 0
            while new_id in used_ids:
                new_id += 1
            
            meas['object_id'] = new_id
            cx, cy = meas['center_pixel']
            self.object_trackers[new_id] = {
                'center': (cx, cy),
                'width': [],
                'height': [],
                'area': [],
                'last_seen': self.frame_count
            }
        
        # Remove old trackers that haven't been seen
        trackers_to_remove = []
        for obj_id, tracker in self.object_trackers.items():
            if self.frame_count - tracker['last_seen'] > self.max_missing_frames:
                trackers_to_remove.append(obj_id)
        
        for obj_id in trackers_to_remove:
            del self.object_trackers[obj_id]
        
        return measurements
    
    def stabilize_measurements(self, measurements):
        """
        Apply moving average filter to stabilize measurements.
        """
        stabilized = []
        
        for meas in measurements:
            obj_id = meas['object_id']
            
            if obj_id not in self.object_trackers:
                # Shouldn't happen, but handle gracefully
                stabilized.append(meas)
                continue
            
            tracker = self.object_trackers[obj_id]
            
            # Store raw measurement
            meas['width_mm_raw'] = meas['width_mm']
            meas['height_mm_raw'] = meas['height_mm']
            meas['width_error_raw'] = meas['width_error']
            meas['height_error_raw'] = meas['height_error']
            meas['area_mm2_raw'] = meas['area_mm2']
            
            # Add to history
            if 'width_error' not in tracker:
                tracker['width_error'] = []
                tracker['height_error'] = []
            
            tracker['width'].append(meas['width_mm'])
            tracker['height'].append(meas['height_mm'])
            tracker['width_error'].append(meas['width_error'])
            tracker['height_error'].append(meas['height_error'])
            tracker['area'].append(meas['area_mm2'])
            
            # Keep only last N frames
            tracker['width'] = tracker['width'][-self.filter_size:]
            tracker['height'] = tracker['height'][-self.filter_size:]
            tracker['width_error'] = tracker['width_error'][-self.filter_size:]
            tracker['height_error'] = tracker['height_error'][-self.filter_size:]
            tracker['area'] = tracker['area'][-self.filter_size:]
            
            # Calculate filtered values (moving average)
            if self.show_raw or self.filter_size == 1:
                # Show raw values
                meas['width_mm_filtered'] = meas['width_mm']
                meas['height_mm_filtered'] = meas['height_mm']
                meas['width_error_filtered'] = meas['width_error']
                meas['height_error_filtered'] = meas['height_error']
                meas['area_mm2_filtered'] = meas['area_mm2']
            else:
                # Show filtered values
                meas['width_mm_filtered'] = np.mean(tracker['width'])
                meas['height_mm_filtered'] = np.mean(tracker['height'])
                meas['width_error_filtered'] = np.mean(tracker['width_error'])
                meas['height_error_filtered'] = np.mean(tracker['height_error'])
                meas['area_mm2_filtered'] = np.mean(tracker['area'])
            
            stabilized.append(meas)
        
        return stabilized
    
    def check_pass_fail(self, width, height):
        """Check if dimensions are within tolerance"""
        # Example tolerances (adjust as needed)
        # For T-nut: typical dimensions might be ~10-20mm
        width_min, width_max = 8.0, 25.0
        height_min, height_max = 8.0, 25.0
        
        if width_min <= width <= width_max and height_min <= height <= height_max:
            return True
        return False
    
    def draw_industrial_overlay(self, image, measurements_list):
        """Draw overlay matching scaling method style"""
        overlay = image.copy()
        h, w = overlay.shape[:2]
        
        # Draw status bar at top (taller for camera info)
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Title - emphasize homography method
        cv2.putText(overlay, "HOMOGRAPHY-BASED MEASUREMENT", (10, 28),
                   font, 0.8, (0, 255, 255), 2)
        
        # Camera info - emphasize GigE Industrial
        camera_text = f"Camera: {self.camera_model} (GigE Industrial)"
        cv2.putText(overlay, camera_text, (10, 55),
                   font, 0.55, (100, 255, 100), 1)
        
        # Object count and status
        obj_count = len(measurements_list) if measurements_list else 0
        filter_mode = "RAW" if self.show_raw else f"FILTERED ({self.filter_size}x)"
        status_text = f"Objects: {obj_count}  |  Mode: {filter_mode}"
        status_color = (200, 200, 200)
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
        if measurements_list:
            for idx, measurement in enumerate(measurements_list):
                color = colors[idx % len(colors)]
                
                rotated_box = measurement['box']
                obj_id = measurement['object_id']
                cx, cy = measurement['center_pixel']
                
                # Get display values
                if self.show_raw:
                    width_mm = measurement['width_mm_raw']
                    height_mm = measurement['height_mm_raw']
                else:
                    width_mm = measurement['width_mm_filtered']
                    height_mm = measurement['height_mm_filtered']
                
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
        if measurements_list:
            table_x = 10
            table_y = 110
            table_width = 510
            row_height = 30
            table_height = min(len(measurements_list) * row_height + 60, h - 250)
            
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
            for idx, measurement in enumerate(measurements_list[:8]):  # Limit to 8 visible
                row_y = header_y + 25 + idx * row_height
                color = colors[idx % len(colors)]
                obj_id = measurement['object_id']
                
                # Get display values
                if self.show_raw:
                    w_val = measurement['width_mm_raw']
                    h_val = measurement['height_mm_raw']
                    w_err = measurement.get('width_error_raw', 0)
                    h_err = measurement.get('height_error_raw', 0)
                else:
                    w_val = measurement['width_mm_filtered']
                    h_val = measurement['height_mm_filtered']
                    w_err = measurement.get('width_error_filtered', 0)
                    h_err = measurement.get('height_error_filtered', 0)
                
                area_val = measurement['area_mm2_filtered'] if not self.show_raw else measurement['area_mm2_raw']
                
                # Calculate angle from box (for display consistency)
                box = measurement['box']
                # Simple angle calculation from bounding box
                dx = box[1][0] - box[0][0]
                dy = box[1][1] - box[0][1]
                angle = np.degrees(np.arctan2(dy, dx))
                
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
                cv2.putText(overlay, f"{angle:5.1f}", (table_x + 310, row_y),
                           font, 0.45, (255, 255, 255), 1)
                cv2.putText(overlay, f"{area_val:4.0f}", (table_x + 370, row_y),
                           font, 0.45, (255, 255, 255), 1)
            
            # If more than 8 objects, show indicator
            if len(measurements_list) > 8:
                more_y = header_y + 25 + 8 * row_height
                cv2.putText(overlay, f"... and {len(measurements_list) - 8} more", 
                           (table_x + 10, more_y),
                           font, 0.4, (150, 150, 150), 1)
        
        # Help text at bottom
        help_y = h - 100
        cv2.rectangle(overlay, (0, help_y), (w, h), (0, 0, 0), -1)
        cv2.putText(overlay, "Controls: [Q]uit  [S]ave  [R]aw  [+/-]Filter", 
                   (10, help_y + 25), font, 0.5, (200, 200, 200), 1)
        cv2.putText(overlay, f"Homography Matrix  |  Filter: {self.filter_size} frames  |  FPS: {self.fps:.1f}  |  Objects: {obj_count}", 
                   (10, help_y + 50), font, 0.5, (200, 200, 200), 1)
        cv2.putText(overlay, f"Ground Truth: W={self.GROUND_TRUTH_WIDTH}mm H={self.GROUND_TRUTH_HEIGHT}mm  |  Error shown in orange", 
                   (10, help_y + 75), font, 0.4, (150, 150, 150), 1)
        
        return overlay
    
    def update_fps(self):
        """Calculate FPS"""
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
    
    def process_frame(self, image):
        """Process single frame and return measurements"""
        # Update FPS
        self.update_fps()
        
        # Preprocess
        mask = self.preprocess_image(image)
        
        # Find all objects
        contours = self.find_all_objects(mask)
        
        # Measure all objects (raw measurements)
        measurements = []
        for contour in contours:
            meas = self.measure_object(contour)
            measurements.append(meas)
        
        # Assign persistent IDs
        measurements = self.assign_object_ids(measurements)
        
        # Apply stabilization filter
        measurements = self.stabilize_measurements(measurements)
        
        # Draw overlay
        result = self.draw_industrial_overlay(image, measurements)
        
        return result, measurements

def setup_camera():
    """Setup Basler camera and return camera info"""
    print("\n🎥 Initializing Basler camera...")
    
    tlFactory = pylon.TlFactory.GetInstance()
    devices = tlFactory.EnumerateDevices()
    
    if len(devices) == 0:
        raise RuntimeError("No Basler camera found!")
    
    print(f"   Found {len(devices)} camera(s)")
    
    camera = pylon.InstantCamera(tlFactory.CreateDevice(devices[0]))
    camera.Open()
    
    # Get camera info
    camera_model = camera.GetDeviceInfo().GetModelName()
    camera_serial = camera.GetDeviceInfo().GetSerialNumber()
    
    print(f"   Model: {camera_model}")
    print(f"   Serial: {camera_serial}")
    
    camera.PixelFormat.SetValue("BayerBG12")
    camera.AcquisitionMode.SetValue("Continuous")
    
    print("✓ Camera configured and ready!")
    
    return camera, camera_model, camera_serial

def main():
    print("=" * 60)
    print("HOMOGRAPHY-BASED MEASUREMENT SYSTEM")
    print("Real-time Multi-Object Measurement with Filter")
    print("=" * 60)
    
    # Initialize measurement system
    try:
        system = TnutMeasurementSystem('homography_calibration.json')
    except FileNotFoundError:
        print("\nError: homography_calibration.json not found!")
        print("Please run the ArUco calibration script first.")
        return
    
    # Setup camera
    camera, camera_model, camera_serial = setup_camera()
    
    # Set camera info in measurement system
    system.camera_model = camera_model
    system.camera_serial = camera_serial
    
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    
    print("\n" + "=" * 60)
    print("System Running - Continuous Measurement")
    print("=" * 60)
    print("\nControls:")
    print("  Q      - Quit")
    print("  S      - Save current measurements")
    print("  R      - Toggle Raw/Filtered display")
    print("  + / =  - Increase filter strength")
    print("  - / _  - Decrease filter strength")
    print("\nFilter Settings:")
    print(f"  Current: {system.filter_size} frames (adjustable 1-20)")
    print("  Higher = More stable but slower response")
    print("  Lower  = Faster response but more noise")
    print("=" * 60 + "\n")
    
    # Window name
    window_name = "Homography Measurement System"
    
    try:
        # Create window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Get first frame to set window size
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            image = converter.Convert(grabResult)
            img = image.GetArray()
            grabResult.Release()
            
            # Set window size based on image dimensions
            height, width = img.shape[:2]
            
            # Scale down if image is too large
            max_display_width = 1280
            max_display_height = 720
            
            if width > max_display_width or height > max_display_height:
                scale = min(max_display_width / width, max_display_height / height)
                display_width = int(width * scale)
                display_height = int(height * scale)
            else:
                display_width = width
                display_height = height
            
            cv2.resizeWindow(window_name, display_width, display_height)
            print(f"✓ Window initialized: {display_width}x{display_height}\n")
        
        while camera.IsGrabbing():
            # Grab frame
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if not grabResult.GrabSucceeded():
                continue
            
            # Convert to BGR
            image = converter.Convert(grabResult)
            img = image.GetArray()
            grabResult.Release()
            
            # Process frame
            result_frame, measurements = system.process_frame(img)
            
            # Display
            cv2.imshow(window_name, result_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\n✓ Quitting...")
                break
            
            elif key == ord('s') or key == ord('S'):
                if measurements:
                    # Save measurements
                    save_data = {
                        'timestamp': datetime.now().isoformat(),
                        'num_objects': len(measurements),
                        'filter_size': system.filter_size,
                        'show_raw': system.show_raw,
                        'measurements': []
                    }
                    
                    for meas in measurements:
                        save_data['measurements'].append({
                            'object_id': meas['object_id'],
                            'width_mm_raw': float(meas['width_mm_raw']),
                            'height_mm_raw': float(meas['height_mm_raw']),
                            'area_mm2_raw': float(meas['area_mm2_raw']),
                            'width_mm_filtered': float(meas['width_mm_filtered']),
                            'height_mm_filtered': float(meas['height_mm_filtered']),
                            'area_mm2_filtered': float(meas['area_mm2_filtered']),
                            'pass': meas['pass']
                        })
                    
                    filename = f"measurements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, 'w') as f:
                        json.dump(save_data, f, indent=4)
                    
                    print(f"✓ Saved {len(measurements)} measurements to '{filename}'")
                else:
                    print("No objects detected to save")
            
            elif key == ord('r') or key == ord('R'):
                # Toggle raw/filtered display
                system.show_raw = not system.show_raw
                mode = "RAW (unfiltered)" if system.show_raw else f"FILTERED ({system.filter_size} frames)"
                print(f"📊 Display mode: {mode}")
            
            elif key == ord('+') or key == ord('='):
                # Increase filter size
                if system.filter_size < 20:
                    system.filter_size += 1
                    print(f"🔧 Filter size: {system.filter_size} frames (more stable)")
            
            elif key == ord('-') or key == ord('_'):
                # Decrease filter size
                if system.filter_size > 1:
                    system.filter_size -= 1
                    print(f"🔧 Filter size: {system.filter_size} frames (more responsive)")
                    if system.filter_size == 1:
                        print("   (Filter disabled - showing real-time values)")
    
    finally:
        camera.StopGrabbing()
        camera.Close()
        cv2.destroyAllWindows()
        print("\n✓ System shutdown complete")

if __name__ == "__main__":
    main()