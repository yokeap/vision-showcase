#!/usr/bin/env python3
"""
Real-Time Interactive Calibration Tool with Basler GigE Camera
==============================================================

This script allows you to calibrate pixel-to-mm conversion using live camera:
1. Live video from Basler camera
2. Freeze frame when ready
3. Click two points on a known distance
4. Save calibration parameters to JSON

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


class RealtimeCalibrator:
    """Real-time calibration tool using Basler camera and mouse clicks."""
    
    def __init__(self):
        """Initialize real-time calibrator."""
        self.camera = None
        self.frozen_frame = None
        self.display_image = None
        self.points = []
        self.is_calibrating = False
        self.window_name = "Real-Time Calibration - Press 'F' to freeze, then click points"
        
        print("=" * 70)
        print("  REAL-TIME CALIBRATION TOOL")
        print("  Basler GigE Camera")
        print("=" * 70)
    
    def setup_camera(self):
        """Setup Basler GigE camera."""
        print("\n🎥 Initializing Basler camera...")
        
        # Create camera instance
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()
        
        if len(devices) == 0:
            raise RuntimeError("No Basler camera found!")
        
        print(f"   Found {len(devices)} camera(s)")
        
        # Use first camera
        self.camera = pylon.InstantCamera(tlFactory.CreateDevice(devices[0]))
        self.camera.Open()
        
        # Print camera info
        print(f"   Model: {self.camera.GetDeviceInfo().GetModelName()}")
        print(f"   Serial: {self.camera.GetDeviceInfo().GetSerialNumber()}")
        
        # Configure camera settings
        self.camera.PixelFormat.SetValue("RGB8")
        self.camera.AcquisitionMode.SetValue("Continuous")
        
        print("✓ Camera configured and ready!")
        
        return self.camera
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click events during calibration."""
        if not self.is_calibrating:
            return
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                # Add point
                self.points.append((x, y))
                
                # Draw point on image
                cv2.circle(self.display_image, (x, y), 8, (0, 0, 255), -1)
                cv2.circle(self.display_image, (x, y), 10, (255, 255, 255), 2)
                cv2.putText(self.display_image, f"P{len(self.points)}", 
                           (x + 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 255), 2)
                
                print(f"✓ Point {len(self.points)}: ({x}, {y})")
                
                # If we have 2 points, draw line between them
                if len(self.points) == 2:
                    cv2.line(self.display_image, self.points[0], self.points[1], 
                            (0, 255, 0), 3)
                    
                    # Calculate pixel distance
                    pixel_dist = np.sqrt(
                        (self.points[1][0] - self.points[0][0])**2 + 
                        (self.points[1][1] - self.points[0][1])**2
                    )
                    
                    # Draw distance text
                    mid_x = (self.points[0][0] + self.points[1][0]) // 2
                    mid_y = (self.points[0][1] + self.points[1][1]) // 2
                    
                    text = f"{pixel_dist:.1f} pixels"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(self.display_image,
                                 (mid_x - text_size[0]//2 - 5, mid_y - text_size[1] - 10),
                                 (mid_x + text_size[0]//2 + 5, mid_y - 5),
                                 (0, 0, 0), -1)
                    cv2.putText(self.display_image, text,
                               (mid_x - text_size[0]//2, mid_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    print(f"\n📏 Pixel distance: {pixel_dist:.2f} pixels")
                    print("✓ Two points selected!")
                    print("  Press ENTER to continue or 'R' to reset points")
    
    def draw_overlay(self, image: np.ndarray, is_frozen: bool) -> np.ndarray:
        """Draw instruction overlay on image."""
        overlay = image.copy()
        h, w = overlay.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Status bar at top
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        
        # Title
        title = "REAL-TIME CALIBRATION"
        cv2.putText(overlay, title, (10, 35), font, 1.0, (0, 255, 0), 2)
        
        # Status
        if not is_frozen:
            status = "LIVE - Press 'F' to FREEZE frame"
            status_color = (0, 255, 0)
        elif len(self.points) == 0:
            status = "FROZEN - Click 2 points on known distance"
            status_color = (0, 255, 255)
        elif len(self.points) == 1:
            status = "FROZEN - Click second point"
            status_color = (0, 255, 255)
        else:
            status = "FROZEN - Press ENTER to finish or 'R' to reset"
            status_color = (0, 255, 255)
        
        cv2.putText(overlay, status, (10, 70), font, 0.6, status_color, 2)
        
        # Point counter
        if is_frozen:
            points_text = f"Points selected: {len(self.points)}/2"
            cv2.putText(overlay, points_text, (10, 100), font, 0.5, (200, 200, 200), 1)
        
        # Help text at bottom
        help_y = h - 100
        cv2.rectangle(overlay, (0, help_y), (w, h), (0, 0, 0), -1)
        
        if not is_frozen:
            cv2.putText(overlay, "Controls: [F] Freeze frame  [Q] Quit", 
                       (10, help_y + 30), font, 0.6, (200, 200, 200), 1)
            cv2.putText(overlay, "Position your ruler/calibration target in view", 
                       (10, help_y + 60), font, 0.5, (150, 150, 150), 1)
        else:
            cv2.putText(overlay, "Controls: [ENTER] Finish  [R] Reset points  [U] Unfreeze  [Q] Quit", 
                       (10, help_y + 30), font, 0.6, (200, 200, 200), 1)
            cv2.putText(overlay, "Click on two points with known distance apart", 
                       (10, help_y + 60), font, 0.5, (150, 150, 150), 1)
        
        return overlay
    
    def run(self):
        """Main real-time calibration loop."""
        
        # Setup camera
        camera = self.setup_camera()
        
        # Start grabbing
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        
        # Converter for BGR format
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        
        print("\n" + "=" * 70)
        print("  CALIBRATION STARTED")
        print("=" * 70)
        print("\nInstructions:")
        print("  1. Position your ruler/calibration target in camera view")
        print("  2. Press 'F' to FREEZE the frame")
        print("  3. Click on TWO points with KNOWN distance apart")
        print("  4. Press ENTER when done")
        print("=" * 70)
        
        is_frozen = False
        
        try:
            # Setup window
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)

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
                
                cv2.resizeWindow(self.window_name, display_width, display_height)
                print(f"✓ Window initialized: {display_width}x{display_height}\n")
            
            while camera.IsGrabbing():
                # Get current frame
                if not is_frozen:
                    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                    
                    if grabResult.GrabSucceeded():
                        # Convert to BGR
                        image = converter.Convert(grabResult)
                        img = image.GetArray()
                        self.frozen_frame = img.copy()
                    
                    grabResult.Release()
                else:
                    # Use frozen frame
                    img = self.frozen_frame
                
                # Create display image
                if self.display_image is None or not is_frozen:
                    self.display_image = img.copy()
                
                # Draw overlay
                display = self.draw_overlay(self.display_image, is_frozen)
                
                # Show frame
                cv2.imshow(self.window_name, display)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\n⚠️  Calibration cancelled")
                    return None
                
                elif key == ord('f') or key == ord('F'):
                    if not is_frozen:
                        # Freeze frame
                        is_frozen = True
                        self.is_calibrating = True
                        self.display_image = self.frozen_frame.copy()
                        self.points = []
                        print("\n⏸  Frame FROZEN - Click 2 points on ruler")
                
                elif key == ord('u') or key == ord('U'):
                    if is_frozen:
                        # Unfreeze
                        is_frozen = False
                        self.is_calibrating = False
                        self.points = []
                        self.display_image = None
                        print("\n▶  Frame UNFROZEN - Position target and press 'F' again")
                
                elif key == ord('r') or key == ord('R'):
                    if is_frozen:
                        # Reset points
                        self.points = []
                        self.display_image = self.frozen_frame.copy()
                        print("\n🔄 Points reset - Click 2 new points")
                
                elif key == 13 or key == 10:  # ENTER key
                    if is_frozen and len(self.points) == 2:
                        # Done selecting points
                        print("\n✓ Point selection complete!")
                        break
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
            return None
        
        finally:
            camera.StopGrabbing()
            camera.Close()
            cv2.destroyAllWindows()
        
        # Return the frozen frame with points
        return {
            'image': self.frozen_frame,
            'points': self.points
        }
    
    def calculate_calibration(self, result: dict, known_distance_mm: float):
        """
        Calculate calibration parameters.
        
        Args:
            result: Result dict from run() containing image and points
            known_distance_mm: Known distance between points in mm
            
        Returns:
            dict: Calibration parameters
        """
        if len(result['points']) != 2:
            raise ValueError("Need exactly 2 points for calibration!")
        
        points = result['points']
        image = result['image']
        
        # Calculate pixel distance
        pixel_distance = np.sqrt(
            (points[1][0] - points[0][0])**2 + 
            (points[1][1] - points[0][1])**2
        )
        
        # Calculate pixels per mm
        pixels_per_mm = pixel_distance / known_distance_mm
        
        # Create calibration data
        calibration = {
            "calibration_date": datetime.now().isoformat(),
            "image_source": "Basler a2A1920-51gcBAS (live capture)",
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "point1": {
                "x": int(points[0][0]),
                "y": int(points[0][1])
            },
            "point2": {
                "x": int(points[1][0]),
                "y": int(points[1][1])
            },
            "known_distance_mm": float(known_distance_mm),
            "pixel_distance": float(pixel_distance),
            "pixels_per_mm": float(pixels_per_mm),
            "mm_per_pixel": float(1.0 / pixels_per_mm)
        }
        
        return calibration
    
    def save_calibration(self, calibration: dict, output_path: str = "calibration.json"):
        """Save calibration to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(calibration, f, indent=4)
        
        print("\n" + "=" * 70)
        print("  CALIBRATION SAVED")
        print("=" * 70)
        print(f"\nFile: {output_path}")
        print(f"\nCalibration Parameters:")
        print(f"  Known distance:    {calibration['known_distance_mm']:.2f} mm")
        print(f"  Pixel distance:    {calibration['pixel_distance']:.2f} pixels")
        print(f"  Pixels per mm:     {calibration['pixels_per_mm']:.4f} px/mm")
        print(f"  MM per pixel:      {calibration['mm_per_pixel']:.6f} mm/px")
        print("\n✓ Ready to use for measurement!")


def main():
    """Main calibration workflow."""
    
    OUTPUT_JSON = "calibration.json"
    SAVE_SNAPSHOT = True  # Save the calibration image
    
    print("\n" + "=" * 70)
    print("  REAL-TIME PIXEL-TO-MM CALIBRATION TOOL")
    print("  Live Camera Calibration")
    print("=" * 70)
    print("\nHardware: Basler a2A1920-51gcBAS")
    print("Platform: Recomputer J4012")
    print("Method: Live camera + Interactive point selection")
    print("=" * 70)
    
    try:
        # Create calibrator
        calibrator = RealtimeCalibrator()
        
        # Run live calibration
        result = calibrator.run()
        
        if result is None:
            print("\n❌ Calibration cancelled or failed")
            return
        
        # Get known distance from user
        print("\n" + "=" * 70)
        print("  ENTER KNOWN DISTANCE")
        print("=" * 70)
        print("\nExamples:")
        print("  • Ruler: 1 cm = 10 mm, 5 cm = 50 mm")
        print("  • Calibration target: Measure with vernier caliper")
        print("  • Known object: Use precise measurement")

        
        
        while True:
            try:
                distance_input = input("\nEnter distance between points (in mm): ")
                known_distance_mm = float(distance_input)
                
                if known_distance_mm <= 0:
                    print("⚠️  Distance must be positive. Try again.")
                    continue
                
                break
            except ValueError:
                print("⚠️  Invalid input. Enter a number (e.g., 10, 25.5, 100)")
        
        # Calculate calibration
        print("\n📐 Calculating calibration parameters...")
        calibration = calibrator.calculate_calibration(result, known_distance_mm)
        
        # Save to JSON
        calibrator.save_calibration(calibration, OUTPUT_JSON)
        
        # Optionally save the calibration snapshot
        if SAVE_SNAPSHOT:
            snapshot_path = "calibration_snapshot.png"
            # Draw points on image
            img_with_points = result['image'].copy()
            p1, p2 = result['points']
            cv2.circle(img_with_points, p1, 8, (0, 0, 255), -1)
            cv2.circle(img_with_points, p2, 8, (0, 0, 255), -1)
            cv2.line(img_with_points, p1, p2, (0, 255, 0), 3)
            cv2.imwrite(snapshot_path, img_with_points)
            print(f"✓ Calibration snapshot saved: {snapshot_path}")
        
        print("\n" + "=" * 70)
        print("  CALIBRATION COMPLETE!")
        print("=" * 70)
        print(f"\n✓ Calibration saved to: {OUTPUT_JSON}")
        print("✓ Ready for real-time measurement!")
        print("\nNext step: Run realtime_measurement.py")
        
    except RuntimeError as e:
        print(f"\n❌ Camera Error: {e}")
        print("\nTroubleshooting:")
        print("  • Is the Basler camera connected?")
        print("  • Check network settings (GigE camera)")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()