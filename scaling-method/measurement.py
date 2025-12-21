#!/usr/bin/env python3
"""
Automatic Measurement Tool with HSV Color Thresholding
=======================================================

This script uses HSV color space to remove blue background and detect objects.
Designed for industrial-style measurement display.

Author: Siwakorn - Kasetsart University
Hardware: Basler a2A1920-51gcBAS on Recomputer J4012
"""

import cv2
import numpy as np
import json
from pathlib import Path


class ObjectMeasurement:
    """Automatic object detection and measurement using HSV color thresholding."""
    
    def __init__(self, calibration_path: str = "calibration.json"):
        """
        Initialize measurement system with calibration.
        
        Args:
            calibration_path: Path to calibration JSON file
        """
        self.calibration = self.load_calibration(calibration_path)
        self.pixels_per_mm = self.calibration['pixels_per_mm']
        self.mm_per_pixel = self.calibration['mm_per_pixel']
        
        print("=" * 70)
        print("  AUTOMATIC MEASUREMENT SYSTEM")
        print("  HSV Color Thresholding Method")
        print("=" * 70)
        print(f"\nCalibration loaded: {calibration_path}")
        print(f"Scale: {self.pixels_per_mm:.4f} pixels/mm")
        print(f"       {self.mm_per_pixel:.6f} mm/pixel")
    
    def load_calibration(self, path: str) -> dict:
        """Load calibration from JSON file."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")
        
        with open(path, 'r') as f:
            calibration = json.load(f)
        
        return calibration
    
    def detect_object_hsv(self, image: np.ndarray,
                         min_area: int = 500) -> list:
        """
        Detect object using HSV color thresholding to remove blue background.
        
        Args:
            image: Input BGR image
            min_area: Minimum contour area to consider
            
        Returns:
            Tuple of (contours list, binary mask)
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define blue color range in HSV
        # Hue: 90-130 (blue/cyan range)
        # Saturation: 50-255 (avoid very desaturated colors)
        # Value: 50-255 (avoid very dark colors)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create mask for blue color
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Invert mask (we want NON-blue areas = object)
        object_mask = cv2.bitwise_not(blue_mask)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(object_mask, (5, 5), 0)
        
        # Threshold to clean up
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Remove small noise
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.erode(thresh, kernel_erode, iterations=1)
        thresh = cv2.dilate(thresh, kernel_erode, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area and sort by size
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
        
        return valid_contours, thresh
    
    def measure_object(self, contour: np.ndarray) -> dict:
        """
        Measure object dimensions from contour.
        
        Args:
            contour: Object contour from cv2.findContours
            
        Returns:
            dict: Measurement data (width, height, center, etc.)
        """
        # Get bounding rectangle
        x, y, w_px, h_px = cv2.boundingRect(contour)
        
        # Convert to millimeters
        width_mm = w_px * self.mm_per_pixel
        height_mm = h_px * self.mm_per_pixel
        
        # Calculate center point
        center_x = x + w_px // 2
        center_y = y + h_px // 2
        
        # Calculate area
        area_px = cv2.contourArea(contour)
        area_mm2 = area_px * (self.mm_per_pixel ** 2)
        
        # Get minimum area rectangle (for rotated bounding box)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Get oriented width and height
        (cx, cy), (w_rot, h_rot), angle = rect
        if w_rot < h_rot:
            w_rot, h_rot = h_rot, w_rot
            angle += 90
        
        width_rot_mm = w_rot * self.mm_per_pixel
        height_rot_mm = h_rot * self.mm_per_pixel
        
        measurement = {
            'bounding_box': (x, y, w_px, h_px),
            'width_mm': width_mm,
            'height_mm': height_mm,
            'width_px': w_px,
            'height_px': h_px,
            'center': (center_x, center_y),
            'area_mm2': area_mm2,
            'area_px': area_px,
            'rotated_rect': {
                'width_mm': width_rot_mm,
                'height_mm': height_rot_mm,
                'angle': angle,
                'box': box
            }
        }
        
        return measurement
    
    def draw_measurement(self, image: np.ndarray, contour: np.ndarray, 
                        measurement: dict, show_details: bool = True) -> np.ndarray:
        """
        Draw industrial-style measurement annotations on image.
        
        Args:
            image: Input image
            contour: Object contour
            measurement: Measurement dictionary
            show_details: Show detailed measurements
            
        Returns:
            Annotated image
        """
        result = image.copy()
        
        # Extract measurement data
        x, y, w, h = measurement['bounding_box']
        width_mm = measurement['width_mm']
        height_mm = measurement['height_mm']
        cx, cy = measurement['center']
        
        # Colors (BGR format)
        COLOR_BOX = (0, 255, 0)      # Green
        COLOR_TEXT = (255, 255, 255)  # White
        COLOR_DIM_LINES = (255, 0, 0) # Blue
        
        # Draw ROTATED bounding box (this is what we measure - TRUE dimensions)
        cv2.drawContours(result, [measurement['rotated_rect']['box']], 0, COLOR_BOX, 3)
        
        # Draw center point
        cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
        cv2.line(result, (cx - 10, cy), (cx + 10, cy), (0, 0, 255), 2)
        cv2.line(result, (cx, cy - 10), (cx, cy + 10), (0, 0, 255), 2)
        
        # Draw dimension lines (industrial style)
        # Width dimension line (top)
        line_offset = 40
        cv2.line(result, (x, y - line_offset), (x + w, y - line_offset), 
                COLOR_DIM_LINES, 2)
        cv2.line(result, (x, y - line_offset - 10), (x, y - line_offset + 10), 
                COLOR_DIM_LINES, 2)
        cv2.line(result, (x + w, y - line_offset - 10), (x + w, y - line_offset + 10), 
                COLOR_DIM_LINES, 2)
        
        # Height dimension line (right)
        cv2.line(result, (x + w + line_offset, y), (x + w + line_offset, y + h), 
                COLOR_DIM_LINES, 2)
        cv2.line(result, (x + w + line_offset - 10, y), (x + w + line_offset + 10, y), 
                COLOR_DIM_LINES, 2)
        cv2.line(result, (x + w + line_offset - 10, y + h), (x + w + line_offset + 10, y + h), 
                COLOR_DIM_LINES, 2)
        
        # Add measurement text (industrial style)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        
        # Width text (above object)
        width_text = f"W: {width_mm:.2f} mm"
        text_size = cv2.getTextSize(width_text, font, font_scale, font_thickness)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y - line_offset - 15
        
        # Add background rectangle for text
        cv2.rectangle(result, 
                     (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5),
                     (0, 0, 0), -1)
        cv2.putText(result, width_text, (text_x, text_y),
                   font, font_scale, COLOR_TEXT, font_thickness)
        
        # Height text (right of object)
        height_text = f"H: {height_mm:.2f} mm"
        text_size = cv2.getTextSize(height_text, font, font_scale, font_thickness)[0]
        text_x = x + w + line_offset + 15
        text_y = y + h // 2
        
        # Add background rectangle for text
        cv2.rectangle(result, 
                     (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5),
                     (0, 0, 0), -1)
        cv2.putText(result, height_text, (text_x, text_y),
                   font, font_scale, COLOR_TEXT, font_thickness)
        
        if show_details:
            # Add title and additional info at top
            title_y = 35
            cv2.rectangle(result, (5, 5), (550, 165), (0, 0, 0), -1)
            cv2.putText(result, "OUTER DIMENSION MEASUREMENT", (10, title_y),
                       font, 1.0, COLOR_TEXT, 2)
            
            # Measurement details
            info_y = 70
            line_spacing = 25
            
            details = [
                f"Width:  {width_mm:.2f} mm ({measurement['width_px']} px)",
                f"Height: {height_mm:.2f} mm ({measurement['height_px']} px)",
                f"Area:   {measurement['area_mm2']:.2f} mm2",
                f"Angle:  {measurement['rotated_rect']['angle']:.1f} deg",
            ]
            
            for i, detail in enumerate(details):
                y_pos = info_y + i * line_spacing
                cv2.putText(result, detail, (10, y_pos),
                           font, 0.6, COLOR_TEXT, 1)
        
        return result
    
    def process_image(self, object_path: str, 
                     output_path: str = None,
                     show_image: bool = True) -> dict:
        """
        Process image: detect object and measure using HSV color thresholding.
        
        Args:
            object_path: Path to object image
            output_path: Path to save result (optional)
            show_image: Display result window
            
        Returns:
            dict: Measurement results
        """
        print("\n" + "=" * 70)
        print("  PROCESSING IMAGE")
        print("=" * 70)
        
        # Load image
        object_img = cv2.imread(object_path)
        
        if object_img is None:
            raise FileNotFoundError(f"Cannot load object image: {object_path}")
        
        print(f"\nObject image: {object_path}")
        print(f"Size: {object_img.shape[1]} x {object_img.shape[0]} pixels")
        
        # Detect objects using HSV color thresholding
        print("\n🔍 Detecting object using HSV color thresholding...")
        print("   Method: Removing blue background color")
        contours, mask = self.detect_object_hsv(object_img)
        
        if len(contours) == 0:
            print("❌ No objects detected!")
            print("   The HSV thresholds may need adjustment")
            
            # Show the mask for debugging
            cv2.imshow("Detection Mask - Check this!", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return None
        
        print(f"✓ Found {len(contours)} object(s)")
        
        # Measure the largest object (T-nut)
        main_contour = contours[0]
        print(f"  Using largest contour (area: {cv2.contourArea(main_contour):.0f} px²)")
        
        print("\n📏 Measuring object...")
        measurement = self.measure_object(main_contour)
        
        # Draw measurement
        print("🎨 Creating visualization...")
        result_image = self.draw_measurement(object_img, main_contour, measurement)
        
        # Print results
        print("\n" + "=" * 70)
        print("  MEASUREMENT RESULTS (OUTER BOUNDING BOX)")
        print("=" * 70)
        print(f"\n  Width:  {measurement['width_mm']:7.2f} mm  ({measurement['width_px']} px)")
        print(f"  Height: {measurement['height_mm']:7.2f} mm  ({measurement['height_px']} px)")
        print(f"  Area:   {measurement['area_mm2']:7.2f} mm²")
        print(f"  Angle:  {measurement['rotated_rect']['angle']:7.1f} degrees")
        print("\n  Method: HSV color thresholding (blue background removal)")
        print("=" * 70)
        
        # Save result if output path specified
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"\n✓ Result saved to: {output_path}")
        
        # Display result
        if show_image:
            # Show mask for verification
            cv2.imshow("Detection Mask", mask)
            
            # Show result
            window_name = "Measurement Result - Press any key to close"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, result_image)
            print("\n💡 Press any key to close windows...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return measurement


def main():
    """Main measurement workflow."""
    
    # Configuration
    CALIBRATION_FILE = "calibration.json"
    OBJECT_IMAGE = "object-2.png"      # Image with T-nut
    OUTPUT_PATH = "measurement_result.png"
    
    print("\n" + "=" * 70)
    print("  INDUSTRIAL MEASUREMENT SYSTEM")
    print("  HSV Color Thresholding Method")
    print("=" * 70)
    print("\nHardware: Basler a2A1920-51gcBAS on Recomputer J4012")
    print("Method: HSV blue background removal + Pixel-to-MM scaling")
    print("=" * 70)
    
    try:
        # Initialize measurement system
        measurer = ObjectMeasurement(CALIBRATION_FILE)
        
        # Process image
        result = measurer.process_image(
            object_path=OBJECT_IMAGE,
            output_path=OUTPUT_PATH,
            show_image=True
        )
        
        if result:
            print("\n✓ Measurement complete!")
            print(f"✓ Result image saved: {OUTPUT_PATH}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  • calibration.json (run calibrator.py first)")
        print("  • object-2.png (image with T-nut on blue background)")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()