#!/usr/bin/env python3
"""
Interactive Calibration Tool
============================

This script allows you to calibrate pixel-to-mm conversion by:
1. Loading a calibration image (ruler)
2. Clicking two points on a known distance
3. Saving the calibration parameters to JSON

Author: Siwakorn - Kasetsart University
Hardware: Basler a2A1920-51gcBAS on Recomputer J4012
"""

import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime


class InteractiveCalibrator:
    """Interactive calibration tool using mouse clicks."""
    
    def __init__(self, image_path: str):
        """
        Initialize calibrator with image.
        
        Args:
            image_path: Path to calibration image (ruler)
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        
        if self.image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        
        # Create a copy for drawing
        self.display_image = self.image.copy()
        
        # Store clicked points
        self.points = []
        self.window_name = "Calibration - Click two points on ruler"
        
        print("=" * 70)
        print("  INTERACTIVE CALIBRATION TOOL")
        print("=" * 70)
        print(f"\nImage loaded: {image_path}")
        print(f"Image size: {self.image.shape[1]} x {self.image.shape[0]} pixels")
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                # Add point
                self.points.append((x, y))
                
                # Draw point on image
                cv2.circle(self.display_image, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(self.display_image, f"P{len(self.points)}: ({x},{y})", 
                           (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 0, 255), 1)
                
                print(f"✓ Point {len(self.points)}: ({x}, {y})")
                
                # If we have 2 points, draw line between them
                if len(self.points) == 2:
                    cv2.line(self.display_image, self.points[0], self.points[1], 
                            (0, 255, 0), 2)
                    
                    # Calculate pixel distance
                    pixel_dist = np.sqrt(
                        (self.points[1][0] - self.points[0][0])**2 + 
                        (self.points[1][1] - self.points[0][1])**2
                    )
                    
                    print(f"\n📏 Pixel distance: {pixel_dist:.2f} pixels")
                    print("\n✓ Two points selected!")
                    print("  Close the window to continue...")
                
                # Update display
                cv2.imshow(self.window_name, self.display_image)
    
    def select_points(self):
        """
        Interactive point selection.
        
        Returns:
            List of two points [(x1, y1), (x2, y2)]
        """
        print("\n" + "=" * 70)
        print("  INSTRUCTIONS")
        print("=" * 70)
        print("\n1. Click on the FIRST point on the ruler (e.g., at 0 cm mark)")
        print("2. Click on the SECOND point on the ruler (e.g., at 1 cm mark)")
        print("3. The points should span a KNOWN distance")
        print("\n   Example: Click at start and end of 1 cm = 10 mm distance")
        print("            Click at 0 cm and 5 cm marks = 50 mm distance")
        print("\n4. Close the window when done\n")
        
        # Setup window and mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Display image
        cv2.imshow(self.window_name, self.display_image)
        
        print("🖱️  Waiting for mouse clicks...")
        
        # Wait for user to close window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return self.points
    
    def calculate_calibration(self, known_distance_mm: float):
        """
        Calculate calibration parameters.
        
        Args:
            known_distance_mm: Known distance between the two points in millimeters
            
        Returns:
            dict: Calibration parameters
        """
        if len(self.points) != 2:
            raise ValueError("Need exactly 2 points for calibration!")
        
        # Calculate pixel distance
        pixel_distance = np.sqrt(
            (self.points[1][0] - self.points[0][0])**2 + 
            (self.points[1][1] - self.points[0][1])**2
        )
        
        # Calculate pixels per mm
        pixels_per_mm = pixel_distance / known_distance_mm
        
        # Create calibration data
        calibration = {
            "calibration_date": datetime.now().isoformat(),
            "image_path": str(self.image_path),
            "image_width": int(self.image.shape[1]),
            "image_height": int(self.image.shape[0]),
            "point1": {
                "x": int(self.points[0][0]),
                "y": int(self.points[0][1])
            },
            "point2": {
                "x": int(self.points[1][0]),
                "y": int(self.points[1][1])
            },
            "known_distance_mm": float(known_distance_mm),
            "pixel_distance": float(pixel_distance),
            "pixels_per_mm": float(pixels_per_mm),
            "mm_per_pixel": float(1.0 / pixels_per_mm)
        }
        
        return calibration
    
    def save_calibration(self, calibration: dict, output_path: str = "calibration.json"):
        """
        Save calibration to JSON file.
        
        Args:
            calibration: Calibration dictionary
            output_path: Output JSON file path
        """
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
    
    # Configuration
    IMAGE_PATH = "bg-scale.png"  # Change this to your calibration image path
    OUTPUT_JSON = "calibration.json"
    
    print("\n" + "=" * 70)
    print("  PIXEL-TO-MM CALIBRATION TOOL")
    print("  Interactive Point Selection Method")
    print("=" * 70)
    print("\nHardware: Basler a2A1920-51gcBAS")
    print("Platform: Recomputer J4012")
    print("Calibration: Ruler with 0.1 cm (1 mm) resolution")
    print("=" * 70)
    
    # Check if image exists
    if not Path(IMAGE_PATH).exists():
        print(f"\n❌ Error: Image not found: {IMAGE_PATH}")
        print("   Please update IMAGE_PATH in the script")
        return
    
    try:
        # Create calibrator
        calibrator = InteractiveCalibrator(IMAGE_PATH)
        
        # Select points interactively
        points = calibrator.select_points()
        
        if len(points) != 2:
            print("\n❌ Error: Need exactly 2 points for calibration")
            print("   Please run the script again")
            return
        
        # Get known distance from user
        print("\n" + "=" * 70)
        print("  ENTER KNOWN DISTANCE")
        print("=" * 70)
        print("\nYour ruler has marks every 0.1 cm (1 mm)")
        print("\nExamples:")
        print("  • 10 ruler marks apart = 10 mm")
        print("  • 1 cm = 10 mm")
        print("  • 5 cm = 50 mm")
        
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
        calibration = calibrator.calculate_calibration(known_distance_mm)
        
        # Save to JSON
        calibrator.save_calibration(calibration, OUTPUT_JSON)
        
        print("\n" + "=" * 70)
        print("  CALIBRATION COMPLETE!")
        print("=" * 70)
        print(f"\n✓ Calibration saved to: {OUTPUT_JSON}")
        print("✓ Use this file in your measurement script")
        print("\nNext step: Run your measurement script with this calibration")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()