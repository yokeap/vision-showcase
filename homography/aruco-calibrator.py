import cv2
import numpy as np
import json
from pypylon import pylon
import time

# ArUco marker configuration - FIXED for your marker
MARKER_SIZE_MM = 25.0  # Physical size of marker in mm
ARUCO_DICT = cv2.aruco.DICT_5X5_50  # Your marker is from this dictionary
MARKER_ID = 23  # Your marker ID

def setup_camera():
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
    camera.PixelFormat.SetValue("BayerBG12")
    
    # Set acquisition mode to continuous
    camera.AcquisitionMode.SetValue("Continuous")
    
    print("✓ Camera configured and ready!")
    print(f"   Looking for Marker ID: {MARKER_ID} (5x5_50, {MARKER_SIZE_MM}mm)")
    
    return camera

def preprocess_image(image):
    """
    Preprocess image to isolate marker from blue background
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for blue background
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Create mask for blue background
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Invert mask to get the marker
    marker_mask = cv2.bitwise_not(blue_mask)
    
    # Clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    marker_mask = cv2.morphologyEx(marker_mask, cv2.MORPH_CLOSE, kernel)
    marker_mask = cv2.morphologyEx(marker_mask, cv2.MORPH_OPEN, kernel)
    
    # Apply mask to original image
    result = cv2.bitwise_and(image, image, mask=marker_mask)
    
    # Convert masked areas to white
    result[marker_mask == 0] = [255, 255, 255]
    
    return result, marker_mask

def detect_aruco_marker_5x5(preprocessed_image):
    """Detect ArUco marker - optimized for 5x5_50 dictionary"""
    # Convert to grayscale
    gray = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Create ArUco detector for 5x5_50 only
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
    parameters = cv2.aruco.DetectorParameters_create()
    
    # Adjust detection parameters for better detection
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.adaptiveThreshConstant = 7
    parameters.minMarkerPerimeterRate = 0.03
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.polygonalApproxAccuracyRate = 0.05
    parameters.minCornerDistanceRate = 0.05
    parameters.minDistanceToBorder = 3
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 30
    parameters.cornerRefinementMinAccuracy = 0.1
    
    # Try both binary and gray
    for test_img in [binary, gray]:
        corners, ids, rejected = cv2.aruco.detectMarkers(test_img, aruco_dict, parameters=parameters)
        
        if ids is not None and len(ids) > 0:
            return corners, ids, binary
    
    return None, None, binary

def calculate_homography_from_marker(marker_corners, marker_size_mm):
    """
    Calculate homography from a single marker
    """
    # Define world coordinates (mm)
    world_points = np.array([
        [0, 0],
        [marker_size_mm, 0],
        [marker_size_mm, marker_size_mm],
        [0, marker_size_mm]
    ], dtype=np.float32)
    
    # Image points (pixels)
    image_points = marker_corners.astype(np.float32)
    
    # Calculate homography
    H, status = cv2.findHomography(world_points, image_points)
    
    return H, world_points, image_points

def pixel_to_world(H, pixel_points):
    """Convert pixel coordinates to world coordinates (mm)"""
    pixel_points = np.array(pixel_points)
    if pixel_points.ndim == 1:
        pixel_points = pixel_points.reshape(1, -1)
    
    ones = np.ones((len(pixel_points), 1))
    pixel_homogeneous = np.hstack([pixel_points, ones])
    
    H_inv = np.linalg.inv(H)
    world_homogeneous = (H_inv @ pixel_homogeneous.T).T
    world_points = world_homogeneous[:, :2] / world_homogeneous[:, 2:]
    
    return world_points

def measure_marker_dimensions(corners, H):
    """Measure marker width and height in mm using homography"""
    # Convert corners to world coordinates
    world_corners = pixel_to_world(H, corners)
    
    # Calculate distances
    width_mm = np.linalg.norm(world_corners[1] - world_corners[0])
    height_mm = np.linalg.norm(world_corners[2] - world_corners[1])
    
    return width_mm, height_mm

def draw_industrial_overlay(image, corners, ids, H, fps, reprojection_error):
    """Draw industrial overlay with measurements and homography matrix"""
    overlay = image.copy()
    
    # Draw detected markers
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
        
        # Get marker corners
        marker_corners = corners[0][0]
        
        # Measure dimensions
        width_mm, height_mm = measure_marker_dimensions(marker_corners, H)
        
        # Draw corner points
        for i, corner in enumerate(marker_corners):
            cv2.circle(overlay, tuple(corner.astype(int)), 5, (0, 255, 0), -1)
            # Corner number
            cv2.putText(overlay, str(i), tuple(corner.astype(int) + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Calculate positions for width and height labels
        # Width label (top edge)
        top_mid = ((marker_corners[0] + marker_corners[1]) / 2).astype(int)
        width_text = f"W: {width_mm:.1f}mm"
        cv2.putText(overlay, width_text, tuple(top_mid - [0, 15]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Height label (right edge)
        right_mid = ((marker_corners[1] + marker_corners[2]) / 2).astype(int)
        height_text = f"H: {height_mm:.1f}mm"
        cv2.putText(overlay, height_text, tuple(right_mid + [15, 0]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Draw measurement lines
        cv2.line(overlay, tuple(marker_corners[0].astype(int)), 
                tuple(marker_corners[1].astype(int)), (0, 255, 255), 2)
        cv2.line(overlay, tuple(marker_corners[1].astype(int)), 
                tuple(marker_corners[2].astype(int)), (0, 255, 255), 2)
    
    # HUD panel
    panel_height = 150
    cv2.rectangle(overlay, (0, 0), (image.shape[1], panel_height), (0, 0, 0), -1)
    
    # Title and FPS
    cv2.putText(overlay, "ARUCO CALIBRATION", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    
    cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Detection status
    if ids is not None and len(ids) > 0:
        status_text = f"ID:{ids[0][0]} 5X5_50"
        status_color = (0, 255, 0)
        cv2.circle(overlay, (image.shape[1] - 30, 25), 10, (0, 255, 0), -1)
    else:
        status_text = "No marker"
        status_color = (0, 0, 255)
        cv2.circle(overlay, (image.shape[1] - 30, 25), 10, (0, 0, 255), -1)
    
    cv2.putText(overlay, status_text, (200, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)
    
    # Homography Matrix overlay (compact format)
    if H is not None:
        cv2.putText(overlay, "Homography:", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Display matrix in compact format - one row per line
        y_pos = 105
        for i, row in enumerate(H):
            row_text = f"[{row[0]:7.2f} {row[1]:7.2f} {row[2]:7.2f}]"
            cv2.putText(overlay, row_text, (10, y_pos + i*15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        
        # Reprojection error
        if reprojection_error is not None:
            error_text = f"Error: {reprojection_error:.4f}px"
            cv2.putText(overlay, error_text, (180, 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    
    # Controls
    cv2.putText(overlay, "'c':Save 's':Debug 'q':Quit", 
               (image.shape[1] - 280, image.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    
    return overlay

def main():
    print("ArUco Marker Homography Calibration - ID 23 (5x5_50)")
    print("=" * 60)
    print("\nPlace the marker in view and press 'c' to capture")
    print("Press 's' to show preprocessing steps")
    print("Press 'q' to quit")
    print("=" * 60)
    
    # Setup camera
    camera = setup_camera()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    
    show_preprocessing = False
    
    # FPS calculation
    fps = 0
    last_time = time.time()
    
    # Current homography
    current_H = None
    current_error = None
    
    # Window name
    window_name = "ArUco Calibration"
    
    try:
        # Create window with proper size
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Get first frame to determine size
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            image = converter.Convert(grabResult)
            img = image.GetArray()
            grabResult.Release()
            
            # Set window size to match image or reasonable display size
            height, width = img.shape[:2]
            
            # If image is too large, scale down for display
            max_display_width = 1280
            max_display_height = 800
            
            if width > max_display_width or height > max_display_height:
                scale = min(max_display_width / width, max_display_height / height)
                display_width = int(width * scale)
                display_height = int(height * scale)
            else:
                display_width = width
                display_height = height
            
            cv2.resizeWindow(window_name, display_width, display_height)
            print(f"\n✓ Window initialized: {display_width}x{display_height}")
        
        while camera.IsGrabbing():
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0
            last_time = current_time
            
            # Grab frame
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if not grabResult.GrabSucceeded():
                continue

            # Convert to BGR
            image = converter.Convert(grabResult)
            img = image.GetArray()
            grabResult.Release()
            
            # Preprocess image
            preprocessed, mask = preprocess_image(img)
            
            # Detect ArUco marker (5x5_50 only)
            corners, ids, binary = detect_aruco_marker_5x5(preprocessed)
            
            # Calculate homography in real-time if marker detected
            if ids is not None and len(ids) > 0 and corners is not None:
                marker_corners = corners[0][0]
                H, world_points, image_points = calculate_homography_from_marker(
                    marker_corners, MARKER_SIZE_MM
                )
                
                # Calculate reprojection error
                ones = np.ones((len(world_points), 1))
                world_homogeneous = np.hstack([world_points, ones])
                projected = (H @ world_homogeneous.T).T
                projected = projected[:, :2] / projected[:, 2:]
                
                errors = np.linalg.norm(image_points - projected, axis=1)
                mean_error = np.mean(errors)
                
                current_H = H
                current_error = mean_error
            
            # Display frame selection
            if show_preprocessing:
                # Show preprocessing steps
                h, w = img.shape[:2]
                
                # Resize all to same size for display
                display_h = h // 2
                display_w = w // 2
                
                img_small = cv2.resize(img, (display_w, display_h))
                preprocessed_small = cv2.resize(preprocessed, (display_w, display_h))
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mask_small = cv2.resize(mask_rgb, (display_w, display_h))
                binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                binary_small = cv2.resize(binary_rgb, (display_w, display_h))
                
                # Arrange in 2x2 grid
                top_row = np.hstack([img_small, preprocessed_small])
                bottom_row = np.hstack([mask_small, binary_small])
                display_frame = np.vstack([top_row, bottom_row])
                
                # Add labels
                cv2.putText(display_frame, "Original", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, "Preprocessed", (display_w + 10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, "Mask", (10, display_h + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, "Binary", (display_w + 10, display_h + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show detection result in debug mode
                if ids is not None and len(ids) > 0:
                    cv2.putText(display_frame, f"DETECTED ID: {ids[0][0]}", (10, display_h + 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "NOT DETECTED", (10, display_h + 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Draw industrial overlay with homography matrix
                if current_H is not None and ids is not None:
                    display_frame = draw_industrial_overlay(img, corners, ids, 
                                                           current_H, fps, current_error)
                else:
                    display_frame = img.copy()
                    cv2.putText(display_frame, f"Looking for Marker ID {MARKER_ID}...", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_frame, "Press 's' to see debug view", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord('s'):
                show_preprocessing = not show_preprocessing
                print(f"Preprocessing view: {'ON' if show_preprocessing else 'OFF'}")
            
            elif key == ord('c'):
                if ids is None or len(ids) == 0:
                    print("\nError: No marker detected. Cannot calculate homography.")
                    print("Tips:")
                    print("  - Press 's' to see preprocessing/debug view")
                    print("  - Ensure marker ID 23 (5x5_50) is visible")
                    print("  - Check lighting and marker is flat")
                    continue
                
                marker_corners = corners[0][0]
                marker_id = ids[0][0]
                
                print(f"\n{'='*60}")
                print(f"Detected Marker ID: {marker_id}")
                print(f"Dictionary: 5X5_50")
                print(f"Marker size: {MARKER_SIZE_MM} mm")
                print(f"{'='*60}")
                
                # Calculate homography
                H, world_points, image_points = calculate_homography_from_marker(
                    marker_corners, MARKER_SIZE_MM
                )
                
                if H is not None:
                    print("\nHomography Matrix:")
                    print(H)
                    
                    # Calculate reprojection error
                    ones = np.ones((len(world_points), 1))
                    world_homogeneous = np.hstack([world_points, ones])
                    projected = (H @ world_homogeneous.T).T
                    projected = projected[:, :2] / projected[:, 2:]
                    
                    errors = np.linalg.norm(image_points - projected, axis=1)
                    mean_error = np.mean(errors)
                    max_error = np.max(errors)
                    
                    # Measure dimensions
                    width_mm, height_mm = measure_marker_dimensions(marker_corners, H)
                    
                    print(f"\nMeasurements:")
                    print(f"  Width:  {width_mm:.3f} mm")
                    print(f"  Height: {height_mm:.3f} mm")
                    
                    print(f"\nReprojection Error:")
                    print(f"  Mean: {mean_error:.3f} pixels")
                    print(f"  Max:  {max_error:.3f} pixels")
                    
                    print(f"\nCorner correspondences:")
                    for i in range(4):
                        print(f"  Corner {i}: World {world_points[i]} mm -> Image {image_points[i]} pixels")
                    
                    # Save to JSON
                    homography_data = {
                        'homography_matrix': H.tolist(),
                        'marker_size_mm': MARKER_SIZE_MM,
                        'marker_id': int(marker_id),
                        'dictionary': '5X5_50',
                        'world_points': world_points.tolist(),
                        'image_points': image_points.tolist(),
                        'measured_width_mm': float(width_mm),
                        'measured_height_mm': float(height_mm),
                        'reprojection_error_mean': float(mean_error),
                        'reprojection_error_max': float(max_error)
                    }
                    
                    output_file = 'homography_calibration.json'
                    with open(output_file, 'w') as f:
                        json.dump(homography_data, f, indent=4)
                    
                    print(f"\n✓ Homography matrix saved to '{output_file}'")
                    print("\nCalibration complete! Press 'q' to quit or 'c' to recalibrate")
                else:
                    print("\nError: Failed to calculate homography matrix")
    
    finally:
        camera.StopGrabbing()
        camera.Close()
        cv2.destroyAllWindows()
        print("\nCamera closed")

if __name__ == "__main__":
    main()