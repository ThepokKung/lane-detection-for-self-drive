import cv2
import numpy as np
import os

def process_image(image, prevLx, prevRx, output_folder, image_name):
    """
    Process a single image for lane detection.
    Save intermediate steps to visualize the process.
    """
    frame = cv2.resize(image, (640, 480))

    ## Choosing points for perspective transformation
    tl = (280, 265)  # Top-left
    bl = (20, 480)
    tr = (360, 265)
    br = (620, 480)

    # Draw the ROI points and connecting lines on the original frame
    roi_color = (0, 0, 255)  # Red color for ROI points and lines
    cv2.circle(frame, tl, 5, (0, 0, 255), -1)  # Top-left
    cv2.circle(frame, bl, 5, (255, 0, 0), -1)  # Bottom-left
    cv2.circle(frame, tr, 5, (0, 255, 0), -1)  # Top-right
    cv2.circle(frame, br, 5, (255, 0, 255), -1)  # Bottom-right

    # Connect the points with lines to visualize the ROI
    cv2.line(frame, tl, bl, roi_color, 2)
    cv2.line(frame, tr, br, roi_color, 2)
    cv2.line(frame, tl, tr, roi_color, 2)
    cv2.line(frame, bl, br, roi_color, 2)

    # Save the ROI visualization
    roi_path = os.path.join(output_folder, "roi")
    os.makedirs(roi_path, exist_ok=True)
    cv2.imwrite(os.path.join(roi_path, image_name), frame)

    ## Apply perspective transformation
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))

    # Save the bird's eye view transformation
    bird_eye_path = os.path.join(output_folder, "bird_eye")
    os.makedirs(bird_eye_path, exist_ok=True)
    cv2.imwrite(os.path.join(bird_eye_path, image_name), transformed_frame)

    ### Object Detection
    # Image Thresholding
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)

    # Define thresholds based on the provided settings
    lower = np.array([12, 60, 38])  # Updated thresholds
    upper = np.array([124, 255, 255])
    mask = cv2.inRange(hsv_transformed_frame, lower, upper)

    # Save the masked image
    mask_path = os.path.join(output_folder, "mask")
    os.makedirs(mask_path, exist_ok=True)
    cv2.imwrite(os.path.join(mask_path, image_name), mask)

    # Histogram
    histogram = np.sum(mask[mask.shape[0] // 2:, :], axis=0)
    midpoint = int(histogram.shape[0] / 2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # Sliding Window
    y = 472
    lx = []
    rx = []

    msk = mask.copy()

    while y > 0:
        ## Left threshold
        if left_base - 50 >= 0 and left_base + 50 < mask.shape[1]:
            img = mask[y - 40:y, left_base - 50:left_base + 50]
            contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    lx.append(left_base - 50 + cx)
                    left_base = left_base - 50 + cx

        ## Right threshold
        if right_base - 50 >= 0 and right_base + 50 < mask.shape[1]:
            img = mask[y - 40:y, right_base - 50:right_base + 50]
            contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    rx.append(right_base - 50 + cx)
                    right_base = right_base - 50 + cx

        cv2.rectangle(msk, (left_base - 50, y), (left_base + 50, y - 40), (255, 255, 255), 2)
        cv2.rectangle(msk, (right_base - 50, y), (right_base + 50, y - 40), (255, 255, 255), 2)
        y -= 40

    # Save the sliding window image
    sliding_window_path = os.path.join(output_folder, "sliding_window")
    os.makedirs(sliding_window_path, exist_ok=True)
    cv2.imwrite(os.path.join(sliding_window_path, image_name), msk)

    # Ensure lx and rx are not empty
    if len(lx) == 0:
        lx = prevLx
    else:
        prevLx = lx
    if len(rx) == 0:
        rx = prevRx
    else:
        prevRx = rx

    # Ensure both lx and rx have valid lengths
    if len(lx) == 0 or len(rx) == 0:
        print("No valid lane points detected. Skipping this frame.")
        return frame, prevLx, prevRx

    # Ensure both lx and rx have the same length
    min_length = min(len(lx), len(rx))

    # Create the top and bottom points for the quadrilateral
    top_left = (lx[0], 472)
    bottom_left = (lx[min_length - 1], 0)
    top_right = (rx[0], 472)
    bottom_right = (rx[min_length - 1], 0)

    # Define the quadrilateral points
    quad_points = np.array([top_left, bottom_left, bottom_right, top_right], dtype=np.int32)
    quad_points = quad_points.reshape((-1, 1, 2))

    # Create a copy of the transformed frame
    overlay = transformed_frame.copy()

    # Draw the filled polygon on the transformed frame
    cv2.fillPoly(overlay, [quad_points], (0, 255, 0))

    alpha = 0.2  # Opacity factor
    cv2.addWeighted(overlay, alpha, transformed_frame, 1 - alpha, 0, transformed_frame)

    # Inverse perspective transformation to map the lanes back to the original image
    inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
    original_perpective_lane_image = cv2.warpPerspective(transformed_frame, inv_matrix, (640, 480))

    # Combine the original frame with the lane image
    result = cv2.addWeighted(frame, 1, original_perpective_lane_image, 0.5, 0)

    return result, prevLx, prevRx

def process_folder(input_folder, output_folder):
    """
    Process all images in a folder for lane detection and save all steps.
    """
    prevLx, prevRx = [], []  # Previous lane data for continuity

    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Could not load image {image_name}")
            continue

        # Process the image
        result, prevLx, prevRx = process_image(image, prevLx, prevRx, output_folder, image_name)

        # Save the final result
        final_path = os.path.join(output_folder, "final")
        os.makedirs(final_path, exist_ok=True)
        cv2.imwrite(os.path.join(final_path, image_name), result)

def main():
    input_folder = "images"  # Folder containing input images
    output_folder = "lane_detection_steps_all"  # Folder to save all steps

    process_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()
