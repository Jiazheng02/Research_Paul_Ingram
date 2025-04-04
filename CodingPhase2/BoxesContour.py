import cv2
import numpy as np

def find_boxes(image_path):
    """
    Detects bounding boxes of potential text areas in an image.

    :param image_path: Path to the input image.
    :return: A list of bounding boxes in the format [(x1, y1, x2, y2), ...]
    """
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding (inverse)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Apply morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # Remove noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Fill gaps

    # Detect and store bounding boxes
    text_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 1000:  # Consider only large enough regions
            text_boxes.append((x, y, x + w, y + h))  # Store bounding box

    return text_boxes

def iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.

    :param box1: First bounding box (x1, y1, x2, y2)
    :param box2: Second bounding box (x1, y1, x2, y2)
    :return: IoU score (value between 0 and 1)
    """
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    # Compute the intersection area
    inter_x1 = max(x1, x1_)
    inter_y1 = max(y1, y1_)
    inter_x2 = min(x2, x2_)
    inter_y2 = min(y2, y2_)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    # Compute the IoU score
    iou_score = inter_area / float(box1_area + box2_area - inter_area)
    return iou_score

def filter_boxes(image_path):
    """
    Filters out overlapping bounding boxes, keeping only the smallest ones.

    :param image_path: Path to the input image.
    :return: A list of filtered bounding boxes [(x1, y1, x2, y2), ...]
    """
    # Detect text bounding boxes
    text_boxes = find_boxes(image_path)
    filtered_boxes = []

    # Compare each bounding box with all others
    for i, box1 in enumerate(text_boxes):
        keep = True
        for j, box2 in enumerate(text_boxes):
            if i != j and iou(box1, box2) > 0.4:  # If two boxes highly overlap
                if (box1[2] - box1[0]) * (box1[3] - box1[1]) > (box2[2] - box2[0]) * (box2[3] - box2[1]):
                    keep = False  # Remove the larger box if overlap is high

        if keep:
            filtered_boxes.append(box1)

    # Remove unwanted boxes positioned at (0,0)
    filtered_boxes = [box for box in filtered_boxes if not (box[0] == 0 and box[1] == 0)]

    return filtered_boxes