import cv2
import numpy as np
from paddleocr import PaddleOCR

class TextBoxesDetector:
    """
    A class to process text boxes detected in an image using PaddleOCR.
    It provides methods to extract, merge, and filter text boxes.
    """

    def __init__(self, image_path, distance_threshold=50):
        """
        Initializes the text box processor.
        
        :param image_path: Path to the image to process.
        :param distance_threshold: Maximum distance between two text boxes to be merged.
        """
        self.image_path = image_path
        self.distance_threshold = distance_threshold

        # Load image
        self.image = cv2.imread(image_path)
        
        # Initialize PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")
        
        # Extract initial text boxes
        self.boxes, self.texts, self.scores = self._extract_text_boxes()

    def _extract_text_boxes(self):
        """
        Extract text boxes, recognized texts, and confidence scores using PaddleOCR.

        :return: Tuple of (boxes, texts, scores).
        """
        result = self.ocr.ocr(self.image, cls=True)
        
        # Extract bounding boxes, recognized text, and confidence scores
        boxes = [line[0] for line in result[0]]  # Detected text regions
        texts = [line[1][0] for line in result[0]]  # Recognized texts
        scores = [line[1][1] for line in result[0]]  # Confidence scores

        return boxes, texts, scores

    def _boxes_distance(self, box1, box2):
        """
        Compute the Euclidean distance between the centers of two text boxes.

        :param box1: Coordinates of the first text box.
        :param box2: Coordinates of the second text box.
        :return: Euclidean distance between the two box centers.
        """
        center1 = np.mean(box1, axis=0)  # Center of first box
        center2 = np.mean(box2, axis=0)  # Center of second box

        return np.linalg.norm(center1 - center2)  # Euclidean distance

    def combine_boxes_and_texts(self):
        """
        Merge close text boxes and concatenate their texts if within the distance threshold.

        :return: Lists of merged text boxes and concatenated texts.
        """
        combined_boxes = []
        combined_texts = []
        used = [False] * len(self.boxes)

        for i in range(len(self.boxes)):
            if used[i]:
                continue  # Skip already merged boxes

            # Initialize current box and text
            current_box = np.array(self.boxes[i], dtype=np.float32)
            current_text = self.texts[i]
            used[i] = True  # Mark as merged

            for j in range(i + 1, len(self.boxes)):
                if used[j]:
                    continue

                # Compute the distance between boxes
                distance = self._boxes_distance(current_box, self.boxes[j])

                # Merge if within threshold
                if distance < self.distance_threshold:
                    # Combine box coordinates
                    combined_points = np.vstack((current_box, self.boxes[j]))
                    combined_points = np.array(combined_points, dtype=np.float32)

                    # Ensure valid bounding box
                    if len(combined_points) >= 2:
                        try:
                            rect = cv2.minAreaRect(combined_points)
                            current_box = cv2.boxPoints(rect)

                            # Merge text
                            current_text += " " + self.texts[j]
                            used[j] = True
                        except Exception as e:
                            print(f"Error in cv2.minAreaRect: {e}")
                            continue

            combined_boxes.append(current_box)
            combined_texts.append(current_text)

        return combined_boxes, combined_texts
    
    def combined_dict(self):
        """
        Merge nearby text boxes and concatenate their corresponding texts if the distance between them is smaller than a specified threshold.

        :return: 
            - combined_dict: A dictionary {tuple(box_coordinates): merged_text}
                             where each key is a 4-point bounding box ((x1, y1), ..., (x4, y4)),
                             and the value is the merged text string.
            - text_boxes_loc: A list of simplified axis-aligned bounding boxes [(x1, y1, x4, y4)] as [(x1, y1, x2, y2)]
                              representing the merged regions.
        """
        combined_dict = {}  # Store box coordinates and text mapping
        used = [False] * len(self.boxes)  # Track which boxes have already been merged

        for i in range(len(self.boxes)):
            if used[i]:
                continue  # Skip already merged boxes

            # Initialize current box and text
            current_box = np.array(self.boxes[i], dtype=np.int32)
            current_text = [self.texts[i]]  # Store text as a list to append later
            used[i] = True  # Mark as merged

            for j in range(i + 1, len(self.boxes)):
                if used[j]:
                    continue  # Skip already-used boxes

                # Compute the distance between boxes
                distance = self._boxes_distance(current_box, self.boxes[j])

                # Merge if within threshold
                if distance < self.distance_threshold:
                    # Combine box coordinates
                    combined_points = np.vstack((current_box, self.boxes[j]))
                    combined_points = np.array(combined_points, dtype=np.int64)

                    # Ensure valid bounding box
                    if len(combined_points) >= 2:
                        try:
                            rect = cv2.minAreaRect(combined_points)
                            current_box = cv2.boxPoints(rect)  # Get the updated box

                            # Merge text
                            current_text.append(self.texts[j])  # Append the text
                            used[j] = True
                        except Exception as e:
                            print(f"Error in cv2.minAreaRect: {e}")
                            continue

            # Convert NumPy array to a tuple (so it can be used as a dictionary key)
            box_tuple = tuple(map(tuple, current_box))  # Convert to ((x1, y1), (x2, y2), ...)
            combined_dict[box_tuple] = " ".join(current_text)  # Store as key-value pair

        # Initialize list to store simplified bounding boxes
        text_boxes_loc = []

        # Convert each 4-point rotated box into an axis-aligned bounding box
        for box_coords in combined_dict.keys():
            coords = list(box_coords)  # Convert tuple to list of points

        # Separate x and y coordinates
        x_coords = [p[0] for p in coords]
        y_coords = [p[1] for p in coords]

        # Calculate axis-aligned bounding box: (min_x, min_y, max_x, max_y)
        bbox = (
            min(x_coords),  # Left boundary
            min(y_coords),  # Top boundary
            max(x_coords),  # Right boundary
            max(y_coords)   # Bottom boundary
        )

        # Add to the result list
        text_boxes_loc.append(bbox)

        return combined_dict, text_boxes_loc
