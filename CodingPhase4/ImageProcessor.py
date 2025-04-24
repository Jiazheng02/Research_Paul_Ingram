import cv2
import numpy as np
from paddleocr import PaddleOCR
from io import BytesIO
from PIL import Image

class ImageProcessor:
    """
    A class to extract text boxes and relationships from images inside a PPT slide.
    """

    def __init__(self, image):
        """
        Initializes the text box processor.
        
        :param image: Input image to process (as a NumPy array or PIL Image).
        """
        # Convert input image to NumPy array if it's a PIL Image
        if isinstance(image, Image.Image):
            self.image = np.array(image)
        else:
            self.image = image

        # Convert the image to grayscale for processing
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        
        # Initialize PaddleOCR for text detection and recognition
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")

        # Extract initial text boxes, recognized texts, and confidence scores
        self.boxes, self.texts, self.scores = self._extract_text_boxes()

        # Extract combined text boxes and their centers
        self.text_boxes_dict, self.text_boxes_loc = self.get_combined_text_dict()
        self.box_centers = self._calculate_box_centers()
        
        # Find and filter potential text boxes in the image
        self.get_boxes = self.find_boxes()
        self.filtered_boxes = self.filter_boxes(self.get_boxes, self.text_boxes_loc)

        # Detect lines in the image
        self.lines = self.find_lines()

        # Match relationships between text boxes based on detected lines
        self.relationships = self.match_relationships()
    
    def _extract_text_boxes(self):
        """
        Extract text boxes, recognized texts, and confidence scores using PaddleOCR.

        :return: Tuple of (boxes, texts, scores).
        """
        # Run PaddleOCR on the image
        result = self.ocr.ocr(self.image, cls=True)

        if result is None or len(result) == 0 or result[0] is None:
            print("Warning: PaddleOCR detected no text in the image.")
            return [], [], []  # Return empty lists to avoid errors
        
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
        # Calculate the center of each box
        center1 = np.mean(box1, axis=0)  # Center of first box
        center2 = np.mean(box2, axis=0)  # Center of second box

        # Compute the Euclidean distance between the centers
        return np.linalg.norm(center1 - center2)

    def get_combined_text_dict(self):
        """
        Merge nearby text boxes and concatenate their corresponding texts if the distance between them is smaller than a specified threshold.

        :return: 
            - combined_dict: A dictionary {tuple(box_coordinates): merged_text}
                             where each key is a 4-point bounding box ((x1, y1), ..., (x4, y4)),
                             and the value is the merged text string.
            - text_boxes_loc: A list of simplified axis-aligned bounding boxes [(x1, y1, x4, y4)] as [(x1, y1, x2, y2)]
                              representing the merged regions.
        """
        distance_threshold = 40  # Maximum distance between two text boxes to be merged
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
                if distance < distance_threshold:
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
    
    def find_boxes(self):
        """
        Detects bounding boxes of potential text areas in the image.

        :return: A list of bounding boxes in the format [(x1, y1, x2, y2), ...]
        """
        # Convert to binary threshold (inverse)
        _, binary = cv2.threshold(self.gray, 128, 255, cv2.THRESH_BINARY_INV)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Apply morphological operations to clean noise
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # Remove noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Fill gaps

        # Detect bounding boxes
        get_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 1000:  # Filter out small regions
                get_boxes.append((x, y, x + w, y + h))

        return get_boxes

    def iou(self, box1, box2):
        """
        Computes the Intersection over Union (IoU) between two bounding boxes.

        :param box1: First bounding box (x1, y1, x2, y2)
        :param box2: Second bounding box (x1, y1, x2, y2)
        :return: IoU score (value between 0 and 1)
        """
        # Compute intersection coordinates
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2

        inter_x1 = max(x1, x1_)
        inter_y1 = max(y1, y1_)
        inter_x2 = min(x2, x2_)
        inter_y2 = min(y2, y2_)

        # Compute intersection area
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_ - x1_) * (y2_ - y1_)

        # Compute IoU
        return inter_area / float(box1_area + box2_area - inter_area)

    def filter_boxes(self, get_boxes, text_boxes_loc):
        """
        Filters out overlapping bounding boxes. This is used to retain the smallest bounding regions.
        If no drawn (plotted) boxes are found, fallback to text box regions.

        :param get_boxes: List of candidate bounding boxes (e.g., from line/edge detection).
        :param text_boxes_loc: Backup list of text-detected bounding boxes (AABB format).
        :return: Filtered list of bounding boxes, excluding large overlaps or (0,0,...) dummies.
        """
        filtered_boxes = []

        usable_boxes = get_boxes if len(get_boxes) > 0 else text_boxes_loc  # Use text boxes if no plotted boxes are found

        for i, box1 in enumerate(usable_boxes):
            keep = True
            for j, box2 in enumerate(usable_boxes):
                if i != j and self.iou(box1, box2) > 0.4:  # Overlapping threshold
                    if (box1[2] - box1[0]) * (box1[3] - box1[1]) > (box2[2] - box2[0]) * (box2[3] - box2[1]):
                        keep = False  # Remove the larger box if overlap is high
            if keep:
                filtered_boxes.append(box1)

        return [box for box in filtered_boxes if not (box[0] == 0 and box[1] == 0)]

    def apply_mask(self):
        """
        Applies a mask to remove detected text areas from the image.
        
        :return: Masked grayscale image and its edge-detected version.
        """
        # Create a mask to hide text regions
        mask = np.ones_like(self.gray) * 255
        for box in self.filtered_boxes:
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = 0  # Mask out text areas

        # Apply the mask to the grayscale image
        masked_gray = cv2.bitwise_and(self.gray, self.gray, mask=mask)
        # Detect edges in the masked image
        mask_edges = cv2.Canny(mask, 100, 200)

        return masked_gray, mask_edges

    @staticmethod
    def is_point_near_edge(x, y, mask_edges, neighborhood_size=5):
        """
        Checks if a point (x, y) is near an edge in the masked image.

        :param x: X-coordinate of the point.
        :param y: Y-coordinate of the point.
        :param mask_edges: Edge-detected version of the masked image.
        :param neighborhood_size: Search window size.
        :return: Boolean indicating whether the point is near an edge.
        """
        # Define the search window around the point
        x_min = max(0, x - neighborhood_size // 2)
        x_max = min(mask_edges.shape[1] - 1, x + neighborhood_size // 2)
        y_min = max(0, y - neighborhood_size // 2)
        y_max = min(mask_edges.shape[0] - 1, y + neighborhood_size // 2)

        # Check if any edge pixel exists in the neighborhood
        return np.any(mask_edges[y_min:y_max + 1, x_min:x_max + 1] == 255)

    @staticmethod
    def are_lines_similar(line1, line2, length_threshold=10, midpoint_threshold=10, angle_threshold=10):
        """
        Determines whether two lines are similar based on length, midpoint, and angle.

        :param line1: First line segment (x1, y1, x2, y2)
        :param line2: Second line segment (x1, y1, x2, y2)
        :return: Boolean indicating similarity.
        """
        def normalize_line(line):
            x0, y0, x1, y1 = line
            return (x0, y0, x1, y1) if x0 < x1 else (x1, y1, x0, y0)

        # Normalize lines to ensure consistent start and end points
        line1, line2 = normalize_line(line1), normalize_line(line2)

        def line_length(line):
            return np.linalg.norm([line[2] - line[0], line[3] - line[1]])

        def line_midpoint(line):
            return ((line[0] + line[2]) / 2, (line[1] + line[3]) / 2)

        def line_angle(line):
            dx, dy = line[2] - line[0], line[3] - line[1]
            return np.degrees(np.arctan2(dy, dx))

        # Compute differences in length, midpoint, and angle
        length_diff = abs(line_length(line1) - line_length(line2))
        midpoint_dist = np.linalg.norm(np.array(line_midpoint(line1)) - np.array(line_midpoint(line2)))
        angle_diff = abs(line_angle(line1) - line_angle(line2))
        angle_diff = min(angle_diff, 360 - angle_diff)  # Handle circular angle differences

        # Check if lines are similar based on thresholds
        return (length_diff < length_threshold and midpoint_dist < midpoint_threshold and angle_diff < angle_threshold)
    
    def find_lines(self):
        """
        Detects unique line segments in the image, filtering out redundant lines.

        :return: List of detected lines in the format [(x1, y1, x2, y2), ...]
        """
        # Apply mask to remove text regions and detect edges
        masked_gray, mask_edges = self.apply_mask()

        # Detect lines using Line Segment Detector
        lsd = cv2.createLineSegmentDetector(0)
        dlines = lsd.detect(masked_gray)

        unique_lines = []

        # Ensure that dlines is not None before processing
        if dlines is None or dlines[0] is None:
            print("Warning: No lines detected in the image.")
            return []  # Return an empty list instead of None to prevent errors

        unique_lines = []

        for dline in dlines[0]:
            x0, y0, x1, y1 = map(int, dline[0])

            # Ignore lines near text regions
            if self.is_point_near_edge(x0, y0, mask_edges) and self.is_point_near_edge(x1, y1, mask_edges):
                continue  # Ignore lines near text regions

            # Filter out short lines
            if np.linalg.norm([x1 - x0, y1 - y0]) > 10:  # Minimum line length
                current_line = (x0, y0, x1, y1)

                # Check if the line is similar to any existing line
                if not any(self.are_lines_similar(current_line, line) for line in unique_lines):
                    unique_lines.append(current_line)

        return unique_lines

    def _calculate_box_centers(self):
        """
        Computes the center points for all text boxes.

        :return: Dictionary {(cx, cy): text}
                 - Key: Center coordinates (cx, cy)
                 - Value: Corresponding text inside the bounding box
        """
        centers_dict = {}
        for box, text in self.text_boxes_dict.items():
            # Compute the center of the bounding box
            cx = np.mean([p[0] for p in box])  # Average x-coordinates
            cy = np.mean([p[1] for p in box])  # Average y-coordinates
            centers_dict[(cx, cy)] = text  # Store as dictionary {center: text}
        return centers_dict

    def _find_nearest_textbox(self, x, y):
        """
        Finds the nearest text box center for a given point (x, y).

        :param x: X-coordinate of the point
        :param y: Y-coordinate of the point
        :return: The text of the nearest box and its center (cx, cy),
                 or None if no match is found
        """
        min_distance = float("inf")
        nearest_text = None
        nearest_center = None  # Store the nearest box center

        for (cx, cy), text in self.box_centers.items():
            distance = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)
            if distance < min_distance:
                min_distance = distance
                nearest_text = text
                nearest_center = (cx, cy)  # Save the nearest center

        return nearest_text, nearest_center

    def match_relationships(self):
        """
        Matches text boxes based on line segment connections (from-bottom-to-top)
        and removes duplicates while ensuring correct ordering.

        :return: List of unique (from_text, to_text) relationships sorted bottom-to-top, left-to-right.
        """
        relationships = set()  # Use a set to remove duplicates

        for line in self.lines:
            x0, y0, x1, y1 = line  # Extract line segment endpoints

            # Find the nearest text box for both endpoints
            text_from, center_from = self._find_nearest_textbox(x0, y0)
            text_to, center_to = self._find_nearest_textbox(x1, y1)

            # Ensure valid matches and maintain "bottom-to-top" relationship
            if text_from and text_to and (text_from != text_to):
                if y0 > y1:  # Ensure "from" is at the bottom
                    relationships.add((text_from, text_to, center_from, center_to))
                else:  # Swap if necessary
                    relationships.add((text_to, text_from, center_to, center_from))

        # **Sorting rules**
        # 1. Sort primarily by `from_center[1]` in descending order (bottom-to-top)
        # 2. If `y` coordinates are the same, sort by `from_center[0]` in ascending order (left-to-right)
        sorted_relationships = sorted(
            relationships, key=lambda item: (-item[2][1], item[2][0])
        )

        # Convert to list of edges
        edges = [(from_text, to_text) for from_text, to_text, _, _ in sorted_relationships]

        # Return the final relationship list
        return edges