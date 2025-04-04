import cv2
import numpy as np

class RelationshipMatcher:
    """
    A class to match text box centers with line segment endpoints to determine relationships.
    """

    def __init__(self, text_boxes_dict, lines):
        """
        Initializes the matcher with detected text boxes and line segments.

        :param text_boxes_dict: Dictionary {tuple(box_coordinates): text}
                                - Key: Bounding box tuple ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
                                - Value: Text contained in the bounding box
        :param lines: List of detected lines [(x0, y0, x1, y1), ...]
                      - Each tuple represents a line segment with two endpoints (x0, y0) → (x1, y1)
        """
        self.text_boxes_dict = text_boxes_dict
        self.lines = lines
        self.box_centers = self._calculate_box_centers()  # Precompute center points of text boxes

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

        # Return the final relationship list
        return [(from_text, to_text) for from_text, to_text, _, _ in sorted_relationships]