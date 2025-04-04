import cv2
import numpy as np
import BoxesContour

class LinesDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.filtered_boxes = BoxesContour.filter_boxes(image_path)

    def apply_mask(self):
        """
        Applies a mask to remove detected text areas.
        """
        mask = np.ones_like(self.gray) * 255
        for box in self.filtered_boxes:
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = 0  # Remove text areas

        return cv2.bitwise_and(self.gray, self.gray, mask=mask), cv2.Canny(mask, 100, 200)

    @staticmethod
    def is_point_near_edge(x, y, mask_edges, neighborhood_size=5):
        """
        Checks if a point (x, y) is near an edge in the masked image.
        """
        x_min = max(0, x - neighborhood_size // 2)
        x_max = min(mask_edges.shape[1] - 1, x + neighborhood_size // 2)
        y_min = max(0, y - neighborhood_size // 2)
        y_max = min(mask_edges.shape[0] - 1, y + neighborhood_size // 2)

        return np.any(mask_edges[y_min:y_max + 1, x_min:x_max + 1] == 255)

    @staticmethod
    def are_lines_similar(line1, line2, length_threshold=10, midpoint_threshold=10, angle_threshold=10):
        """
        Determines whether two lines are similar based on length, midpoint, and angle.
        """
        def normalize_line(line):
            x0, y0, x1, y1 = line
            return (x0, y0, x1, y1) if x0 < x1 else (x1, y1, x0, y0)

        line1, line2 = normalize_line(line1), normalize_line(line2)

        def line_length(line):
            return np.linalg.norm([line[2] - line[0], line[3] - line[1]])

        def line_midpoint(line):
            return ((line[0] + line[2]) / 2, (line[1] + line[3]) / 2)

        def line_angle(line):
            dx, dy = line[2] - line[0], line[3] - line[1]
            return np.degrees(np.arctan2(dy, dx))

        length_diff = abs(line_length(line1) - line_length(line2))
        midpoint_dist = np.linalg.norm(np.array(line_midpoint(line1)) - np.array(line_midpoint(line2)))
        angle_diff = abs(line_angle(line1) - line_angle(line2))
        angle_diff = min(angle_diff, 360 - angle_diff)  # Handle circular angle differences

        return (length_diff < length_threshold and midpoint_dist < midpoint_threshold and angle_diff < angle_threshold)

    def find_lines(self):
        """
        Detects unique line segments in the image, filtering out redundant lines.
        """
        masked_gray, mask_edges = self.apply_mask()

        # Detect lines using Line Segment Detector
        lsd = cv2.createLineSegmentDetector(0)
        dlines = lsd.detect(masked_gray)

        unique_lines = []

        if dlines is not None:
            for dline in dlines[0]:
                x0, y0, x1, y1 = map(int, dline[0])

                if self.is_point_near_edge(x0, y0, mask_edges) and self.is_point_near_edge(x1, y1, mask_edges):
                    continue  # Ignore lines near text regions

                if np.linalg.norm([x1 - x0, y1 - y0]) > 10:  # Minimum line length
                    current_line = (x0, y0, x1, y1)

                    if not any(self.are_lines_similar(current_line, line) for line in unique_lines):
                        unique_lines.append(current_line)

        return unique_lines