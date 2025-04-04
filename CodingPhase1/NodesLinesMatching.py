from pptx import Presentation
from typing import List, Tuple, Dict, Optional, NamedTuple
from dataclasses import dataclass
from NodesExtraction import extract_node_boxes
from LinesExtraction import extract_lines

@dataclass
class Point:
    """Represents a 2D point with x and y coordinates."""
    x: float
    y: float

@dataclass
class BoundingBox:
    """Represents a bounding box with top, left, right, and bottom coordinates."""
    top: float
    left: float
    right: float
    bottom: float

    @property
    def center(self) -> Point:
        """Calculate the center point of the bounding box."""
        return Point(
            x=(self.left + self.right) / 2,
            y=(self.top + self.bottom) / 2
        )

def calculate_distance(point1: Point, point2: Point) -> float:
    """Calculate Euclidean distance between two points."""
    return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

def create_node_mappings(node_boxes: List[Tuple]) -> Tuple[Dict[str, BoundingBox], Dict[str, int]]:
    """
    Create mappings for node boxes to their coordinates and order.
    
    Args:
        node_boxes: List of node box tuples (_, top, left, right, bottom, text)
        
    Returns:
        Tuple containing:
        - node_map: Dictionary mapping text to bounding box coordinates
        - node_order: Dictionary mapping text to index order
        
    Raises:
        ValueError: If node_boxes is empty or contains invalid data
    """
    if not node_boxes:
        raise ValueError("node_boxes cannot be empty")
        
    node_map: Dict[str, BoundingBox] = {}
    node_order: Dict[str, int] = {}
    
    for idx, (_, top, left, right, bottom, text) in enumerate(node_boxes):
        if not isinstance(text, str):
            raise ValueError(f"Invalid text type in node box: {text}")
            
        node_map[text] = BoundingBox(
            top=float(top),
            left=float(left),
            right=float(right),
            bottom=float(bottom)
        )
        node_order[text] = idx
        
    return node_map, node_order

def find_closest_nodes(
    line_start: Point,
    line_end: Point,
    node_map: Dict[str, BoundingBox]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find the closest nodes to the start and end points of a line.
    
    Args:
        line_start: Start point of the line
        line_end: End point of the line
        node_map: Dictionary mapping text to bounding box coordinates
        
    Returns:
        Tuple of (from_text, to_text) representing closest nodes
        
    Raises:
        ValueError: If node_map is empty
    """
    if not node_map:
        raise ValueError("node_map cannot be empty")
        
    from_text, to_text = None, None
    min_distance_from = float("inf")
    min_distance_to = float("inf")
    
    for text, box in node_map.items():
        center = box.center
        
        dist_from = calculate_distance(center, line_start)
        dist_to = calculate_distance(center, line_end)
        
        if dist_from < min_distance_from:
            from_text = text
            min_distance_from = dist_from
            
        if dist_to < min_distance_to:
            to_text = text
            min_distance_to = dist_to
            
    return from_text, to_text

def create_and_sort_edges(
    lines: Dict[str, Tuple[float, float, float, float]],
    node_map: Dict[str, BoundingBox],
    node_order: Dict[str, int]
) -> List[Tuple[str, str]]:
    """
    Create and sort edges based on line positions and node order.
    
    Args:
        lines: Dictionary of line coordinates
        node_map: Dictionary mapping text to bounding box coordinates
        node_order: Dictionary mapping text to index order
        
    Returns:
        List of sorted edges as (from_text, to_text) tuples
        
    Raises:
        ValueError: If any input is empty or invalid
    """
    if not lines:
        return []
        
    if not node_map or not node_order:
        raise ValueError("node_map and node_order cannot be empty")
        
    edges = []
    
    for line_id, (x1, y1, x2, y2) in lines.items():
        start_point = Point(x=float(x1), y=float(y1))
        end_point = Point(x=float(x2), y=float(y2))
        
        from_text, to_text = find_closest_nodes(start_point, end_point, node_map)
        
        if from_text and to_text and from_text != to_text:
            from_box = node_map[from_text]
            to_box = node_map[to_text]
            
            if to_box.top > from_box.top:
                from_text, to_text = to_text, from_text
                
            edges.append((from_text, to_text))
    
    return sorted(edges, key=lambda edge: (
        node_order.get(edge[0], float("inf")),
        node_order.get(edge[1], float("inf"))
    ))

def matching_lines_nodes(slide: Presentation, slide_num: int) -> Tuple[List[Tuple[str, str]], str, int, int]:
    """
    Matches each line shape's start and end points to the nearest text box (node)
    within a given slide.

    Args:
        slide: The slide object from the PowerPoint presentation.
        slide_num: The slide number.

    Returns:
        Tuple containing:
        - sorted_edges: List of tuples (from_text, to_text) representing matched connections
        - respondent: The top-leftmost text box identifier
        - node_count: Total number of nodes detected
        - line_count: Total number of lines detected
        
    Raises:
        ValueError: If slide is None or slide_num is invalid
    """
    if not slide:
        raise ValueError("slide cannot be None")
        
    if slide_num < 0:
        raise ValueError("slide_num must be non-negative")
        
    node_boxes, respondent, node_count = extract_node_boxes(slide, slide_num)
    lines, line_count = extract_lines(slide)
    
    try:
        node_map, node_order = create_node_mappings(node_boxes)
        sorted_edges = create_and_sort_edges(lines, node_map, node_order)
    except Exception as e:
        raise RuntimeError(f"Error processing slide {slide_num}: {str(e)}")
    
    return sorted_edges, respondent, node_count, line_count