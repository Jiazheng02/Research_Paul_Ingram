from pptx import Presentation
from typing import List, Tuple, Dict, Optional
from NodesExtraction import extract_node_boxes
from LinesExtraction import extract_lines

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def create_node_mappings(node_boxes: List[Tuple]) -> Tuple[Dict[str, Tuple], Dict[str, int]]:
    """
    Create mappings for node boxes to their coordinates and order.
    
    Args:
        node_boxes: List of node box tuples (_, top, left, right, bottom, text)
        
    Returns:
        Tuple containing:
        - node_map: Dictionary mapping text to bounding box coordinates
        - node_order: Dictionary mapping text to index order
    """
    node_map = {}
    node_order = {}
    
    for idx, (_, top, left, right, bottom, text) in enumerate(node_boxes):
        node_map[text] = (top, left, right, bottom)
        node_order[text] = idx
        
    return node_map, node_order

def find_closest_nodes(
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
    node_map: Dict[str, Tuple]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find the closest nodes to the start and end points of a line.
    
    Args:
        line_start: (x, y) coordinates of line start point
        line_end: (x, y) coordinates of line end point
        node_map: Dictionary mapping text to bounding box coordinates
        
    Returns:
        Tuple of (from_text, to_text) representing closest nodes
    """
    from_text, to_text = None, None
    min_distance_from = float("inf")
    min_distance_to = float("inf")
    
    for text, (top, left, right, bottom) in node_map.items():
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        
        dist_from = calculate_distance((center_x, center_y), line_start)
        dist_to = calculate_distance((center_x, center_y), line_end)
        
        if dist_from < min_distance_from:
            from_text = text
            min_distance_from = dist_from
            
        if dist_to < min_distance_to:
            to_text = text
            min_distance_to = dist_to
            
    return from_text, to_text

def create_and_sort_edges(
    lines: Dict,
    node_map: Dict[str, Tuple],
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
    """
    edges = []
    
    for line_id, (x1, y1, x2, y2) in lines.items():
        from_text, to_text = find_closest_nodes((x1, y1), (x2, y2), node_map)
        
        if from_text and to_text and from_text != to_text:
            from_top, _, _, from_bottom = node_map[from_text]
            to_top, _, _, to_bottom = node_map[to_text]

            # Swap if `to_text` is positioned higher than `from_text`
            if to_top > from_top:
                from_text, to_text = to_text, from_text

            # Append the valid edge (connection)
            edges.append((from_text, to_text))
    
    # Sort edges based on the order of `node_boxes` for logical output
    sorted_edges = sorted(edges, key=lambda edge: (node_order.get(edge[0], float("inf")), 
                                                   node_order.get(edge[1], float("inf"))))

    return sorted_edges, respondent, node_count, line_count