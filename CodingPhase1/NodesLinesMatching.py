from pptx import Presentation
from NodesExtraction import extract_node_boxes
from LinesExtraction import extract_lines

def matching_lines_nodes(slide, slide_num):
    """
    Matches each line shape's start and end points to the nearest text box (node)
    within a given slide.

    Args:
        slide: The slide object from the PowerPoint presentation.
        slide_num (int): The slide number.

    Returns:
        tuple:
            - sorted_edges (list): A list of tuples (from_text, to_text) representing the matched 
              connections between text boxes based on line positions, sorted based on `node_boxes` order.
            - respondent (str): The top-leftmost text box, often used as an identifier for the slide.
            - line_count (int): The total number of line shapes detected on the slide.
    """
    edges = []  # Stores matched edges (connections) between text boxes
    node_boxes, respondent, node_count = extract_node_boxes(slide, slide_num)  # Extract nodes with bounding box coordinates
    lines, line_count = extract_lines(slide)  # Extract lines from the slide and count them

    # Create mappings from node_boxes
    node_map = {}  # Maps text → (top, left, right, bottom)
    node_order = {}  # Maps text → index (ranking order)

    for idx, (_, top, left, right, bottom, text) in enumerate(node_boxes):
        node_map[text] = (top, left, right, bottom)  # Store bounding box coordinates for each text label
        node_order[text] = idx  # Assign index based on sorted order

    # Iterate through all detected lines
    for line_id, (x1, y1, x2, y2) in lines.items():
        from_text, to_text = None, None  # Initialize the closest text boxes for each line
        min_distance_from = float("inf")  # Track the minimum distance for the line's start point
        min_distance_to = float("inf")  # Track the minimum distance for the line's end point

        # Iterate over all nodes to find the closest match for both start and end points
        for text, (top, left, right, bottom) in node_map.items():
            # Compute the center point of the text box
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2

            # Calculate Euclidean distances between the line endpoints and the text box center
            dist_from = ((center_x - x1) ** 2 + (center_y - y1) ** 2) ** 0.5
            dist_to = ((center_x - x2) ** 2 + (center_y - y2) ** 2) ** 0.5

            # Update the closest matching text box for the start point
            if dist_from < min_distance_from:
                from_text = text
                min_distance_from = dist_from

            # Update the closest matching text box for the end point
            if dist_to < min_distance_to:
                to_text = text
                min_distance_to = dist_to

        # Ensure a valid connection (avoid self-connections)
        # Reorder nodes to ensure `from_text` appears lower (has a larger Y-coordinate)
        # PowerPoint uses an inverted Y-axis, meaning larger Y values are lower on the slide
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