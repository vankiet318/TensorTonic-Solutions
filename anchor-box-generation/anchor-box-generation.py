import numpy as np

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    stride = image_size / feature_size

    anchor_boxes = []
    for i in range(feature_size):
        for j in range(feature_size):
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride
            for scale in scales:
                for aspect_ratio in aspect_ratios:
                    w = scale * (aspect_ratio ** 0.5)
                    h = scale / (aspect_ratio ** 0.5)
    
                    anchor_boxes.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    return anchor_boxes