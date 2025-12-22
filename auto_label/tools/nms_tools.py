import torch
import torchvision

def nms_across_all_classes(detections, iou_threshold=0.5):
    """
    Applies Class-Agnostic Non-Maximum Suppression (NMS) to a list of detected objects.

    This function ignores the class labels and treats all boxes as potential conflicts.
    If boxes from different classes (e.g., 'pothole', 'manhole', 'crack') overlap significantly 
    (IoU > threshold), only the one with the highest confidence score is kept.
    
    This is particularly useful for eliminating false positives where a single object 
    is detected as multiple classes (e.g., a manhole detected as both 'manhole' and 'pothole').

    Args:
        detections (list): A list of dictionaries where each dictionary represents a detected object.
            Expected format: 
            [
                {'box': [x1, y1, x2, y2], 'score': 0.9, 'label': 'pothole'},
                ...
            ]
        iou_threshold (float): The Intersection over Union (IoU) threshold. 
                               Boxes with IoU greater than this value will be suppressed. 
                               Defaults to 0.5.

    Returns:
        list: A filtered list of detections containing only the surviving objects.
    """
    
    # 1. Handle empty input case to avoid errors
    if not detections:
        return []

    # 2. Extract boxes and scores from the list of dictionaries
    #    Convert them into PyTorch tensors required for NMS operation.
    #    Assumption: 'box' key contains [x1, y1, x2, y2]
    boxes_tensor = torch.tensor([d['box'] for d in detections], dtype=torch.float32)
    scores_tensor = torch.tensor([d['score'] for d in detections], dtype=torch.float32)

    # 3. Execute NMS (Non-Maximum Suppression)
    #    By passing all boxes and scores together without class IDs, 
    #    this operation acts as 'Class-Agnostic NMS'.
    #    It keeps the highest scoring box and removes any overlapping box regardless of its label.
    keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold)

    # 4. Retrieve the surviving objects using the kept indices
    #    Convert tensor indices back to a Python list and filter the original detections.
    final_results = [detections[i] for i in keep_indices.tolist()]
    
    return final_results