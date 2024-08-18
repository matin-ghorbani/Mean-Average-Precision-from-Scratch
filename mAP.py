from collections import Counter

import torch
from IoU import intersection_over_union


def mean_average_precision(
    predictions,
    true_boxes,
    iou_thresh=.5,
    box_format='corners',
    num_classes=20
):
    average_precision = []
    epsilon = 1e-6

    # Predictions: [[train_idx, class, probability, x1, y1, x2, y2], ...]
    for c in range(num_classes):
        detections = []
        ground_truths = []

        for pred in predictions:
            if pred[1] == c:
                detections.append(pred)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # img 0 has 3 bboxes
        # img 1 has 5 bboxes
        # amount_bboxes = {0: 3, 1: 5}
        amount_bboxes = Counter([
            gt[0]
            for gt in ground_truths
        ])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # After this loop:
        # amount_bboxes = {
        #     0: torch.tensor([0, 0, 0]),
        #     1: torch.tensor([0, 0, 0, 0, 0])
        # }

        detections = detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox
                for bbox in ground_truths
                if bbox[0] == detection[0]
            ]
            best_iou = 0
            best_gt_idx = None

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),  # x1, y1, x2, y2
                    torch.tensor(gt[3:]),  # x1, y1, x2, y2
                    box_format=box_format
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            if best_iou > iou_thresh:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            
            else:
                FP[detection_idx] = 1
        
        # [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
        TP_cum_sum = torch.cumsum(TP, dim=0)
        FP_cum_sum = torch.cumsum(FP, dim=0)

        recalls = TP_cum_sum / (total_true_bboxes + epsilon)
        precisions = TP_cum_sum / (TP_cum_sum + FP_cum_sum + epsilon)
        
        recalls = torch.concat((
            torch.tensor([0]),
            recalls
        ))
        precisions = torch.concat((
            torch.tensor([1]),
            precisions
        ))

        average_precision.append(torch.trapz(  # To calculate the area
            precisions,  # Y values
            recalls  # X values
        ))
    
    return sum(average_precision) / len(average_precision)
