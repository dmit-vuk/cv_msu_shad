def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4

    # Write code here
    x1_l, y1_l, x1_r, y1_r = bbox1
    x2_l, y2_l, x2_r, y2_r = bbox2
    x_l, x_r = max(x1_l, x2_l), min(x1_r, x2_r)
    y_l, y_r = max(y1_l, y2_l), min(y1_r, y2_r)
    if x_l >= x_r or y_l >= y_r:
        return 0
    intersect = (y_r - y_l) * (x_r - x_l)
    union = (y1_r - y1_l)*(x1_r - x1_l) + (y2_r - y2_l)*(x2_r - x2_l) - intersect
    return intersect / union


def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here

        # Step 1: Convert frame detections to dict with IDs as keys
        detections_hyp = {detect[0]: detect[1:] for detect in frame_hyp}
        detections_obj = {detect[0]: detect[1:] for detect in frame_obj}

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        new_match = {}
        for match_id, match_detect in matches.items():
            if match_id in detections_obj and match_id in detections_hyp:
                iou = iou_score(detections_obj[match_id], detections_hyp[match_id])
                if  iou > threshold:
                    dist_sum += iou
                    match_count += 1
                    new_match[match_id] = detections_obj[match_id]
                    detections_obj.pop(match_id)
                    detections_hyp.pop(match_id)

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        ids = []
        for obj_id, obj_detect in detections_obj.items():
            if obj_id in detections_hyp:
                iou = iou_score(detections_hyp[obj_id], obj_detect)
                if iou > threshold:
                    ids.append([iou, obj_id])
        
        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for i in ids:
            dist_sum += i[0]
            match_count += 1
            new_match[i[1]] = detections_obj[i[1]]

        # Step 5: Update matches with current matched IDs
        matches = new_match

    # Step 6: Calculate MOTP
    MOTP = dist_sum / match_count
    return MOTP


def motp_mota(obj, hyp, threshold=0.5):
    """Calculate MOTP/MOTA

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    all_bbox = 0
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Step 1: Convert frame detections to dict with IDs as keys
        detections_hyp = {detect[0]: detect[1:] for detect in frame_hyp}
        detections_obj = {detect[0]: detect[1:] for detect in frame_obj}
        all_bbox += len(frame_obj)

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        new_match = {}
        for match_id, hyp_id in matches.items():
            if match_id in detections_obj and hyp_id in detections_hyp:
                iou = iou_score(detections_obj[match_id], detections_hyp[hyp_id])
                if  iou > threshold:
                    dist_sum += iou
                    match_count += 1
                    detections_obj.pop(match_id)
                    detections_hyp.pop(hyp_id)
        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        ids = []
        for obj_id, obj_detect in detections_obj.items():
            for hyp_id, hyp_detect in detections_hyp.items():
                iou = iou_score(hyp_detect, obj_detect)
                if iou > threshold:
                    ids.append([iou, obj_id, hyp_id])
        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for i in ids:
            dist_sum += i[0]
            match_count += 1
            new_match[i[1]] = i[2]
            if i[1] in detections_obj:
                detections_obj.pop(i[1])
            if i[2] in detections_hyp:
                detections_hyp.pop(i[2])

        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error
        curr_match = set(new_match.keys())
        prev_match = set(matches.keys())
        mismatch_error += len(prev_match & curr_match)

        # Step 6: Update matches with current matched IDs
        matches |= new_match

        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        # All remaining objects are considered misses
        false_positive += len(detections_hyp.keys())
        missed_count += len(detections_obj.keys())

    # Step 8: Calculate MOTP and MOTA
    MOTP = dist_sum / match_count
    MOTA = 1 - (missed_count + false_positive + mismatch_error) / all_bbox
    return MOTP, MOTA
