import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, average_precision_score, jaccard_score


# Define a global dictionary for heads
HEADS = {
    0: 'labels',
    1: 'boundaries',
    2: 'actions',
    3: 'center'
}

def compute_frame_metrics(predictions, target, metric='f1', classes=1, threshold=0.5):
    args = {
        'y_true': target,
        'y_pred': np.asarray(predictions) > threshold
    }

    if metric == 'accuracy':
        return  balanced_accuracy_score(**args)
    else:
        args.update({
            'labels': None if classes == 1 else np.arange(classes),
            'average': 'binary' if classes == 1 else 'weighted'
        })
        if metric == 'f1':
            return f1_score(**args)
        elif metric == 'precision':
            return  precision_score(**args)
        elif metric == 'recall':
            return  recall_score(**args)
        elif metric == 'AP':
            args = {
                'y_score': predictions,
                'y_true': target
            }
            return average_precision_score(**args)
    return

def average_predictions(predictions, dataset, batch_size, threshold=0.5):
    # Create a map containing each frame predictions and its label
    idx_map = {match_id: {1: {i: {'preds': [], 'label': 0} for i in range(dataset.nframes[match_id][1])}, 2: {i: {'preds': [], 'label': 0} for i in range(dataset.nframes[match_id][2])}} for match_id in range(len(dataset.nframes))}

    # Iterate over all batches:
    for batch_idx, batch in enumerate(predictions):
        for i, clip in enumerate(batch):
            window_idx = i + batch_idx * batch_size # TODO: Review window_idx for incomplete batches
            window = dataset.windows[window_idx]
            match_id, half_id, window_ids = window['match'], window['half'], window['frames']
            labels = dataset.get_labels(window)['labels']
            window_range = np.arange(window_ids[0], window_ids[-1]+1)
            for idx, frame_idx in enumerate(window_range):
                idx_map[match_id][half_id+1][frame_idx]['preds'].append(clip[idx].item())
                idx_map[match_id][half_id+1][frame_idx]['label'] = labels[idx].item()
         
    # Average frames            
    outputs, labels = [], []
    for match_key in idx_map.keys():
        for half_key in idx_map[match_key].keys():
            for frame_key in idx_map[match_key][half_key].keys():
                outputs.append((np.mean(idx_map[match_key][half_key][frame_key]['preds']) > threshold).astype(int))
                labels.append(idx_map[match_key][half_key][frame_key]['label'])
    
    return outputs, labels

def average_windows(windows, dataset, modality, threshold=0.5):
    # Create a map containing each frame predictions and its label
    idx_map = {match_id: {1: {i: {'preds': [], 'label': None} for i in range(dataset.nframes[match_id][1])}, 2: {i: {'preds': [], 'label': None} for i in range(dataset.nframes[match_id][2])}} for match_id in range(len(dataset.nframes))}
    
    for window_idx, clip in enumerate(windows):
        window = dataset.windows[window_idx]
        match_id, half_id, window_ids = window['match'], window['half'], window['frames']
        labels = dataset.get_labels(window)[modality]
        window_range = np.arange(window_ids[0], window_ids[-1]+1)
        
        for idx, frame_idx in enumerate(window_range):
            idx_map[match_id][half_id+1][frame_idx]['preds'].append(clip[idx])
            if idx_map[match_id][half_id+1][frame_idx]['label'] is None:
                idx_map[match_id][half_id+1][frame_idx]['label'] = labels[idx]

    # Average frames            
    for match_key in idx_map.keys():
        for half_key in idx_map[match_key].keys():
            for frame_key in idx_map[match_key][half_key].keys():
                if modality == 'labels' or modality == 'actions':
                    idx_map[match_key][half_key][frame_key]['preds'] = (np.mean(idx_map[match_key][half_key][frame_key]['preds']) > threshold).astype(int)
                else:
                    idx_map[match_key][half_key][frame_key]['preds'] = (np.mean(np.array(idx_map[match_key][half_key][frame_key]['preds'])[:, 0]), np.mean(np.array(idx_map[match_key][half_key][frame_key]['preds'])[:, 1]))
                idx_map[match_key][half_key][frame_key]['label'] = idx_map[match_key][half_key][frame_key]['label']
    
    return idx_map
            
def keyshot_selection(predictions, dataset, threshold=0.5, metrics=None):
    # Number of different head outputs
    n_outputs = len(predictions[0])
    # Initialize a window list for each head output
    windows = {i: [] for i in range(n_outputs)}
    # [n_steps]x[n_heads]x(batch_size, frames_per_window, output_dim)
    for head in predictions:
        # [n_heads]x(batch_size, frames_per_window, output_dim)
        for i, batch in enumerate(head):
            # (batch_size, frames_per_window, output_dim)
            windows[i].extend(batch)
    # [n_heads]x(n_windows, frames_per_window, output_dim)
    windows = {i: np.asarray(windows[i]) for i in range(n_outputs)}
    
    # Create an empty map containing each frame predictions and its label as well as the frame ids of each window
    idx_map = {match_id: {1: {**{i: {'preds': [], 'label': []} for i in range(n_outputs)},'ids': []}, 2: {**{i: {'preds': [], 'label': []} for i in range(n_outputs)},'ids': []}} for match_id in range(len(dataset.nframes))}

    for i in range(n_outputs):
        # Check if overlapping exists
        if dataset.dataset_info['stride'] == dataset.dataset_info['frames_per_window']:
            for idx in range(windows[i].shape[0]):
                # Get the information of the current window
                window = dataset.windows[idx]
                match_id, half_id, window_ids = window['match'], window['half'], window['frames']
                # Get the ground truth
                labels = dataset.get_labels(window)
                try:
                    labels = labels[list(labels.keys())[i]].numpy()
                except IndexError:
                    print("Index out of bounds")
                # Save each prediction ground truth in the corresponding match
                idx_map[match_id][half_id+1][i]['preds'].append(windows[i][idx])
                idx_map[match_id][half_id+1][i]['label'].append(labels)
                # Save the corresponding frame ids only the first time
                if i == 0:
                    frame_idxs = np.asarray([range(window_ids[0], window_ids[-1]+1)]).reshape(-1, 1)
                    # Check incomplete windows
                    mod = frame_idxs.shape[0] % dataset.dataset_info['frames_per_window']
                    frame_idxs = np.concatenate([frame_idxs, np.tile(frame_idxs[-1:], (dataset.dataset_info['frames_per_window'] - mod, 1))], axis=0) if mod != 0 else frame_idxs
                    # Append current window indices
                    idx_map[match_id][half_id+1]['ids'].append(frame_idxs)
        else:
            idx_map_i = average_windows(windows[i], dataset, HEADS[i])
            for match_key, match_dict in idx_map_i.items():
                for half_key, half_dict in match_dict.items():
                    # Process frames in the current half
                    for frame_key, frame_data in half_dict.items():
                        idx_map[match_key][half_key][i]['preds'].append(frame_data['preds'])
                        idx_map[match_key][half_key][i]['label'].append(frame_data['label'])
                        # Save the corresponding frame ids only the first time
                        if i == 0:
                            idx_map[match_key][half_key]['ids'].append(frame_key)
                    # Combine predictions and labels into NumPy arrays and update idx_map
                    preds = np.stack(idx_map[match_key][half_key][i]['preds'])
                    idx_map[match_key][half_key][i]['preds'] = np.expand_dims(preds, axis=-1) if len(preds.shape) == 1 else preds
                    labels = np.stack(idx_map[match_key][half_key][i]['label'])
                    idx_map[match_key][half_key][i]['label'] = np.expand_dims(labels, axis=-1) if len(labels.shape) == 1 else labels
                    # Update frame indices only once per half
                    if i == 0:
                        idx_map[match_key][half_key]['ids'] = np.expand_dims(np.stack(idx_map[match_key][half_key]['ids']), axis=-1)

    # Create a map for the key shots proposal
    keyshots = {match_id: {1: {'segment': (0,0), 'score': 0.0, 'scores': [], 'ground_truth': (0,0), 'TP': 0, 'TP_masked': 0, 'FP': 0, 'FP_masked': 0, 'FN': 0, 'FN_masked': 0}, 2: {'segment': (0,0), 'score': 0.0, 'scores': [], 'ground_truth': (0,0), 'TP': 0, 'TP_masked': 0, 'FP': 0, 'FP_masked': 0, 'FN': 0, 'FN_masked': 0}} for match_id in range(len(dataset.nframes))}
    for match_idx in idx_map.keys():
        for half_idx in idx_map[match_idx].keys():
            # Get the number of different outputs
            n = len(idx_map[match_idx][half_idx].keys()) - 1
            if n > 1:
                # Get the positive examples from the predictions
                labels_key = [key for key, value in HEADS.items() if value == 'labels'][0]
                positives = (np.asarray(idx_map[match_idx][half_idx][labels_key]['preds']) > threshold).astype(bool)
                # Keep the boundaries for the positive predictions only
                boundaries_key = [key for key, value in HEADS.items() if value == 'boundaries'][0]
                # Avoid empty predictions
                if idx_map[match_idx][half_idx][boundaries_key]['preds'].__len__() == 0:
                    continue
                if dataset.dataset_info['stride'] == dataset.dataset_info['frames_per_window']:
                    boundaries = np.asarray([
                        np.expand_dims(np.asarray(idx_map[match_idx][half_idx][boundaries_key]['preds'])[:, :, i], axis=-1)[positives] 
                        for i in range(np.asarray(idx_map[match_idx][half_idx][boundaries_key]['preds']).shape[-1])
                    ]).reshape((-1, np.asarray(idx_map[match_idx][half_idx][boundaries_key]['preds']).shape[-1]))
                else:
                    boundaries =  np.asarray([
                        np.expand_dims(idx_map[match_idx][half_idx][boundaries_key]['preds'][:,i], axis=-1)[positives]
                        for i in range(idx_map[match_idx][half_idx][boundaries_key]['preds'].shape[-1])
                    ]).reshape((-1, idx_map[match_idx][half_idx][boundaries_key]['preds'].shape[-1]))

                # Keep the score for the positive predictions only
                if dataset.dataset_info['predict_center']:
                    if n > 2:
                        center_key = [key for key, value in HEADS.items() if value == 'center'][0]
                        center_score = np.asarray(idx_map[match_idx][half_idx][center_key]['preds'])[positives].reshape((-1, np.asarray(idx_map[match_idx][half_idx][center_key]['preds']).shape[-1]))
                    else:
                        if dataset.dataset_info['stride'] == dataset.dataset_info['frames_per_window']:
                            start = np.expand_dims(np.asarray(idx_map[match_idx][half_idx][boundaries_key]['preds'])[:, :, 0], axis=-1)
                            end = np.expand_dims(np.asarray(idx_map[match_idx][half_idx][boundaries_key]['preds'])[:, :, 1], axis=-1)
                            center_score = (np.minimum(start, end) / (np.maximum(start, end) + 1e-8))[positives].reshape((-1, np.asarray(idx_map[match_idx][half_idx][labels_key]['preds']).shape[-1]))
                        else:
                            start = idx_map[match_idx][half_idx][boundaries_key]['preds'][:,0]
                            end = idx_map[match_idx][half_idx][boundaries_key]['preds'][:,1]
                            center_score = (np.minimum(start, end) / (np.maximum(start, end) + 1e-8))[positives].reshape((-1, idx_map[match_idx][half_idx][labels_key]['preds']).shape[-1])
                            
                # Get the frame ids for the positive examples
                if dataset.dataset_info['stride'] == dataset.dataset_info['frames_per_window']:
                    frame_idxs = np.concatenate(idx_map[match_idx][half_idx]['ids'])
                    # Reshape and keep only positive samples
                    shape = np.asarray(idx_map[match_idx][half_idx][labels_key]['preds']).shape
                    frame_idxs = frame_idxs.reshape(shape)[positives].reshape((-1, shape[-1]))
                else:
                    frame_idxs = idx_map[match_idx][half_idx]['ids'][positives].reshape((-1, idx_map[match_idx][half_idx][labels_key]['preds'].shape[-1]))
                    
                # When positive prediction get the segment boundaries and score
                if boundaries.shape[0] != 0:
                    segments = (np.round(frame_idxs - np.expand_dims(boundaries[:,0], -1)).astype(int), np.round(frame_idxs + np.expand_dims(boundaries[:,1], -1)).astype(int))
                    if dataset.dataset_info['predict_center']:
                        scores = center_score * np.expand_dims(np.asarray(idx_map[match_idx][half_idx][labels_key]['preds'])[positives], -1)
                    else:
                        scores = np.expand_dims(np.asarray(idx_map[match_idx][half_idx][labels_key]['preds'])[positives], -1)
                    # Filter segments using NMS
                    keyshots[match_idx][half_idx]['segment'], keyshots[match_idx][half_idx]['score'] = nms(segments, scores)
                    # Convert GT intervals to a NumPy array of [start, end] pairs
                    keyshots[match_idx][half_idx]['ground_truth'] = np.array([[item['start'], item['end']] for item in dataset.intervals[match_idx][half_idx]])
                    # Save prediction scores
                    keyshots[match_idx][half_idx]['scores'] = get_frame_score_per_segment(keyshots[match_idx][half_idx]['segment'], np.concatenate(idx_map[match_idx][half_idx][0]['preds']).squeeze())
                    if metrics:
                        if dataset.dataset_info['masked']:
                            half_info = dataset.dataset_info['games_info']['time'][match_idx][half_idx]
                            mask = np.zeros(dataset.nframes[match_idx][half_idx])
                            mask[half_info['start']:half_info['start']+half_info['duration']] = 1
                            metrics, keyshots = update_metrics(keyshots, match_idx, half_idx, metrics, mask)
                        else:
                            metrics, keyshots = update_metrics(keyshots, match_idx, half_idx, metrics)
    
    # Return keyshots map for every match with its intervals and scores and the computed evaluation metrics
    return keyshots, metrics, idx_map

def get_frame_score_per_segment(segments, predictions):
    scores = []
    for segment in segments:
        start, end = segment[0], min(predictions.shape[0], segment[1] + 1)
        scores.append(predictions[start:end].tolist())
    return scores

def nms(segments, scores, threshold=0.0):
    # Only keep segments where the end time is greater than the start time
    keep = (segments[1] > segments[0]).astype(bool)
    segments = (segments[0][keep], segments[1][keep])
    scores = scores[keep]

    # Sort indeces following the importance scores from highest to lowest
    args_sorted = np.argsort(-scores)

    # Sort the importance scores and segments
    scores_remaining = scores[args_sorted]
    segments_remaining = (segments[0][args_sorted], segments[1][args_sorted])

    filtered_segments = []
    filtered_scores = []
    # Apply NMS
    while segments_remaining[0].shape[0] > 0:
        segment = segments_remaining[0][0], segments_remaining[1][0]
        filtered_segments.append(segment)
        filtered_scores.append(scores_remaining[0])
        
        # Compute IoU
        intersect =  np.minimum(segments_remaining[1], segment[1]) - np.maximum(segments_remaining[0], segment[0])
        intersect[intersect < 0] = 0
        union = np.maximum(segments_remaining[0], segment[1]) - np.minimum(segments_remaining[0], segment[0])
        union[union <= 0] = 1e-8
        
        iou = intersect / union
        keep_indices = (iou <= threshold)
        
        # Filter
        segments_remaining = (segments_remaining[0][keep_indices], segments_remaining[1][keep_indices])
        scores_remaining = scores_remaining[keep_indices]

    return np.asarray(filtered_segments), np.expand_dims(np.asarray(filtered_scores), axis=-1)

def update_metrics(keyshots, match_idx, half_idx, metrics, mask=None):
    def flatten_segments(segments, mask=None):
        """Flatten segments into a set of frames, optionally applying a mask."""
        frames = set()
        for seg in segments:
            if mask is not None:
                indices = np.where(mask[seg[0]:seg[1] + 1] == 1)[0]
                if indices.size > 0:
                    indices += seg[0]
                    frames.update(range(indices[0], indices[-1] + 1))
            else:
                frames.update(range(seg[0], seg[1] + 1))
        return frames

    def compute_metrics(pred_frames, gt_frames):
        """Compute TP, FP, FN, and return their counts."""
        true_positives = pred_frames.intersection(gt_frames)
        false_positives = pred_frames.difference(gt_frames)
        false_negatives = gt_frames.difference(pred_frames)
        return len(true_positives), len(false_positives), len(false_negatives)

    def update_keyshot_metrics(metrics, keyshot_data, prefix=""):
        """Update keyshot metrics with computed values."""
        TP, FP, FN = metrics
        keyshot_data[f'TP{prefix}'] = TP
        keyshot_data[f'FP{prefix}'] = FP
        keyshot_data[f'FN{prefix}'] = FN

    def compute_aggregate_metrics(TP, FP, FN):
        """Compute precision, recall, F1, and IoU."""
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        IoU = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        return precision, recall, f1, IoU

    def update_metrics_dict(metrics_dict, values, prefix=""):
        """Update general metrics with aggregated values."""
        metrics_dict[f'precision{prefix}'] += values[0]
        metrics_dict[f'recall{prefix}'] += values[1]
        metrics_dict[f'f1{prefix}'] += values[2]
        metrics_dict[f'IoU{prefix}'] += values[3]

    # Flatten segments for prediction and ground truth
    pred_segments = keyshots[match_idx][half_idx]['segment']
    gt_segments = keyshots[match_idx][half_idx]['ground_truth']
    pred_frames = flatten_segments(pred_segments)
    gt_frames = flatten_segments(gt_segments)

    # Compute and update metrics for unmasked data
    TP, FP, FN = compute_metrics(pred_frames, gt_frames)

    update_keyshot_metrics((TP, FP, FN), keyshots[match_idx][half_idx])

    metrics['TP'] += TP
    metrics['FP'] += FP
    metrics['FN'] += FN

    if half_idx == 2:
        TP += keyshots[match_idx][half_idx - 1]['TP']
        FP += keyshots[match_idx][half_idx - 1]['FP']
        FN += keyshots[match_idx][half_idx - 1]['FN']

        aggregated = compute_aggregate_metrics(TP, FP, FN)

        update_metrics_dict(metrics, aggregated)

        for key, value in zip(['precision', 'recall', 'f1', 'IoU'], aggregated):
            keyshots[match_idx][half_idx][key] = value

    # Handle masked data
    if mask is not None:
        pred_frames_masked = flatten_segments(pred_segments, mask)
        gt_frames_masked = flatten_segments(gt_segments, mask)
        TP_masked, FP_masked, FN_masked = compute_metrics(pred_frames_masked, gt_frames_masked)

        update_keyshot_metrics((TP_masked, FP_masked, FN_masked), keyshots[match_idx][half_idx], prefix="_masked")

        metrics['TP_masked'] += TP_masked
        metrics['FP_masked'] += FP_masked
        metrics['FN_masked'] += FN_masked

        if half_idx == 2:
            TP_masked += keyshots[match_idx][half_idx - 1]['TP_masked']
            FP_masked += keyshots[match_idx][half_idx - 1]['FP_masked']
            FN_masked += keyshots[match_idx][half_idx - 1]['FN_masked']

            aggregated_masked = compute_aggregate_metrics(TP_masked, FP_masked, FN_masked)

            update_metrics_dict(metrics, aggregated_masked, prefix="_masked")

            for key, value in zip(['precision_masked', 'recall_masked', 'f1_masked', 'IoU_masked'], aggregated_masked):
                keyshots[match_idx][half_idx][key] = value

            aggregated_offmatch = compute_aggregate_metrics(TP-TP_masked, FP-FP_masked, FN-FN_masked)

            update_metrics_dict(metrics, aggregated_offmatch, prefix="_offmatch")

            for key, value in zip(['precision_offmatch', 'recall_offmatch', 'f1_offmatch', 'IoU_offmatch'], aggregated_masked):
                keyshots[match_idx][half_idx][key] = value

    return metrics, keyshots

def compute_segment_metrics(metrics, n_matches=None, masked=False):
    def compute_metrics(tp, fp, fn):
        intersection = tp
        union = tp + fp + fn
        IoU = intersection / union if union > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return IoU, precision, recall, f1

    def compute_averages(metrics, n_matches):
        return {key: metrics[key] / n_matches if n_matches else 0 for key in ['IoU', 'precision', 'recall', 'f1']}

    IoU, precision, recall, f1_score = compute_metrics(
        metrics['TP'], metrics['FP'], metrics['FN']
    )
    averages = compute_averages(metrics, n_matches)

    log_metrics = {
        'IoU': IoU,
        'average/IoU': averages['IoU'],
        'precision': precision,
        'average/precision': averages['precision'],
        'recall': recall,
        'average/recall': averages['recall'],
        'f1': f1_score,
        'average/f1': averages['f1'],
    }

    if masked:
        IoU_masked, precision_masked, recall_masked, f1_score_masked = compute_metrics(
            metrics['TP_masked'], metrics['FP_masked'], metrics['FN_masked']
        )
        averages_masked = compute_averages(
            {k: metrics[f"{k}_masked"] for k in ['IoU', 'precision', 'recall', 'f1']}, n_matches
        )
        log_metrics.update({
            'onmatch/IoU': IoU_masked,
            'onmatch/average/IoU': averages_masked['IoU'],
            'onmatch/precision': precision_masked,
            'onmatch/average/precision': averages_masked['precision'],
            'onmatch/recall': recall_masked,
            'onmatch/average/recall': averages_masked['recall'],
            'onmatch/f1': f1_score_masked,
            'onmatch/average/f1': averages_masked['f1'],
        })

        IoU_offmatch, precision_offmatch, recall_offmatch, f1_score_offmatch = compute_metrics(
            metrics['TP']-metrics['TP_masked'], metrics['FP']-metrics['FP_masked'], metrics['FN']-metrics['FN_masked']
        )

        averages_offmatch = compute_averages(
            {k: metrics[f"{k}_offmatch"] for k in ['IoU', 'precision', 'recall', 'f1']}, n_matches
        )

        log_metrics.update({
            'offmatch/IoU': IoU_offmatch,
            'offmatch/average/IoU': averages_offmatch['IoU'],
            'offmatch/precision': precision_offmatch,
            'offmatch/average/precision': averages_offmatch['precision'],
            'offmatch/recall': recall_offmatch,
            'offmatch/average/recall': averages_offmatch['recall'],
            'offmatch/f1': f1_score_offmatch,
            'offmatch/average/f1': averages_offmatch['f1'],
        })

    return log_metrics

def get_metric_per_match(keyshots, metric):
    scores = [
        segments[2][metric]  # Get the value of the metric
        for segments in keyshots.values()  # Loop through the values of keyshots
        if 2 in segments and metric in segments[2]  # Check if key 2 and the metric exist
    ]
    return scores

def select_temporal_keyshots(keyshots, clip_segment=False):
    """
    Select top-scoring keyshot segments across both halves of each match,
    constrained to the total number of ground truth (GT) frames.

    Segments are ranked by their confidence scores. The function selects as many
    predicted segments as needed to match the number of GT frames, either fully or 
    by clipping the last segment if `clip_segment=True`.

    The selected segments are then separated by half (1st or 2nd half) using
    the original indexing order.

    Parameters:
    -----------
    keyshots : dict
        Nested dictionary containing predicted segments and scores along with
        ground truth segments for each match and half. Structure:
        {
            match_idx: {
                1: {
                    'segment': [(start1, end1), ...],
                    'score': [score1, ...],
                    'ground_truth': [(gt_start1, gt_end1), ...]
                },
                2: {
                    ...
                }
            },
            ...
        }

    clip_segment : bool, default=False
        If True, the last selected segment may be clipped to exactly match the
        total number of GT frames.

    Returns:
    --------
    selected_keyshots : dict
        Dictionary structured identically to `keyshots`, but only containing
        the selected top segments and their scores. Structure:
        {
            match_idx: {
                1: [{'segment': (start, end), 'score': s}, ...],
                2: [{'segment': (start, end), 'score': s}, ...]
            },
            ...
        }
    """
    # Initialize result container
    selected_keyshots = {
        match_idx: {half_idx: {'segment': [], 'score': [], 'ground_truth': []} for half_idx in keyshots[match_idx].keys()}
        for match_idx in keyshots
    }

    for match_idx in keyshots.keys():
        all_pred_segments = []
        all_pred_scores = []
        all_gt_segments = []
        n_segments = {half_idx: 0 for half_idx in keyshots[match_idx].keys()} # Number of predicted segments per half

        # Accumulate all segments and scores from both halves
        for half_idx in keyshots[match_idx].keys():
            pred_segments = keyshots[match_idx][half_idx]['segment']
            pred_scores = keyshots[match_idx][half_idx]['score']

            # Avoid empty representations
            if not 'ground_truth' in keyshots[match_idx][half_idx].keys() or len(pred_segments) == 2:
                continue
            
            gt_segments = keyshots[match_idx][half_idx]['ground_truth']

            selected_keyshots[match_idx][half_idx]['ground_truth'] = gt_segments

            n_segments[half_idx] = len(pred_segments)

            all_pred_segments.extend(pred_segments)
            all_pred_scores.extend(pred_scores)
            all_gt_segments.extend(gt_segments)

        # Sort indices of predictions by descending score
        sorted_indices = sorted(
            range(len(all_pred_scores)), 
            key=lambda i: all_pred_scores[i], 
            reverse=True
        )
        sorted_pred_segments = [all_pred_segments[i] for i in sorted_indices]
        
        # Total number of frames in ground truth segments
        total_gt_frames = sum(end - start + 1 for start, end in all_gt_segments)

        # Select predicted segments up to the GT frame count
        selected_pred_segments = []
        frame_accumulator = 0

        for segment in sorted_pred_segments:
            start, end = segment
            segment_length = end - start + 1

            if frame_accumulator + segment_length <= total_gt_frames:
                selected_pred_segments.append(segment)
                frame_accumulator += segment_length
            elif clip_segment:
                # Clip last segment if needed
                remaining_frames = total_gt_frames - frame_accumulator
                if remaining_frames > 0:
                    selected_pred_segments.append((start, start + remaining_frames - 1))
                break

        # Use sorted indices to find which half each selected segment belongs to
        clipped_sorted_indices = sorted_indices[:len(selected_pred_segments)]
        
        for idx in clipped_sorted_indices:
            # If idx is less than number of segments from half 1, it's from half 1
            half_idx = 1 if idx < n_segments[1] else 2

            selected_keyshots[match_idx][half_idx]['segment'].append(all_pred_segments[idx])
            selected_keyshots[match_idx][half_idx]['score'].append(all_pred_scores[idx])
 
    return selected_keyshots

def compute_temporal_metrics(keyshots, n_frames, average=False, preds=None):
    """
    Compute temporal-averaged AP, F1, and IoU for predicted segments against ground truth.

    Parameters
    ----------
    keyshots : dict
        Predicted segments and their scores, per match and half.
    gt : dict
        Ground truth segments per match and half.
    n_frames : dict
        Number of frames per match and half.
    average : bool
        Whether to return global average metrics instead of per-match.

    Returns
    -------
    dict
        Either per-match metrics or global average metrics.
    """
    metrics = {
        match_idx: {'AP': 0, 'f1_score': 0, 'IoU': 0} for match_idx in keyshots
    }

    for match_idx in keyshots.keys():
        total_frames = sum(n_frames[match_idx][half] for half in n_frames[match_idx])
        prediction, target = np.zeros(total_frames), np.zeros(total_frames)
        
        offsets = {
            1: 0,
            2: n_frames[match_idx][1]
        }

        for half_idx in keyshots[match_idx].keys():
            segments = keyshots[match_idx][half_idx]['segment']
            scores = keyshots[match_idx][half_idx]['score']
            ground_truth = keyshots[match_idx][half_idx]['ground_truth']

            # Fill prediction array with scores for predicted segments
            for idx, segment in enumerate(segments):
                start, end = segment[0] + offsets[half_idx], segment[1] + 1 + offsets[half_idx] 
                if preds is None:
                    # One score per shot
                    prediction[start:end] = scores[idx]
                else:
                    # Score per frame
                    score = np.concatenate(preds[match_idx][half_idx][0]['preds']).squeeze()[segment[0]:segment[1]+1]
                    for i, s in enumerate(score):
                        if start + i < len(prediction):
                            prediction[start+i] = s 

            
            # Fill target array with ones for ground truth segments
            for segment in ground_truth:
                start, end = segment[0] + offsets[half_idx], segment[1] + 1 + offsets[half_idx]
                target[start:end] = 1

        # Compute metrics
        try:
            ap = average_precision_score(y_true=target, y_score=prediction)
        except ValueError:
            ap = 0.0

        pred_binary = (prediction > 0.0).astype(int)
        f1 = f1_score(y_true=target, y_pred=pred_binary, zero_division=0)
        precision = precision_score(y_true=target, y_pred=pred_binary)
        recall = recall_score(y_true=target, y_pred=pred_binary)
        iou = jaccard_score(y_true=target, y_pred=pred_binary, zero_division=0)
        
        metrics[match_idx] = {'AP': ap, 'precision':precision, 'recall': recall, 'f1_score': f1, 'IoU': iou}

    if average:
        n_matches = len(metrics.keys())
        average_metrics = {
            'mAP': sum(m['AP'] for m in metrics.values()) / n_matches,
            'precision': sum(m['precision'] for m in metrics.values()) / n_matches,
            'recall': sum(m['recall'] for m in metrics.values()) / n_matches,
            'f1_score': sum(m['f1_score'] for m in metrics.values()) / n_matches,
            'mIoU': sum(m['IoU'] for m in metrics.values()) / n_matches
        }
        return average_metrics
    
    return metrics

def compute_shot_mAP(keyshots, nframes):
    y_true, y_score = np.zeros(1), np.zeros(1)

    for match_idx in keyshots.keys():
        for half_idx in keyshots[match_idx].keys():
            n_frames = nframes[match_idx][half_idx]
            gt_frames, pred_frames = np.zeros(n_frames), np.zeros(n_frames)
            segments = keyshots[match_idx][half_idx]['segment']
            scores = keyshots[match_idx][half_idx]['scores']
            ground_truth = keyshots[match_idx][half_idx]['ground_truth']
            
            if not scores:
                continue

            for gt in ground_truth:
                start, end = gt[0], min(n_frames, gt[1] + 1)
                gt_frames[start:end] = 1

            for segment, score in zip(segments, scores):
                start, end = segment[0], min(n_frames, segment[1] + 1)
                for i, idx in enumerate(range(start,end)):
                    try:
                        pred_frames[idx] = score[i]
                    except IndexError:
                        print("Index out of bounds")
                        
            y_true = np.concatenate([y_true, gt_frames])
            y_score = np.concatenate([y_score, pred_frames])

    mAP = average_precision_score(
        y_true=y_true,
        y_score=y_score
    )

    return mAP

def print_evaluation(segment_metrics, temporal_metrics):
    divider = '=' * 40
    print(f"\n{divider}\nEvaluation Summary\n{divider}")

    print("\nSummarization Metrics:")
    for k, v in segment_metrics.items():
        print(f"  - {k:<20}: {v:.4f}")

    print("\nGT-Conditioned Metrics (@T):")
    for k, v in temporal_metrics.items():
        print(f"  - {k:<20}: {v:.4f}")

    print(f"{divider}\n")