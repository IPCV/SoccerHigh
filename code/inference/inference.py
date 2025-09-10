import numpy as np

from collections import defaultdict

from evaluation.evaluate import nms
from inference.utils import save_to_json, frameidx2timestamp


def keyshot_selection(predictions, dataset, threshold=0.5):
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
    windows = {i: np.stack(windows[i]) for i in range(n_outputs)}
        
    # Create an empty map containing each frame predictions as well as the frame ids of each window
    idx_map = {match_id: {1: {**{i: {'preds': []} for i in range(n_outputs)},'ids': []}, 2: {**{i: {'preds': []} for i in range(n_outputs)},'ids': []}} for match_id in range(len(dataset.nframes))}
    
    for i in range(n_outputs):
        for idx in range(windows[i].shape[0]):
            # Get the information of the current window
            window = dataset.windows[idx]
            match_id, half_id, window_ids = window['match'], window['half'], window['frames']
            # Save each prediction in the corresponding match
            idx_map[match_id][half_id+1][i]['preds'].append(windows[i][idx])
            # Save the corresponding frame ids only the first time
            if i == 0:
                frame_idxs = np.asarray([range(window_ids[0], window_ids[-1]+1)]).reshape(-1, 1)
                # Check incomplete windows
                mod = frame_idxs.shape[0] % dataset.dataset_info['frames_per_window']
                frame_idxs = np.concatenate([frame_idxs, np.tile(frame_idxs[-1:], (dataset.dataset_info['frames_per_window'] - mod, 1))], axis=0) if mod != 0 else frame_idxs
                # Append current window indices
                idx_map[match_id][half_id+1]['ids'].append(frame_idxs)

    # Create a map for the key shots proposal
    keyshots = {match_id: {1: {'segment': (0,0), 'score': 0.0}, 2: {'segment': (0,0), 'score': 0.0}} for match_id in range(len(dataset.nframes))}
    for match_idx in idx_map.keys():
        for half_idx in idx_map[match_idx].keys():
            # Get the number of different outputs
            n = len(idx_map[match_idx][half_idx].keys()) - 1
            if n > 1:
                # Get the positive examples from the predictions
                positives = (np.asarray(idx_map[match_idx][half_idx][0]['preds']) > threshold).astype(bool)
                # Keep the boundaries for the positive predictions only
                boundaries = np.asarray([
                    np.expand_dims(np.asarray(idx_map[match_idx][half_idx][1]['preds'])[:, :, i], axis=-1)[positives] 
                    for i in range(np.asarray(idx_map[match_idx][half_idx][1]['preds']).shape[-1])
                ]).reshape((-1, np.asarray(idx_map[match_idx][half_idx][1]['preds']).shape[-1]))
                # Keep the score for the positive predictions only
                if dataset.dataset_info['predict_center']:
                    if n > 2:
                        center_score = np.asarray(idx_map[match_idx][half_idx][2]['preds'])[positives].reshape((-1, np.asarray(idx_map[match_idx][half_idx][2]['preds']).shape[-1]))
                    else:
                        start = np.expand_dims(np.asarray(idx_map[match_idx][half_idx][1]['preds'])[:, :, 0], axis=-1)
                        end = np.expand_dims(np.asarray(idx_map[match_idx][half_idx][1]['preds'])[:, :, 1], axis=-1)
                        center_score = (np.minimum(start, end) / (np.maximum(start, end) + 1e-8))[positives].reshape((-1, np.asarray(idx_map[match_idx][half_idx][0]['preds']).shape[-1]))

                # Get the frame ids for the positive examples
                frame_idxs = np.concatenate(idx_map[match_idx][half_idx]['ids'])
                # Reshape and keep only positive samples
                shape = np.asarray(idx_map[match_idx][half_idx][0]['preds']).shape
                frame_idxs = frame_idxs.reshape(shape)[positives].reshape((-1, shape[-1]))
                # When positive prediction get the segment boundaries and score
                if boundaries.shape[0] != 0:
                    segments = (np.round(frame_idxs - np.expand_dims(boundaries[:,0], -1)).astype(int), np.round(frame_idxs + np.expand_dims(boundaries[:,1], -1)).astype(int))
                    if dataset.dataset_info['predict_center']:
                        scores = center_score * np.expand_dims(np.asarray(idx_map[match_idx][half_idx][0]['preds'])[positives], -1)
                    else:
                        scores = np.expand_dims(np.asarray(idx_map[match_idx][half_idx][0]['preds'])[positives], -1)
                    # Filter segments using NMS
                    keyshots[match_idx][half_idx]['segment'], keyshots[match_idx][half_idx]['score'] = nms(segments, scores, threshold=0.0)

    # Return keyshots map for every match with its intervals and scores
    return keyshots 

def format_output(output, game_list, save=False, fname=None):
    output = reformat_and_transform_data(output, game_list)

    if save:
        save_to_json(output, fname)

# Combined function to filter, reformat, transform, rename keys, and add empty dicts for missing keys
def reformat_and_transform_data(data, game_list):
    transformed = defaultdict(dict)
    
    # Define mappings for 1 and 2 to "1st half" and "2nd half"
    key_mappings = {1: '1st half', 2: '2nd half'}
    
    # Identify all possible outer keys
    all_outer_keys = range(len(game_list))  # Ensure we cover all games in the list
    
    for outer_key in all_outer_keys:
        # Get the game name from the game list
        game_name = game_list[outer_key]
        
        # Access the sub_dict if it exists, or use an empty dict if it doesn't
        sub_dict = data.get(outer_key, {})
        
        has_data = False  # Flag to track if this outer key has valid data

        for inner_key in range(1, 3):  # Only check for "1st half" and "2nd half"
            values = sub_dict.get(inner_key, {'segment': (0, 0), 'score': 0.0})
            segment = values['segment']
            score = values['score']
            
            # Handle non-array scores directly
            if isinstance(score, float) and score == 0.0:
                continue
            
            # Filter zero scores for arrays, convert segments to list of tuples, flatten scores
            if isinstance(score, np.ndarray):
                non_zero_indices = np.where(score.flatten() != 0.0)[0]
                segment = [tuple(segment[i]) for i in non_zero_indices]
                score = score.flatten()[non_zero_indices].tolist()
            
            # Skip if there's nothing left after filtering
            if not score:
                continue

            has_data = True  # Mark that this outer key has valid data
            
            # Map to "1st half" or "2nd half" based on inner_key
            section_name = key_mappings.get(inner_key, f"Section {inner_key}")
            
            # Build list of {'segment': ..., 'score': ...} dictionaries
            segments_scores = [{'segment': (frameidx2timestamp(seg[0]), frameidx2timestamp(seg[1])), 'score': sc} for seg, sc in zip(segment, score)]
            
            # Assign to transformed dictionary
            transformed[game_name][section_name] = segments_scores

        # If no data was added for this outer_key, ensure it has an empty dict
        if not has_data:
            transformed[game_name] = {}

    return dict(transformed)