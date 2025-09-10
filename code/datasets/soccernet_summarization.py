import os
import torch

import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Union
from omegaconf import ListConfig, DictConfig

from datasets.soccernet_games import SoccerNetGames
from datasets.utils import read_srt, subs2segments, join_match_intervals, timestamp2frameidx


class SoccerNetSummarization(SoccerNetGames):
    def __init__(
        self,
        data_dir: Union[Path, str] = '/datasets/soccernet',
        frame_rate: int = 2,
        split: str = None,
        tiny: int = None,
        game_list: Union[ListConfig, str] = None,
        filename: str = 'intervals.srt',
        frames_dir: str = 'HQ_224p',
        window_size_sec: int = 30,
        stride: int = 0,
        regression: bool = False,
        preload_features: DictConfig = None,
        init_file: str = None,
        masked: bool = False,
        include_actions: bool = False,
        include_cameras: bool = False,
        predict_center: bool = False,
        superclass: bool = False
    ):
        super().__init__(
            data_dir,
            frame_rate,
            split,
            tiny,
            game_list, 
            init_file,
            include_actions,
            include_cameras
        )

        self.dataset_info['filename'] = filename
        self.dataset_info['frames_dir'] = frames_dir
        self.dataset_info['frames_per_window'] = window_size_sec * frame_rate
        self.dataset_info['stride'] = stride if stride !=0 else self.dataset_info['frames_per_window']
        self.dataset_info['regression'] = regression
        self.dataset_info['masked'] = masked
        self.dataset_info['predict_center'] = predict_center

        if preload_features:
            self.dataset_info['features'] = {
                'name': preload_features.name,
                'dim': preload_features.ndim,
                'filename': preload_features.filename
            }
            if not superclass:
                self.preload_features(split='half')

        if not superclass:
            self.load_nframes()
            self.load_games_info()
            self.load_summary_segments()
            self.create_windows()
            self.squeeze_windows()
        
    @property
    def samples_per_class(self):
        if hasattr(self, 'intervals') and hasattr(self, 'nframes'):
            # Calculate the total sum of the differences (end - start)
            positive_sum = 0
            for entry in self.intervals:
                for key in entry:
                    for interval in entry[key]:
                        positive_sum += interval['end'] - interval['start'] + 1
            return (self.total_frames - positive_sum, positive_sum)
        else:
            return None
        
    @property
    def total_frames(self):
        if hasattr(self, 'nframes'):
            # Iterate through the list and sum the values for both keys 1 and 2
            total_sum = 0
            for entry in self.nframes:
                total_sum += entry[1] + entry[2]
            return total_sum
        else:
            return None

    @property
    def compression_ratio(self):
        if hasattr(self, 'nframes') and hasattr(self, 'intervals'):
            n_frames = [entry[1] + entry[2] for entry in self.nframes]
            key_frames = [0] * len(n_frames)

            for match_id, entry in enumerate(self.intervals):
                for key in entry:
                    for interval in entry[key]:
                        key_frames[match_id] += interval['end'] - interval['start'] + 1

        return np.asarray(n_frames) / np.asarray(key_frames)

    def preload_features(self, split='half', sequence='frame'):
        # Obtain the list of games
        games_path = self.get_games_path()

        # Initialize the features list
        if 'frame' in sequence:
            self.features = []
        else:
            setattr(self, f'features_{sequence}', [])

        # Loop over all the specified game loading its features
        with tqdm(total=len(games_path), desc="Loading Features") as pbar:
            for game in games_path:
                game_features = []
                for half in range(2):
                    # Use parent loading function
                    feat = self.load_features(
                        data_dir=game, 
                        half=half+1, 
                        backbone=self.dataset_info['features']['name'],
                        sequence=sequence
                    )
                    game_features.append(feat)

                # Update features list depending on its sequence type
                if 'frame' in sequence:
                    self.features.append(game_features if split == 'half' else np.concatenate(game_features))
                else:
                    features_list = getattr(self, f'features_{sequence}')
                    features_list.append(game_features if split == 'half' else np.concatenate(game_features))
                    setattr(self, f'features_{sequence}', features_list)

                # Update progress bar
                pbar.update(1)
    
    def load_summary_segments(self):
        if not self.game_list:
            raise Exception("First define the games list")
        
        self.intervals = []
        
        filename = self.dataset_info['filename']
        
        for game in self.game_list:
            game_path = self.data_dir.joinpath(game)
            intervals = []

            for half in range(2):
                # Read and process subtitle segments
                srt_file = game_path.joinpath(f"{half+1}_{filename}")
                segments_srt = read_srt(srt_file)
                segments = subs2segments(segments_srt)
                
                # Convert start and end times to frame indices
                for segment in segments:
                    segment['start'] = timestamp2frameidx(segment['start'])
                    segment['end'] = timestamp2frameidx(segment['end'])

                # Consider not having segments
                if not segments:
                    segments.append({
                        'start': 0,
                        'end': 0,
                        'idx': 0
                    })

                intervals.append(segments)
            
            # Join intervals for the match
            self.intervals.append(join_match_intervals(intervals))

    def load_nframes(self):
        if not self.game_list:
            raise Exception("First define the games list")
        
        self.nframes = []

        frames_dir = self.dataset_info['frames_dir']

        for game_idx, game in enumerate(self.game_list):
            game_path = self.data_dir.joinpath(game)
            # Check that the game path exists
            if not game_path.exists():
                print(f"Game {game_path} does not exist")
                continue
            # Read the directory and compute the number of frames per half
            if hasattr(self, 'features'):
                nframes = {half+1: self.features[game_idx][half].shape[0] for half in range(2)}
            else:
                nframes = {half+1: len(os.listdir(game_path.joinpath(f"{half+1}_{frames_dir}"))) for half in range(2)}
            self.nframes.append(nframes)

    def create_windows(self, replicate=False, drop_last=False):
        if not hasattr(self, 'nframes') or not self.nframes:
            raise Exception("First compute the number of frames per match")

        frames_per_window = self.dataset_info['frames_per_window']
        stride = self.dataset_info['stride']

        self.windows = [{1: [], 2: []} for _ in range(len(self.nframes))]

        for idx, n in enumerate(self.nframes):
            for half in range(2):
                nframes = n[half+1]
                
                # Iterate over the frames of the feature 'f' with a given stride
                for i in range(0, nframes, stride):
                    window = np.arange(i, min(nframes, i + frames_per_window))
                    # Handle the case when there are fewer frames than 'frames_per_window' remaining, and 'drop_last' is not enabled
                    if (nframes - i - frames_per_window) < 0 and replicate:
                        # Calculate the number of remaining frames needed to fill 'frames_per_window'
                        n_remaining_frames = (i + frames_per_window - nframes)
                        # Replicate the last frame to fill the remaining frames
                        replicate_frames = np.reshape(
                            np.tile(
                                window[-1],
                                n_remaining_frames,
                            ),
                            (
                                n_remaining_frames
                            )
                        )
                        # Concatenate the replicated frames to the window
                        window = np.concatenate([window, replicate_frames], axis=0)

                    # Avoid padding extra windows, when the current window arrives to the number of total frames breaks the loop
                    if min(nframes, i + frames_per_window) == nframes and drop_last:
                        break

                    self.windows[idx][half+1].append(window)

    def squeeze_windows(self):
        windows_flatten =  [{} for w in self.windows for v in w.values() for _ in v]
        idx = 0
        for i, window in enumerate(self.windows):
            for half in range(2):
                for w in window[half+1]:
                    windows_flatten[idx] = {
                        'match': i,
                        'half': half,
                        'frames': w
                    }
                    idx +=1
        self.windows = windows_flatten

    def get_actions_from_interval(self, interval, match_id, half_id, positive=False):
        actions = self.dataset_info['games_info']['actions'][match_id].get(half_id, [])
        ids = []
        if actions:
            for frame_id in actions.keys():
                if positive:
                    if interval['start'] <= frame_id <= interval['end']:
                        # Map & transform to label array
                        ids.extend([SoccerNetGames.get_action_id(action) for action in actions[frame_id]]) 
                else:
                    ids.append([SoccerNetGames.get_action_id(actions[frame_id]) if frame_id in actions.keys() else None for frame_id in interval]) 
        return ids

    def get_labels(self, window):
        # Define dictionary to return
        sample = {}

        # Get the match half intervals and the current window frame indices
        intervals = self.intervals[window['match']][window['half']+1]
        frames = window['frames']
        
        # Check if elements are in the specified range
        in_range = np.any([(frames >= interval['start']) & (frames <= interval['end']) for interval in intervals], axis=0)
        labels = torch.from_numpy(in_range.astype(int)).unsqueeze(-1)

        # Save classification labels
        sample.update({'labels': labels})
        
        # Use 'masked' to mask out of the game frames
        if self.dataset_info['masked']:
            start = self.dataset_info['games_info']['time'][window['match']][window['half']+1]['start']
            end = start + self.dataset_info['games_info']['time'][window['match']][window['half']+1]['duration']
            mask = torch.from_numpy(np.any([(frames >= start) & (frames < end)], axis=0).astype(int)).unsqueeze(-1)
            sample.update({'mask': mask})

        # Define action labels
        if self.dataset_info['include_actions']:
                actions = np.zeros((labels.shape[0], len(self.ActionLabels)))

        # Check if regression and compute its ground truth
        if self.dataset_info['regression']:
            # Define segement boundaries and segment center
            boundaries = np.full((labels.shape[0], 2), -1)  # Initialize with (-1, -1)
            if self.dataset_info['predict_center']:
                center_offset = -1.0 * np.ones_like(labels)

            # Compute the segment information for the positive examples
            for frame_pos in range(labels.shape[0]):
                if labels[frame_pos] == 0:
                    continue
                frame_idx = frames[frame_pos]
                for interval in intervals:
                    # If frame_idx inside an interval get its boundaries
                    if interval['start'] <= frame_idx <= interval['end']:
                        start_offset = frame_idx - interval['start']
                        end_offset = interval['end'] - frame_idx
                        boundaries[frame_pos] = (start_offset, end_offset)
                        if self.dataset_info['predict_center']:
                            center_offset[frame_pos] = min(start_offset, end_offset) / (max(start_offset, end_offset) + 1e-8)
                        if self.dataset_info['include_actions'] == 'positive':
                            # Check if the current interval includes actions
                            actions_id = self.get_actions_from_interval(interval, window['match'], window['half'], True)
                            # If actions detected, activate the corresponding label
                            actions[frame_pos, actions_id] = 1 if actions_id else 0
                        break
            
            # Update the corresponding fields depending on the configuration
            sample.update({'boundaries': torch.from_numpy(boundaries)})
            if self.dataset_info['predict_center']:
                sample.update({'center_offset': torch.from_numpy(center_offset)})

        else:
            if self.dataset_info['include_actions']:
                # Check if the current window includes actions
                actions_id = self.get_actions_from_interval(frames, window['match'], window['half'], False)
                # If actions detected, activate the corresponding label
                for frame_idx in frames:
                    if actions_id[frame_idx] is not None:
                        actions[frame_idx, actions_id[frame_idx]] = 1 

        # Update actions
        if self.dataset_info['include_actions']:
            sample.update({'actions': torch.from_numpy(actions)})
        
        return sample
    
    def replicate_last_element(self, array):
        return torch.cat([array, array[-1:].expand(self.dataset_info['frames_per_window']-array.shape[0], -1)], dim=0)

    def __getitem__(self, index):
        # Get window indices
        window = self.windows[index]
        match_id, half_id, window_ids = window['match'], window['half'], window['frames']

        # Define dict to return
        sample = {
            'match_id': torch.tensor(match_id),
            'half_id': torch.tensor(half_id),
            'window_start_id': torch.tensor(window_ids[0]),
            'window_end_id': torch.tensor(window_ids[-1])
        }
        
        # Load features or frames depending on the modality
        if self.preload_features:
            imgs = self.features[match_id][half_id][window_ids[0]:window_ids[-1]+1]
        else:
            imgs = self.load_frames(self.data_dir.joinpath(self.game_list[match_id]), half_id+1)[window_ids[0]:window_ids[-1]]

        # Apply last element padding for incomplete clips
        if imgs.shape[0] != self.dataset_info['frames_per_window']:
            imgs = self.replicate_last_element(imgs)
        # Save image representations
        sample.update({'imgs': imgs})

        # Get the ground-truth labels
        labels = self.get_labels(window)

        # Save each GT element
        for key, value in labels.items():
            # Apply last element padding for incomplete labels
            if value.shape[0] != self.dataset_info['frames_per_window']:
                value = self.replicate_last_element(value)
            sample.update({key: value})

        # Return clip information
        return sample

    def __len__(self):
        return len(self.windows)