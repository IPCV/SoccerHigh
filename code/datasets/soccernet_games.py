import os
import torch
import json
import numpy as np

import torchvision.transforms.v2 as v2

from torch.utils.data import Dataset 
from pathlib import Path
from SoccerNet.utils import getListGames
from tqdm import tqdm
from PIL import Image
from typing import Union
from omegaconf import ListConfig
from configparser import ConfigParser
from enum import Enum


class SoccerNetGames(Dataset):
    def __init__(
            self,
            data_dir: Union[Path, str],
            frame_rate: int = 2,
            split: str = None,
            tiny: int = None,
            game_list: Union[ListConfig, str] = None,
            init_file: Union[str, Path] = 'video.ini',
            include_actions: bool = False,
            include_cameras: bool = False,
            labels_file: Union[str, Path] = 'Labels-v2.json',
            cameras_file: Union[str, Path] = 'Labels-cameras.json'  
    ):
        self.data_dir = data_dir if isinstance(data_dir, Path) else Path(data_dir)
        self.dataset_info = {
            'split': split,
            'frame_rate': frame_rate,
            'tiny': tiny,
            'init_file': init_file if isinstance(init_file, Path) else Path(init_file),
            'labels_file': labels_file if isinstance(labels_file, Path) else Path(labels_file),
            'cameras_file': cameras_file if isinstance(cameras_file, Path) else Path(cameras_file),
            'include_actions': include_actions,
            'include_cameras': include_cameras
        }
        
        if game_list:
            self.set_manual_game_list(game_list)

    @property
    def game_list(self):
        if hasattr(self, 'manual_game_list'):
            return self.manual_game_list
        else:
            split, tiny = self.dataset_info['split'], self.dataset_info['tiny']
            return getListGames(split=split) if not tiny else getListGames(split=split)[:tiny]
    
    @property
    def info(self):
        return self.dataset_info
    
    def set_manual_game_list(self, game_list):
        if isinstance(game_list, ListConfig):
            if not hasattr(self, 'manual_game_list'):
                setattr(self, 'manual_game_list', list(game_list))
            else:
                self.manual_game_list.extend(game_list)
        else:
            if '.txt' in game_list:
                with open(game_list, 'r') as file:
                    if not hasattr(self, 'manual_game_list'):
                        setattr(self, 'manual_game_list', [line.strip() for line in file])
                    else:
                        for line in file:
                            self.manual_game_list.extend(line.strip())
            else:
                raise TypeError('Please use a ListConfig or a .txt file to define the games')
    
    def get_games_path(self):
        return [self.data_dir.joinpath(Path(game_path)) for game_path in self.game_list]
    
    def load_frames(self, data_dir: Path, half:int = 0, transform: v2 = None):
        data_dir = data_dir.joinpath(f"{half}_HQ_224p") if half != 0 else data_dir.joinpath("summary_224p")

        # Check if the directory exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"The directory {data_dir} does not exist.")
        
        frame_list = os.listdir(data_dir)
        frame_list.sort()
        
        frames = []
        with tqdm(total=len(frame_list)) as pbar:
            for img in frame_list:
                filename = data_dir.joinpath(f"{img}")
                image = Image.open(filename)
                frames.append(transform(image) if transform else image)
                pbar.update(1)
                
        return frames

    def load_features(self, data_dir: Path, half: int = 0, backbone: str = 'dino', sequence: str = 'frame'):
        if 'dino' in backbone:
            if 'v2' in backbone:
                data_dir = data_dir.joinpath(f"{half}_HQ_{backbone}_384.npy") if half != 0 else data_dir.joinpath(f"summary_{backbone}.npy")
            else:
                data_dir = data_dir.joinpath(f"{half}_{backbone}.npy") if half != 0 else data_dir.joinpath(f"summary_{backbone}.npy")
        elif 'videoMAEv2' in backbone:
            data_dir = data_dir.joinpath(f"{half}_{self.dataset_info['features']['filename']}") if 'frame' in sequence else data_dir.joinpath(f"{half}_{self.dataset_info[f'features_{sequence}']['filename']}")
        elif 'ResNet152' in backbone:
            data_dir = data_dir.joinpath(f"{half}_HQ_{backbone}_2048.npy") if half != 0 else data_dir.joinpath(f"summary_{backbone}.npy")
        elif 'CLIP' in backbone:
            data_dir = data_dir.joinpath(f"{half}_{backbone}-ViTB32_datacomp_xl_s13b_b90k_512.npy") if half != 0 else data_dir.joinpath(f"summary_{backbone}.npy")
        else:
            raise NotImplementedError(f"{backbone} not implemented yet.")

        # Check if the directory exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"The directory {data_dir} does not exist.")
        
        return torch.from_numpy(np.load(data_dir))
    
    def save_features(self, features, data_dir: Path, half: int = 0, backbone: str = 'dino', overwrite=False):
        filename = data_dir.joinpath(f"{half}_{backbone}.npy") if half != 0 else data_dir.joinpath(f"summary_{backbone}.npy")
        if not os.path.exists(filename) or overwrite:
            np.save(filename, features.numpy())
            print(f"Features successfully saved to {filename}")

    def load_games_info(self):
        game_list = self.game_list
        if game_list:
            # Create empty lists
            games_info = []
            if self.dataset_info['include_actions']:
                actions_info = []
            if self.dataset_info['include_cameras']:
                cam_info = []
            # Create a ConfigParser object for 'video.ini' file
            parser = ConfigParser()
            for game in game_list:
                # Append game timing information
                games_info.append(self.get_game_times(game, parser))
                if self.dataset_info['include_actions']:
                    actions_info.append(self.get_game_actions(game, (games_info[-1][1]['start'], games_info[-1][2]['start'])))
                if self.dataset_info['include_cameras']:
                    cam_info.append(self.get_game_camera_info(game, (games_info[-1][1]['start'], games_info[-1][2]['start'])))
            # Save times and actions info into dataset_info
            self.dataset_info['games_info'] = {'time': games_info, 'actions': actions_info} if self.dataset_info['include_actions'] else {'time': games_info}
            if self.dataset_info['include_cameras']:
                self.dataset_info['games_info']['cameras'] = cam_info
            
    def get_game_times(self, game, parser):
        # Initialize the dictionary that will hold the formatted information
        video_info = {}
        # Build complete path name
        fpath = self.data_dir.joinpath(f"{game}/{self.dataset_info['init_file']}")
        # Read game file
        file = parser.read(fpath)
        # Loop through the sections and extract start time and duration
        for section in parser.sections():
            # Extract the index from the section name (e.g., "1_HQ.mkv" -> 1)
            index = int(section.split('_')[0])
            # Get start_time and duration values and multiply by fps
            start_time = float(parser[section]['start_time_second']) * self.dataset_info['frame_rate']
            duration = float(parser[section]['duration_second']) * self.dataset_info['frame_rate']
            # Add to the dictionary
            video_info[index] = {'start': int(start_time), 'duration': int(duration)}
        return video_info
    
    def get_game_actions(self, game, offsets=(0,0)):
        # Build complete path name
        fpath = self.data_dir.joinpath(f"{game}/{self.dataset_info['labels_file']}")
        # Load JSON data from a file
        with open(fpath, 'r') as file:
            data = json.load(file)
        # Initialize the dictionary that will hold the formatted information
        actions_info = {i+1: {} for i in range(2)}
        # Extract required information
        for annotation in data['annotations']:
            # Avoid actions in the background
            if annotation['visibility'] == 'not shown':
                continue
            
            game_time = annotation['gameTime']
            label = annotation['label']
            
            # Split gameTime to get half and time
            half_str, time_str = game_time.split(' - ')
            half = int(half_str.strip())
            
            # Convert time to frame_id
            frame_id = convert_time_to_frame(time_str, self.dataset_info['frame_rate'], offsets[half-1])
            
            # Initialize the frame_id in the half if it doesn't exist yet
            if frame_id not in actions_info[half]:
                actions_info[half][frame_id] = []
            elif actions_info[half][frame_id][-1] == label:
                # Avoid repeated actions
                continue
            
            # Append action to the current frame_id
            actions_info[half][frame_id].append(label)
            
        return actions_info
    
    def get_game_camera_info(self, game, offsets=(0,0)):
        # Build complete path name
        fpath = self.data_dir.joinpath(f"{game}/{self.dataset_info['cameras_file']}")
        # Load JSON data from a file
        with open(fpath, 'r') as file:
            data = json.load(file)
        # Initialize the dictionary that will hold the formatted information
        cameras_info = {i+1: {} for i in range(2)}
        # Add each camera annotation
        for annotation in data['annotations']:
            game_time = annotation['gameTime']

            # Split gameTime to get half and time
            half_str, time_str = game_time.split(' - ')
            half = int(half_str.strip())

            # Convert time to frame_id
            frame_id = convert_time_to_frame(time_str, self.dataset_info['frame_rate'], offsets[half-1])
            
            # Update cameras type
            cameras_info[half].update({
                frame_id: {
                    'change_type': annotation['change_type'],
                    'camera_type': annotation['label'],
                }
            })

            # Add replay info, ignore it if it's a real-time play
            if 'replay' in annotation['replay']:
                cameras_info[half][frame_id]['replay'] = {
                    'half': int(annotation['link']['half']),
                    'frame_id': convert_time_to_frame(annotation['link']['time'], self.dataset_info['frame_rate'], offsets[half-1]),
                    'team': annotation['link']['team'],
                    'action': annotation['link']['label']
                }

        return cameras_info
    
    class ActionLabels(Enum):
        Penalty = 0
        Kick_off = 1
        Goal = 2
        Substitution = 3
        Offside = 4
        Shots_on_target = 5
        Shots_off_target = 6
        Clearance = 7
        Ball_out_of_play = 8
        Throw_in = 9
        Foul = 10
        Indirect_free_kick = 11
        Direct_free_kick = 12
        Corner = 13
        Yellow_card = 14
        Red_card = 15
        Yellow_to_red_card = 16

    class CameraViewType(Enum):
        Close_up_player_or_field_referee = 0
        Main_behind_the_goal = 1
        Spider_camera = 2
        Main_camera_right = 3
        Main_camera_left = 4
        Main_camera_center = 5
        Close_up_side_staff = 6
        Close_up_behind_the_goal = 7
        Public = 8
        Inside_the_goal = 9
        Other = 10
        Close_up_corner = 11
        Goal_line_technology_camera = 12
        Unknown = 13
        
    @classmethod
    def to_dict(cls):
        return {event.value: event.name.replace("_", " ") for event in cls.ActionLabels}
    
    @classmethod
    def get_action_id(cls, event_name):
        # Format the name to match the Enum member names
        formatted_name = event_name.strip().replace(" ", "_").replace("->", "_to_").replace("-", "_")
        try:
            return cls.ActionLabels[formatted_name].value
        except KeyError:
            return None  # Return None if the event name doesn't match any label
        
    @classmethod
    def get_camera_view_id(cls, view_name):
        formatted_name = (
            view_name.strip()
            .replace(" ", "_")
            .replace("->", "_to_")
            .replace("-", "_")
            .replace("/", "_")
        ) if view_name != '' else 'Unknown'
        try:
            return cls.CameraViewType[formatted_name].value
        except KeyError:
            return None  # Return None if the view name doesn't match any label
        
# Function to convert time string to frame_id
def convert_time_to_frame(time_str, fps, offset=0):
    minutes, seconds = map(int, time_str.split(':'))
    total_seconds = minutes * 60 + seconds
    return (total_seconds * fps) + offset