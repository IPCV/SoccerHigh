import os
import torch
import hydra
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from omegaconf import DictConfig
from models.dino import vit_small
from models.transnetv2 import TransNetV2


def compute_transnet_boundaries(dpath, config, device):
    transform = transforms.Compose([
        transforms.Resize((27,48)),
        transforms.PILToTensor()
    ])
    def preprocess(img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = transform(img)
        if img.dtype != torch.uint8:
            img = img.to(torch.uint8)
        return img
    
    # Load or compute feature representations
    if not os.path.exists(dpath) or len(os.listdir(dpath)) == 0:
        raise FileNotFoundError(f'{dpath} not found')
    
    model_path = os.path.join(config.directory, config.name)

    # Check the model path already exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'{model_path} not found')
    
    model = TransNetV2().to(device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    
    images = []
    files = sorted([f for f in os.listdir(dpath)])
    # Read images in order
    for file in files:
        img_path = os.path.join(dpath, file)
        img = Image.open(img_path)
        images.append(preprocess(img))
    images = torch.stack(images).permute(0,2,3,1).unsqueeze(0)

    with torch.no_grad():
    # shape: batch dim x video frames x frame height x frame width x RGB (not BGR) channels
        single, all_frame = [], []
        single_frame_pred, all_frame_pred = model(images.to(device))
        single.append(torch.sigmoid(single_frame_pred).cpu().numpy().squeeze())
        all_frame.append(torch.sigmoid(all_frame_pred["many_hot"]).cpu().numpy().squeeze())

    idxs = [idx for idx, score in enumerate(single[0]) if score > config.threshold]

    start = 0
    chunks = []
    for idx in idxs:
        chunks.append([start, idx+1])
        start = idx+1

    return chunks

def get_video_chunks(indices, n_neighbors=10):
    chunks = []
    idx = 0

    while idx < len(indices):
        data_array = np.array(indices[idx])

        # Define bounds for outliers
        lower_bound = data_array[0] - n_neighbors
        upper_bound = data_array[0] + n_neighbors

        # Filter out outliers
        filtered_data = data_array[(data_array >= lower_bound) & (data_array <= upper_bound)]

        # Find removed values (i.e., those in data_array but not in filtered_data)
        removed_values = set(data_array) - set(filtered_data)
        
        # Convert removed values to a list and sort if needed
        removed_values_list = sorted(list(removed_values))
        
        if removed_values_list and len(filtered_data) > 1:
            if max(filtered_data) == data_array[0] and chunks:
                chunk = chunks.pop(-1)
                chunks.append([chunk[0], chunk[-1] + 1])
                idx += 1
            else:
                last_slice_end = chunks[-1][-1] if chunks else None
                if last_slice_end and last_slice_end != data_array[0]: 
                    if last_slice_end in filtered_data:
                        chunk = chunks.pop(-1)
                        chunks.append([chunk[0], max(filtered_data)+1])
                    else:
                        chunks.append([last_slice_end, data_array[0]])
                        chunks.append([data_array[0], max(filtered_data)+1])
                else:
                    chunks.append([data_array[0], max(filtered_data)+1])
                idx = max(filtered_data) + 1
        elif len(filtered_data) == 1 and chunks:
            chunk = chunks.pop(-1)
            chunks.append([chunk[0], chunk[-1] + 1])
            idx += 1
        else:
            idx += 1
            
    return chunks

def segment_matching(sliding_window, target, N):
    n_dim = target.shape[-1]
    
    features = [torch.stack(s).view(-1, N*n_dim) for s in sliding_window]

    chunks_id = [[[chunk_id + i for i in range(N)] for chunk_id in range(len(s))] for s in sliding_window]
        
    nbrs = [NearestNeighbors(n_neighbors=1, algorithm='auto').fit(f) for f in features]
    d, n_id = zip(*[nbr.kneighbors(target.view(-1, N*n_dim)) for nbr in nbrs])

    half = int(np.argmin(d))
    
    return chunks_id[half][n_id[half].item()], half+1

def display_segments(coincidences, init_frame, path, N, show=False):
    fig, axs = plt.subplots(2, N, figsize=(15, 3.5))
    frames = os.listdir(path)
    
    for id, idx in enumerate(coincidences):
        frame = Image.open(path.joinpath(f"{frames[idx]}"))
        image = Image.open(path.parent.joinpath(f"summary_224p/{frames[id + init_frame]}"))

        axs[0,id].set_title(f"{id + init_frame}")
        axs[0,id].imshow(image)
        axs[0,id].axis('off')
        axs[1,id].set_title(f"{idx}")
        axs[1,id].imshow(frame)
        axs[1,id].axis('off')
    
    plt.show()
    print("\n\n\n")
    
def compute_intervals(outputs, thresh=2.5, frame_rate=2):
    intervals = []

    for chunk in outputs:
        start_time = chunk[0] / frame_rate
        end_time = chunk[-1] / frame_rate

        if intervals and abs(start_time - intervals[-1][-1]) < thresh:
            intervals[-1][-1] = end_time
        else:
            intervals.append([start_time, end_time])
            
    return intervals

def convert_time_intervals(time_intervals):
    def convert_seconds_to_srt_time(seconds):
        # Split seconds into whole seconds and milliseconds
        whole_seconds = int(seconds)
        milliseconds = int((seconds - whole_seconds) * 1000)
        
        # Convert seconds into hours, minutes, seconds
        minutes, seconds = divmod(whole_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        
        # Format time for SRT
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    formatted_intervals = []
    for start, end in time_intervals:
        start_time = convert_seconds_to_srt_time(start)
        end_time = convert_seconds_to_srt_time(end)
        formatted_intervals.append(f"{start_time} --> {end_time}")
    
    return formatted_intervals

def generate_srt_file_content(intervals):
    srt_content = []
    for i, interval in enumerate(intervals, start=1):
        srt_content.append(f"{i}")
        srt_content.append(f"{interval}")
        srt_content.append(f"Segment {i}")
        srt_content.append("")  # Blank line after each subtitle
    return srt_content

def write_srt_file(match_srt, data_path, overwrite=False, suffix=""):
    for i, srt_content in enumerate(match_srt):
        output_file = data_path.joinpath(f"{i+1}_intervals{suffix}.srt")
        if not output_file.exists() or overwrite:
            with open(output_file, 'w') as file:
                file.write("\n".join(srt_content))

            print(f"SRT file '{output_file}' successfully created.")
        else:
            print(f"SRT file '{output_file}' already exists.")
            

def check_existing_srt_files(data_path: Path, suffix: str = "") -> bool:
    files = [f"{i}_intervals{suffix}.srt" for i in (1, 2)]
    return all((data_path / f).exists() for f in files)


@hydra.main(
    config_path="../configs/scripts",
    config_name="trim_summary"
)
def main(config: DictConfig):
    
    dataset = hydra.utils.instantiate(config.dataset)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device('cuda')  # Use the first CUDA device
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')  # Fallback to CPU
        print("CUDA is not available. Using CPU.")
    
    
    if config.checkpoint:
        checkpoint = torch.load(Path(config.checkpoint.directory).joinpath(config.checkpoint.name))
        
        
        model = vit_small().to(device)
        model.load_state_dict(checkpoint)
            
    with tqdm(total=len(dataset.game_list)) as pbar:
        for game in dataset.get_games_path():
            suffix = f"_{config.backbone}_TransNetv2_{int(round(float(config.transnet.threshold) * 100)):03d}" if config.transnet else f"_{config.backbone}"
            # Check if the file is already created
            if check_existing_srt_files(game, suffix) and not config.overwrite:
                print(f"{game} SRT files are already created. Skipping ...")
                continue
            
            # Load or compute feature representations
            try:
                features = (dataset.load_features(game, i, config.backbone) for i in range(3))
                summary, first_half, second_half = features
            except FileNotFoundError as e:
                print(e)
                
                transform = transforms.Compose([
                    transforms.ToTensor()  # Converts the image to a tensor with values in [0, 1]
                ])
                
                frames = (torch.stack(dataset.load_frames(game, i, transform)) for i in range(3))
                summary, first_half, second_half = frames
                
                ## Extract DINO features
                if not dataset.data_dir.joinpath(f"summary_{config.backbone}.npy").exists():
                    summary = torch.stack([model(summary[i].unsqueeze(0).to(device)).squeeze().detach().cpu() for i in range(len(summary))])
                    dataset.save_features(summary, game, 0, config.backbone) # Save features
                else:
                    summary = dataset.load_features(game, 0, config.backbone)
                
                if not dataset.data_dir.joinpath(f"1_{config.backbone}.npy").exists():
                    first_half = torch.stack([model(first_half[i].unsqueeze(0).to(device)).squeeze().detach().cpu() for i in range(len(first_half))])
                    dataset.save_features(first_half, game, 1, config.backbone) # Save features
                else:
                    first_half = dataset.load_features(game, 1, config.backbone)
                
                if not dataset.data_dir.joinpath(f"2_{config.backbone}.npy").exists():
                    second_half = torch.stack([model(second_half[i].unsqueeze(0).to(device)).squeeze().detach().cpu() for i in range(len(second_half))])
                    dataset.save_features(second_half, game, 2, config.backbone) # Save features
                else:
                    second_half = dataset.load_features(game, 2, config.backbone)
                
                
            # Trim summary
            if config.transnet:
                chunks = compute_transnet_boundaries(
                    dpath=os.path.join(game, 'summary_224p'),
                    config=config.transnet,
                    device=device
                )
            else:
                ## Compute nearest neighbors to localize video cuts
                nbrs = NearestNeighbors(n_neighbors=config.n_neighbors, algorithm='auto').fit(summary)
                _, indices = nbrs.kneighbors(summary)
                chunks = get_video_chunks(indices, config.n_neighbors)
            
            coincidences_first_half, coincidences_second_half = [], []
            for chunk in chunks:
                init_frame, end_frame = chunk[0], chunk[1]
                N = end_frame - init_frame
                
                first_sliding_window = [first_half[i:i+N] for i in range(len(first_half) - N + 1)]
                second_sliding_window = [second_half[i:i+N] for i in range(len(second_half) - N + 1)]
                segment, half = segment_matching((first_sliding_window, second_sliding_window), summary[init_frame:end_frame], N)
                
                if half == 1:
                    coincidences_first_half.append(segment)
                else:
                    coincidences_second_half.append(segment)
                    
                if config.display:
                    if half == 1:
                        display_segments(coincidences_first_half[-1], init_frame, game.joinpath("1_HQ_224p"), N)
                    else:
                        display_segments(coincidences_second_half[-1], init_frame, game.joinpath("2_HQ_224p"), N)
                        
            ## Compute time intervals
            first_half_intervals = compute_intervals(coincidences_first_half, config.intervals_thrsh, dataset.dataset_info['frame_rate'])
            second_half_intervals = compute_intervals(coincidences_second_half, config.intervals_thrsh, dataset.dataset_info['frame_rate'])
            
            ## Format intervals to time
            first_half_intervals, second_half_intervals = convert_time_intervals(first_half_intervals), convert_time_intervals(second_half_intervals)
            
            ## Convert to srt format
            first_half_intervals, second_half_intervals = generate_srt_file_content(first_half_intervals), generate_srt_file_content(second_half_intervals)
            
            # Write srt file
            write_srt_file([first_half_intervals, second_half_intervals], game, config.overwrite, suffix)
                
            pbar.update(1)

if __name__ == "__main__":
    main()