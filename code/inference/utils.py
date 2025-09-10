import json
import numpy as np

from datetime import timedelta


# Convert numpy to native Python types
def convert_numpy_types(obj):
    """Recursively converts numpy types to native Python types."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):  # Handle tuple case as well
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.generic):  # Handles np.int64, np.float32, etc.
        return obj.item()  # Convert numpy type to native Python type (int or float)
    else:
        return obj

# Function to export the data to a JSON file
def save_to_json(data, filename):
    # Convert any numpy types to native Python types
    data = convert_numpy_types(data)

    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def frameidx2timestamp(frame_idx, frame_rate=2):
    """
    Convert frame index to timestamp in string format 'hh:mm:ss.mmm'.
    
    :param frame_idx: The frame index to convert.
    :param frame_rate: The frame rate (frames per second). Default is 2.
    :return: A string representing the timestamp in 'hh:mm:ss.mmm' format.
    """
    # Calculate total seconds from the frame index
    total_seconds = frame_idx / frame_rate
    
    # Convert total seconds to timedelta object
    timestamp = timedelta(seconds=total_seconds)
    
    # Extract hours, minutes, seconds, and milliseconds
    hours = timestamp.seconds // 3600
    minutes = (timestamp.seconds % 3600) // 60
    seconds = timestamp.seconds % 60
    milliseconds = int(timestamp.microseconds / 1000)
    
    # Format the time components into a string
    timestamp_str = f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"
    
    return timestamp_str