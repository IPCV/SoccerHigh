import pysrt

def read_srt(path):
    # Load the SRT file
    subs = pysrt.open(path)
    # Return the subtitles
    return subs

def subs2segments(subs):
    # Create a list to save the information of each segment
    segments = []
    # For each segment get its information
    for sub in subs:
        # Get the idx from the subtitle text
        idx = int(sub.text.split(' ')[-1])
        segments.append({
            'start': sub.start,
            'end': sub.end,
            'idx': idx
        })
    # Sort the list using the idx key
    segments.sort(key=lambda x: x['idx'])
    # Return a list of dicts with the start and ending information of each segment
    return segments

def join_match_intervals(intervals):
    # Join first and second half segments into a single list
    return {half+1: intervals[half] for half in range(2)}

def timestamp2frameidx(timestamp, frame_rate=2):
    # Covert pysrt timestamp to frame indices
    return int(frame_rate * (timestamp.hours * 3600 + timestamp.minutes * 60 + timestamp.seconds + timestamp.milliseconds / 1000))