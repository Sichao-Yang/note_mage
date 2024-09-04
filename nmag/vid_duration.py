import subprocess
import os

def video_length_seconds(filename):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            "--",
            filename,
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout)
    except ValueError:
        raise ValueError(result.stderr.rstrip("\n"))

def format_duration(total_seconds):
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def calc_vid_duration(folder_path):
    total_duration = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(('.mp4', '.mkv', '.avi')):  # Add more extensions if needed
            file_path = os.path.join(folder_path, filename)
            total_duration += video_length_seconds(file_path)

    formatted_duration = format_duration(total_duration)
    print(f"Total Duration: {formatted_duration}")

if __name__ == "__main__":
    calc_vid_duration('./')