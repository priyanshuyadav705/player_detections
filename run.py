from src.detect_and_track import detect_and_track

video_path = "/Users/priyanshuyadav/Desktop/player_detection/video/15sec_input_720p.mp4"
model_path = "/Users/priyanshuyadav/Desktop/player_detection/model/best.pt"
output_path = "/Users/priyanshuyadav/Desktop/player_detection/output/tracked_output.mp4"

detect_and_track(video_path, model_path, output_path)
