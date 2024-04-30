from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator


def process_video(input_video_path, output_video_path):
    # Read video frames
    video_frames = read_video(input_video_path)

    # Initialize tracker
    tracker = Tracker('Models/best.pt')

    # Track objects 
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=False,
                                       stub_path="stubs/newTrack.pkl"
                                       )
    print("Done wih tracking")
    tracker.add_position_to_tracks(tracks)
    # Estimate camera movement
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames)
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Transform view
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    # Estimate speed and distance
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
        
    team_assigner.add_team_tracker(tracks,video_frames)


     # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_possession= player_assigner.get_team_possesion(tracks)
    

    # Draw output
    output_frames = tracker.draw_annotations(video_frames, tracks,team_ball_possession)
    output_frames = camera_movement_estimator.draw_camera_movement(output_frames,camera_movement_per_frame)
    speed_and_distance_estimator.draw_speed_and_distance(output_frames, tracks)

    # Save video
    save_video(output_frames, output_video_path)


def main():
    input_video_path = 'input_videos/0a2d9b_9.mp4'
    output_video_path = 'output_videos/output_video1234.avi'
    process_video(input_video_path, output_video_path)


if __name__ == '__main__':
    main()
