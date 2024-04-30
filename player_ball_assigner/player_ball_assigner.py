import sys 
import numpy as np
sys.path.append('../')
from utils import get_center_of_bbox,measure_distance

class PlayerBallAssigner:
    def __init__(self):
        self.maxplayer_ball_distance = 70
        
        
    def assign_ball_to_player(self,players,ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)
        
        minimum_distance = 9999999
        assigned_player = -1
        
        for player_id, player in players.items():
            player_bbox = player['bbox']
            
            distance_left = measure_distance((player_bbox[0],player_bbox[-1]),ball_position) 
            distance_right = measure_distance((player_bbox[2],player_bbox[-1]),ball_position) 
            
            distance = min(distance_left,distance_right)
            
            if distance < self.maxplayer_ball_distance:
                minimum_distance = distance
                assigned_player = player_id
                
                
        return assigned_player
    
    def get_team_possesion(self,tracks):
        team_ball_control = []
        for frame_num, player_track in enumerate(tracks['players']):
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            assigned_player = self.assign_ball_to_player(player_track, ball_bbox)

            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
            else:
                team_ball_control.append(team_ball_control[-1])
        team_ball_control= np.array(team_ball_control)
        
        return team_ball_control