# FPV Saliency Maps

Saliency Maps from gaze fixation recordings for First Person View (FPV) drone racing flight trajectories recorded from 20 human pilots.

The animation below shows a single lap flight along the figure-eight "wave" track:

![](/media/anim.gif)

## Preparing the data for training

Before creating an index and generating ground-truth, individual laps should be filtered by whether they follow an "expected" trajectory (and thus are useful for learning to fly well):
```shell script
python src/data/generate_expected_trajectory_entries.py -tn flat/wave
```
This script will first create an image of all valid trajectories and then overlay each lap individually on top of it, with the user having to choose whether it follows an expected trajectory or not (buttons "Yes" or "No"). For now this has to be executed two times, once for each track name ("flat" and "wave").

To generate a global index of all frames including certain properties to filter by (e.g. `valid_lap`, `expected_trajectory`), run:
```shell script
python src/data/index_data.py
```

After that the "context" for the generation of ground-truth is established, and it an be generated for attention map prediction and control input prediction with the following two commands respectively:
```shell script
python src/data/generate_ground_truth.py -gtt moving_window_frame_mean_gt
```
```shell script
python src/data/generate_ground_truth.py -gtt drone_control_frame_mean_gt
```

To be able to use masked videos, one should first compute the mean mask(s).


## Experiment overview

20 human FPV pilots performing each a 2-hour experiment using drone racing simulator.

Each participant completed several 2-min flying sessions in the following order:

[1] ___Hairpin___: flying around vertical poles

[2] ___Slalom___: flying a slalom along vertical poles

[3] ___Split-S Large___: flying Split-S maneuvers through 4 gates with large vertical distance

[4] ___Split-S Small___: flying Split-S maneuvers through 4 gates with small vertical distance

[5] ___Flat___: Drone racing task on figure-eight "flat" track (5 sessions)

[6] ___Wave___: Drone racing task on figure-eight "wave" track (5 sessions)


Participants were instructed to fly as many laps as possible within the 2-min while avoiding crashing.

To be considered a valid lap, participants had to pass the gates in the correct order.



## File overview

The data is organized in the following strucutre: __data/SubjectNumber/SessionNumber_SessionName/__ 

Example: data/s004/11_wave/ contains data from Subject 4, Session 11, Drone racing on "wave" track


Each session folders contains the files:

- ___drone.csv___ : Drone state and control command data logged at 500 Hz

- ___events.csv___ : Gate-passing event timestamps [in sec]

- ___gaze_traces.csv___ : Camera positions and gaze projections in 3D, logged at 200 Hz
 
- ___laptimes.csv___ : Laptime summary

- ___track.csv___ : Gate positions and orientations



## Variable description

### drone.csv

__'ts'__ : Timestamp in sec

__'PositionX', 'PositionY', 'PositionZ'__ : 

Quadrotor position in meters in world frame (x=forward, y=left, z=up) 

__'RotationX', 'RotationY', 'RotationZ'__ : 

Quadrotor rotation euler angles in radians in world frame (use scipy .from_euler('xyz'))

__'rot_x_quat', 'rot_y_quat', 'rot_z_quat', 'rot_w_quat'__ : 

Quadrotor rotation quaternion in world frame (use scipy .from_quat())

__'VelocityX', 'VelocityY', 'VelocityZ'__ : 

Quadrotor velocity in meters per second in world frame (centered on the quadrotor)

__'AngularX', 'AngularY', 'AngularZ'__ : 

Quadrotor angular velocity in radians per second in body frame (x=forward, y=left, z=up)

__'GyroX', 'GyroY', 'GyroZ'__ : 

Gyroscope data

__'AccX', 'AccY', 'AccZ'__ : 

Accelerometer data

__'Throttle', 'Roll', 'Pitch', 'Yaw'__ : 

Pilot control commands for collective thrust (Throttle) and body rates (Roll, Pitch, Yaw), in the range of -1000 to 1000

__'TrackProgress'__ : 

Progress along the racing track in meters (based on shortest path)



### events.csv

__'ts'__ : 

Timestamp in sec

__'gate'__ : 

Gate identifier (0/10=start/finish gate, 1-9 intermediate gates)

__'lap'__ : 

Lap count (starts with 0)

__'is_valid'__ :

Whether the lap is considered valid (1) or invalid (0)



### gaze_traces.csv

__'ts'__ :

Timestamp of the gaze data logger in sec

__'norm_x_su, norm_y_su'__ :

Gaze fixation in screen coordinates [x=right, y=up, orgin at lower left corner]

__'ts_drone'__ :

Timestamp of the drone data logger closest to the current gaze timestamp in sec

__'cam_pos_x, cam_pos_y, cam_pos_z'__ :

Camera position in meters

__'cam_rot_x_quat, cam_rot_y_quat, cam_rot_z_quat, cam_rot_w_quat'__ :

Camera rotation quaternion in world frame (use scipy .from_quat())

__'ray_origin_pos_x, ray_origin_pos_y, ray_origin_pos_z'__ :

Origin of the gaze projection 3D ray in meters (identical with camera origin)

__'ray_endpoint_pos_x, ray_endpoint_pos_y, ray_endpoint_pos_z'__ :

Endpoint of the gaze projection 3D ray in meters assuming a length of 1000 m (in world coordinates)

__'norm_ray_x, norm_ray_y, norm_ray_z'__ :

Endpoint of the unit vector (length of 1 meter) of the gaze projection 3D ray, assuming a world origin, in world coordinates

__'intersect_OBJ_pos_x, intersect_OBJ_pos_y, intersect_OBJ_pos_z'__:

Coordinate of gaze 3d projection intersection with a surface object (where OBJ stands for gate0 - gate9, or floor) in meters in 3D World coordinates

__'intersect_OBJ_2d_x, intersect_gate0_2d_y'__ :

Coordinate of gaze 3d projection intersection with a surface object in 2D coordinates of the object

__'intersect_OBJ_distance'__ :

Length in meters of the gaze 3D projection from ray origin to ray intersection point with current object

__'num_intersections'__ :

Number of objects the current gaze 3D projection intersects with

__'close_intersect_name'__ :

Name of the intersection object that is closest to the origin

__'close_intersect_distance'__ :

Length in meters between gaze 3D projection origin and closest interaction object

__'close_intersect_pos_x, close_intersect_pos_y, close_intersect_pos_z'__ :

3D coordinates of the intersection point of the gaze 3D projection with the closest object



### laptimes.csv

__'lap'__ :

Lap count (starts with 0)

__'ts_start'__ :

Timestamp of lap start

__'ts_end'__ :

Timestamp of lap end

__'lap_time'__ :

Lap time in sec (lap_time = lap_end - lap_start)

__'is_valid'__ :

Whether the lap is considered valid (1) or invalid (0)

__'gate_id'__ : 

List of gate identifier for gate passing events (order indicates the order in which gates were passed)

__'gate_timestamps'__ :

List of timestamps for gate passing events (corresponding to the 'gate_id' identifiers)

__'num_events'__ :

Number of gate passing events (counting each gate passing event)

__'num_unique'__ : 

Number of unique gate passing events (counting only unique events, e.g. if gate 3 was passed multiple times, it will be counted only once)

__'num_collisions'__ :

Number of collisions of the quadrotor with a gate or the ground floor

__'collision_ts'__ : 

List of timestamps of the collsions



### track.csv

__'world'__ :

The environment where the race took place, always 'addition-arena' for all flights

__'track'__ :

The name of the track

__'pos_x', 'pos_y', 'pos_z'__ :

Gate position in meters (center of gate) in World frame 

__'rot_x_quat', 'rot_y_quat', 'rot_z_quat', 'rot_w_quat'__ :

Gate rotation quaternion in World frame (x=forward in the direction of flight, y=left, z=up)

__'rot_x_rad', 'rot_y_rad', 'rot_z_rad'__ :

Gate rotation euler angles in radians
       
__'rot_x_deg', 'rot_y_deg', 'rot_z_deg'__ :

Gate rotation euler angles in degrees

__'dim_x', 'dim_y', 'dim_z'__ :
       
Gate inner dimensions im meters

__'gate_id'__ :

Gate identifier (10=start/finish gate, 1-9 intermediate gates)



Author: cpfeiffe@ifi.uzh.ch