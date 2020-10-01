import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tableone import TableOne
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
from scipy.stats import iqr
import seaborn as sn
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

try:
    from src.functions.laptracker_utils import *
    from src.functions.heatmap_utils import *
    from src.functions.preprocessing_utils import *
except:
    from functions.laptracker_utils import *
    from functions.heatmap_utils import *
    from functions.preprocessing_utils import *


def get_camera_pose(D, camera_pos=np.array([0.2, 0., 0.1]), camera_uptilt_deg=30):
    #quadrotor pose
    p_quad = D.loc[:, ('PositionX', 'PositionY', 'PositionZ')].values
    r_quad = D.loc[:, ('rot_x_quat','rot_y_quat','rot_z_quat','rot_w_quat')].values
    #camera rotation
    camera_rot_quat = Rotation.from_euler('y', -camera_uptilt_deg, degrees=True).as_quat()
    #camera pose
    p_cam = Rotation.from_quat(r_quad).apply(np.tile(camera_pos, (p_quad.shape[0], 1))) + p_quad
    r_cam = (Rotation.from_quat(r_quad) * Rotation.from_quat(np.tile(camera_rot_quat, (p_quad.shape[0], 1)))).as_quat()
    pose_cam = np.hstack((p_cam, r_cam))
    names_cam = ['cam_pos_x', 'cam_pos_y', 'cam_pos_z', 'cam_rot_x_quat', 'cam_rot_y_quat', 'cam_rot_z_quat',
                 'cam_rot_w_quat']
    for i in range(len(names_cam)):
        D[names_cam[i]] = pose_cam[:, i]
    return D

def distance_to_waypoint(p, w):
    return np.linalg.norm(p.reshape(-1,3) - w.reshape(-1,3), axis=1)

def compute_velocity_acceleration(t, p, method='smooth'):
    t = t.flatten()
    p = p.reshape(-1,3)
    if method=='windowed':
        v = np.empty(t.shape)
        v[:] = np.nan
        a = np.empty(t.shape)
        a[:] = np.nan
        win = (-0.5, 0.)
        for i in range(v.shape[0]):
            t_curr = t[i]
            ind = (t >= (t_curr + win[0])) & (t < (t_curr + win[1]))
            # print(t_curr, ind.shape, ind)
            if True in ind:
                p_curr = p[ind, :]
                dist = np.sum(np.linalg.norm(np.diff(p_curr, axis=0),axis=1))
                v[i] = dist / np.diff(win)
        a = np.diff(v, axis=0)
        a = np.vstack((a[0, :], a))
    elif method=='smooth':
        sr = 1000 #upsample frequency
        sg_samples = 501 #savgol filter width (needs to be odd)
        sg_order = 3 #savgol filter order
        #upsample and smooth position data
        tnew = np.arange(np.min(t), np.max(t) + 1/sr, 1/sr)
        pnew = np.empty((tnew.shape[0], 3))
        pnew[:] = np.nan
        for i in range(pnew.shape[1]):
            pnew[:, i] = np.interp(tnew.flatten(), t.flatten(), p[:, i].flatten())
            pnew[:, i] = savgol_filter(pnew[:, i].flatten(), sg_samples, sg_order)
        #compute velocity and acceleration
        vnew = np.diff(pnew, axis=0) / np.diff(tnew, axis=0).reshape(-1, 1)
        vnew = np.vstack((vnew[0, :], vnew))
        anew = np.diff(vnew, axis=0)
        anew = np.vstack((anew[0, :], anew))
        #downsample
        v = None
        a = None
        for i in range(vnew.shape[1]):
            vvals = np.interp(t, tnew, vnew[:, i])
            avals = np.interp(t, tnew, anew[:, i])
            if v is None:
                v = np.empty((vvals.shape[0], 3))
                v[:] = np.nan
            if a is None:
                a = np.empty((avals.shape[0], 3))
                a[:] = np.nan
            v[:, i] = vvals
            a[:, i] = avals

    return v, a

def plot_pose(P, R, ax=None, length=1., axis_order='xyz'):
  '''
  plot_pose(P,R,ax=None,l=1.)

  Makes a 3d matplotlib plot of poses showing the axes direction:
  x=red, y=green, z=blue

  P : np.ndarray, position [poses x axes]
  R : np.ndarray, rotation [poses x quaternions]
  '''
  # make a new figure if no axis was provided
  if ax is None:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    is_new_figure = True
  else:
    is_new_figure = False
  #check whether rotation data is quaternions or euler angles
  if R.shape[1] == 4:
      is_quaternion = True
  else:
      is_quaternion = False
  # make sure input data is 2D, where axes are in the 2nd dimension
  P = P.reshape((-1, 3))
  if is_quaternion:
    R = R.reshape((-1, 4))
  else:
    R = R.reshape((-1, 3))
  # loop over poses
  for i in range(P.shape[0]):
    # current position
    p0 = P[i, :]
    r = R[i, :]
    # loop over dimensions and plot the axes
    for dim, col in zip(np.arange(0, 3, 1), ['r', 'g', 'b']):
      u = np.zeros((1, 3))
      u[0, dim] = 1.
      if is_quaternion:
        v = Rotation.from_quat(r).apply(u)[0]
      else:
        v = Rotation.from_euler(axis_order, r, degrees=False).apply(u)[0]
      p1 = p0 + length * v
      ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=col)
  #figure formatting in case a new figure was created
  if is_new_figure:
      width_val = np.max(np.nanmax(P, axis=0) - np.nanmin(P, axis=0)) / 2.
      mean_val = np.nanmean(P, axis=0)
      ax.set_xlim((mean_val[0] - width_val, mean_val[0] + width_val))
      ax.set_ylim((mean_val[1] - width_val, mean_val[1] + width_val))
      ax.set_zlim((mean_val[2] - width_val, mean_val[2] + width_val))
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')

def plot_vector(P, V, ax=None, scaling=1., col='m'):
  '''
  plot_pose(P,R,ax=None,l=1.)

  Makes a 3d matplotlib plot with vector pointing from the position:
  x=red, y=green, z=blue

  P : np.ndarray, position [poses x axes]
  V : np.ndarray, vector [poses x axes]
  '''
  # make a new figure if no axis was provided
  if ax is None:
      fig = plt.figure(figsize=(10, 10))
      ax = fig.add_subplot(projection='3d')
      is_new_figure = True
  else:
      is_new_figure = False
  # make sure input data is 2D, where axes are in the 2nd dimension
  P = P.reshape((-1, 3))
  V = V.reshape((-1, 3))
  # loop over poses
  for i in range(P.shape[0]):
    # current position
    p = P[i, :]
    # current vector
    v = V[i, :] * scaling
    ax.plot([p[0], p[0]+v[0]], [p[1], p[1]+v[1]], [p[2], p[2]+v[2]], color=col)
  #figure formatting in case a new figure was created
  if is_new_figure:
      width_val = np.max(np.nanmax(P, axis=0) - np.nanmin(P, axis=0)) / 2.
      mean_val = np.nanmean(P, axis=0)
      ax.set_xlim((mean_val[0] - width_val, mean_val[0] + width_val))
      ax.set_ylim((mean_val[1] - width_val, mean_val[1] + width_val))
      ax.set_zlim((mean_val[2] - width_val, mean_val[2] + width_val))
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')

def get_norm_trajectories(D, t0, t1, gate_id, gate_ts, win=None, res=None):
    if isinstance(gate_id, str):
        gate_id = np.fromstring(gate_id.strip('[]'), dtype=int, sep=' ')
    if isinstance(gate_ts, str):
        gate_ts = np.fromstring(gate_ts.strip('[]'), dtype=float, sep=' ')
    #pointer to current lap
    ind = (D.ts.values >= t0) & (D.ts.values <= t1)
    #if too few data was found return
    if np.sum(ind)<2:
        return None
    #else proceed
    else:
        #select data for current lap
        D = D.loc[ind, :]
        #extract drone state variables from drone data
        time = D.loc[:, ('ts')].values
        time = time - t0
        progress = D.loc[:, ('TrackProgress')].values
        progress = progress - progress[0]
        position = D.loc[:, ('PositionX', 'PositionY', 'PositionZ')].values
        reference = D.loc[:, ('ProjectionShortestPathX', 'ProjectionShortestPathY', 'ProjectionShortestPathZ')].values
        deviation_from_reference = position - reference
        norm_deviation_from_reference = np.linalg.norm(deviation_from_reference, axis=1)
        velocity = D.loc[:, ('VelocityX', 'VelocityY', 'VelocityZ')].values
        norm_velocity = np.linalg.norm(velocity, axis=1)
        acceleration = D.loc[:, ('DroneAccelerationX', 'DroneAccelerationY', 'DroneAccelerationZ')].values
        norm_acceleration = np.linalg.norm(acceleration, axis=1)
        angularvelocity = D.loc[:, ('AngularX', 'AngularY', 'AngularZ')].values
        norm_angularvelocity = np.linalg.norm(angularvelocity, axis=1)
        bodyrates = D.loc[:, ('Roll', 'Pitch', 'Yaw')].values
        throttle = D.loc[:, ('Throttle')].values
        gate_ts = gate_ts - t0
        gate_progress = np.empty(gate_ts.shape)
        gate_progress[:] = np.nan
        for i in range(gate_id.shape[0]):
            vals = progress[time >= gate_ts[i]]
            if vals.shape[0] > 1:
                vals = vals[0]
            elif vals.shape[0] == 0:
                vals = np.nan
            gate_progress[i] = vals
        #now extract variables (medians) for different track progress
        time2, progress2 = normalize_by_progress(time, progress, win, res)
        position2, _ = normalize_by_progress(position, progress, win, res)
        deviation_from_reference2, _ = normalize_by_progress(deviation_from_reference, progress, win, res)
        norm_deviation_from_reference2, _ = normalize_by_progress(norm_deviation_from_reference, progress, win, res)
        velocity2, _ = normalize_by_progress(velocity, progress, win, res)
        norm_velocity2, _ = normalize_by_progress(norm_velocity, progress, win, res)
        acceleration2, _ = normalize_by_progress(acceleration, progress, win, res)
        norm_acceleration2, _ = normalize_by_progress(norm_acceleration, progress, win, res)
        angularvelocity2, _ = normalize_by_progress(angularvelocity, progress, win, res)
        norm_angularvelocity2, _ = normalize_by_progress(norm_angularvelocity, progress, win, res)
        bodyrates2, _ = normalize_by_progress(bodyrates, progress, win, res)
        throttle2, _ = normalize_by_progress(throttle, progress, win, res)
        gates2 = np.empty(time2.shape)
        gates2[:] = np.nan
        for i in range(gate_id.shape[0]):
            idx = np.argmin(np.abs(progress2 - gate_progress[i]))
            gates2[idx] = gate_id[i]

        #make output dataframe
        df = pd.DataFrame(
            {'progress' : progress2.flatten(), 'time' : time2.flatten(),
             'position_x' : position2[:, 0], 'position_y' : position2[:, 1], 'position_z' : position2[:, 2],
             'deviation_shortest_path_x': deviation_from_reference2[:, 0], 'deviation_shortest_path_y': deviation_from_reference2[:, 1], 'deviation_shortest_path_z': deviation_from_reference2[:, 2],
             'norm_deviation_shortest_path': norm_deviation_from_reference2.flatten(),
             'velocity_x': velocity2[:, 0], 'velocity_y': velocity2[:, 1], 'velocity_z': velocity2[:, 2],
             'norm_velocity': norm_velocity2.flatten(),
             'acceleration_x': acceleration2[:, 0], 'acceleration_y': acceleration2[:, 1], 'acceleration_z': acceleration2[:, 2],
             'norm_acceleration': norm_acceleration2.flatten(),
             'angularvelocity_x': angularvelocity2[:, 0], 'angularvelocity_y': angularvelocity2[:, 1], 'angularvelocity_z': angularvelocity2[:, 2],
             'norm_angularvelocity': norm_angularvelocity2.flatten(),
             'throttle': throttle2.flatten(),
             'roll': bodyrates2[:, 0], 'pitch': bodyrates2[:, 1], 'yaw': bodyrates2[:, 2],
             'gates': gates2.flatten()}, index=np.arange(0, time2.shape[0], 1))
        return df

def normalize_by_progress(y, x, win=None, res=None):
    if win is None:
        win = (np.nanmin(x), np.nanmax(x))
    if res is None:
        res = np.nanmedian(np.diff(np.sort(x)))
    x2 = np.arange(win[0], win[1] + res, res)
    if len(y.shape) == 1:
        y = y.reshape((-1, 1))
    y2 = np.empty((x2.shape[0], y.shape[1]))
    for i in range(y.shape[1]):
        y2[:, i] = np.interp(x2, x, y[:, i].flatten())
    #set boundary data to nan
    ind = (x2 < np.nanmin(x)) | (x2 > np.nanmax(x))
    y2[ind, :] = np.nan
    return y2, x2

def compute_flight_paths_subject(PATH, win=(0, 200), res=0.001, tracks=['flat', 'wave'], exclude_recordings=None,
                                 max_lap_start_time=120., min_lap_unique_events=3):
    # get subfolder names
    folders = []
    for walker in os.walk(PATH):
        if walker[0] == PATH:
            vals = walker[1]
    # loop over folder names and check for in/exclusion
    for val in vals:
        if val.find('_') != -1:
            # by default consider the current folder valid
            is_valid = True
            # get current recording number and track name
            curr_recording = int(val.split('_')[0])
            curr_track = val.split('_')[1]
            # check for exclusion criteria
            if exclude_recordings is not None:
                if curr_recording in exclude_recordings:
                    is_valid = False
            if curr_track not in tracks:
                is_valid = False
            # eventually append or not the current folder to the output list
            if is_valid:
                folders.append(val)
    # sort folder names
    folders = sorted(folders)
    # only proceed if some valid folders were found
    if len(folders) > 0:
        # set the track progress window according to parameters
        x = np.arange(win[0], win[1], res)  # spatial resolution in meters [here: 1 mm]
        ## Step 1. Get flight paths for valid laps
        for folder in folders:
            # load laptime data
            L = pd.read_csv(PATH + folder + '/laptimes.csv')
            # select laps with starttime less than criterion
            if max_lap_start_time is not None:
                ind = (L.ts_start.values - L.ts_start.min()) < max_lap_start_time
                L = L.loc[ind, :]
            # select laps with minimum number of events
            if min_lap_unique_events is not None:
                ind = (L.num_unique > min_lap_unique_events)
                L = L.loc[ind, :]
            # select valid laps
            ind = (L.is_valid == 1)
            L = L.loc[ind, :]
            # continue only if some laps remain after selection
            if L.shape[0] > 0:
                # load dronedata
                D = pd.read_csv(PATH + folder + '/drone.csv')
                # loop over laps
                Y = np.array([])
                for i in range(L.shape[0]):
                    # pointer to raw data from the current lap
                    ind = (D.ts >= L.ts_start.iloc[i]) & (D.ts < L.ts_end.iloc[i])
                    # get the raw track progress for the current lap
                    x_raw = D.loc[ind, ('TrackProgress')].values
                    # for all except first lap, set the starting point to zero
                    # first lap started on podium, thus not at zero gate
                    if L.lap.iloc[i] > 0:
                        x_raw = x_raw - x_raw[0]
                    p_raw = D.loc[ind, ('PositionX', 'PositionY', 'PositionZ')].values
                    # bin and upsample the data by linear interpolation
                    p = np.empty((x.shape[0], p_raw.shape[1]))
                    p[:] = np.nan
                    for j in range(p_raw.shape[1]):
                        p[:, j] = np.interp(x, x_raw, p_raw[:, j].flatten())
                    # remove all interpolate values out of the current raw data range
                    p[(x < x_raw[0]), :] = np.nan
                    p[(x > x_raw[-1]), :] = np.nan
                    # add data to output vector
                    if Y.size:
                        Y = np.append(Y, p.reshape((p.shape[0], p.shape[1], 1)), axis=2)
                    else:
                        Y = p.reshape((p.shape[0], p.shape[1], 1))
                # save flight path data from current run, only if some data available
                if Y.size:
                    np.save(PATH + folder + '/flight_path_track_progression.npy', x)
                    L.to_csv(PATH + folder + '/flight_path_info.csv', index=False)
                    np.save(PATH + folder + '/flight_path_run.npy', Y)
                    np.save(PATH + folder + '/flight_path_run_mean.npy', np.nanmean(Y, axis=2))
                    np.save(PATH + folder + '/flight_path_run_median.npy', np.nanmedian(Y, axis=2))
        ## Step 2. Compute average and median flight path per subject
        for cond in tracks:
            # collect all flightpaths for the current condition
            F = np.array([])
            for folder in [name for name in folders if (name.find(cond) != -1)]:
                # check if flight path position data is available
                if os.path.isfile(PATH + folder + '/flight_path_run.npy'):
                    # load flight path position data
                    curr_fp = np.load(PATH + folder + '/flight_path_run.npy')
                    # append to the total array
                    if F.size:
                        F = np.append(F, curr_fp, axis=2)
                    else:
                        F = curr_fp
            # save average and median flight path for the current condition
            for folder in [name for name in folders if (name.find(cond) != -1)]:
                if os.path.isfile(PATH + folder + '/flight_path_track_progression.npy') == False:
                    np.save(PATH + folder + '/flight_path_track_progression.npy', x)
                np.save(PATH + folder + '/flight_path_subj_mean.npy', np.nanmean(F, axis=2))
                np.save(PATH + folder + '/flight_path_subj_median.npy', np.nanmedian(F, axis=2))

def compute_flight_performance_subject(PATH):
    for walker in os.walk(PATH):
        if ('laptimes.csv' in walker[2]) and ('flight_path_subj_mean.npy' in walker[2]):
            runpath = walker[0] + '/'
            # load data from current run (all laps)
            T = pd.read_csv(runpath + 'track.csv')
            D = pd.read_csv(runpath + 'drone.csv')
            L = pd.read_csv(runpath + 'laptimes.csv')
            x_avg = np.load(runpath + 'flight_path_track_progression.npy')
            y_avg = np.load(runpath + 'flight_path_subj_mean.npy')
            #Velocity
            val_mean = []
            val_std = []
            val_max = []
            for i in range(L.shape[0]):
                ind = (D.ts >= L.ts_start.iloc[i]) & (D.ts < L.ts_end.iloc[i])
                val = D.loc[ind, ('VelocityX', 'VelocityY', 'VelocityZ')].values
                val = np.linalg.norm(val, axis=1)
                val_mean.append(np.nanmean(val))
                val_std.append(np.nanstd(val))
                val_max.append(np.nanmax(val))
            L['Velocity_Mean [m/s]'] = val_mean
            L['Velocity_Std [m/s]'] = val_std
            L['Velocity_Max [m/s]'] = val_max
            #Path Length
            #shortest path (first lap, other laps)
            l0, l1 = compute_shortest_path_length(T, False)
            #track progress for first lap, starting from podium
            track_progress_podium = 2.499885
            val_flight = []
            val_shortest = []
            val_subj_mean = []
            for i in range(L.shape[0]):
                ind = (D.ts >= L.ts_start.iloc[i]) & (D.ts <= L.ts_end.iloc[i])
                val = D.loc[ind, ('PositionX', 'PositionY', 'PositionZ')].values
                val_flight.append(compute_path_length(val))
                if i==0:
                    val_shortest.append(l0)
                    val_subj_mean.append(compute_path_length(y_avg, x_avg, track_progress_podium))
                else:
                    val_shortest.append(l1)
                    val_subj_mean.append(compute_path_length(y_avg))
            L['Path_Length [m]'] = val_flight
            L['Path_Length_Shortest [m]'] = val_shortest
            L['Path_Length_Subj_Mean [m]'] = val_subj_mean
            L['Path_Length_Deviation_From_Shortest [%]'] = 100 * ((L['Path_Length [m]'] / L['Path_Length_Shortest [m]']) - 1.)
            L['Path_Length_Deviation_From_Subj_Mean [%]'] = 100 * ((L['Path_Length [m]'] / L['Path_Length_Subj_Mean [m]']) - 1.)
            #Path Deviation
            val_mean = []
            val_std = []
            val_max = []
            for i in range(L.shape[0]):
                ind = (D.ts >= L.ts_start.iloc[i]) & (D.ts <= L.ts_end.iloc[i])
                curr_x = D.loc[ind, ('TrackProgress')].values
                if i > 0:
                    curr_x = curr_x - curr_x[0]
                curr_y = D.loc[ind, ('PositionX', 'PositionY', 'PositionZ')].values
                curr_y_avg = np.empty(curr_y.shape)
                curr_y_avg[:] = np.nan
                for j in range(curr_y_avg.shape[1]):
                    curr_y_avg[:, j] = np.interp(curr_x, x_avg, y_avg[:, j])
                val = curr_y - curr_y_avg
                val = np.linalg.norm(val, axis=1)
                val_mean.append(np.nanmean(val, axis=0))
                val_std.append(np.nanstd(val, axis=0))
                val_max.append(np.nanmax(val, axis=0))
            L['Path_Deviation_Mean [m]'] = val_mean
            L['Path_Deviation_Std [m]'] = val_std
            L['Path_Deviation_Max [m]'] = val_max
            #Tilt Angle (between body and world z-axis)
            val_mean = []
            val_std = []
            val_max = []
            for i in range(L.shape[0]):
                ind = (D.ts >= L.ts_start.iloc[i]) & (D.ts < L.ts_end.iloc[i])
                r = D.loc[ind, ('rot_x_quat', 'rot_y_quat', 'rot_z_quat', 'rot_w_quat')].values
                z = np.empty((r.shape[0],3))
                z[:] = 0.
                z[:, 2] = 1.
                z = z / np.linalg.norm(z, axis=1).reshape((-1, 1))
                q = Rotation.from_quat(r).apply(z)
                q = q / np.linalg.norm(q, axis=1).reshape((-1, 1))
                ang = []
                for j in range(q.shape[0]):
                    ang.append(np.arccos(np.clip(np.dot(q[j, :], z[j, :]), -1.0, 1.0)))
                ang = np.array(ang)
                val_mean.append(np.nanmean(ang))
                val_std.append(np.nanstd(ang))
                val_max.append(np.nanmax(ang))
            L['Z_Tilt_Angle_Mean [rad]'] = val_mean
            L['Z_Tilt_Angle_Std [rad]'] = val_std
            L['Z_Tilt_Angle_Max [rad]'] = val_max
            #save the flight performance features
            print('..saving {}'.format(runpath + 'flight_performance.csv'))
            L.to_csv(runpath + 'flight_performance.csv', index=False)

def plot_flight_paths_subject(PATH, track):
    folders = []
    for walker in os.walk(PATH):
        if walker[0] == PATH:
            folders = walker[1]
    folders = sorted(folders)
    if len(folders) > 0:
        P = np.array([])  # progression
        A = np.array([])  # position all laps
        R = np.array([])  # position run averages
        S = np.array([])  # position subject average
        for folder in [name for name in folders if (name.find(track) != -1)]:
            # check if flight path position data is available
            if os.path.isfile(PATH + folder + '/flight_path_run.npy'):
                # load flightpath for all laps
                a = np.load(PATH + folder + '/flight_path_run.npy')
                if A.size:
                    A = np.append(A, a, axis=2)
                else:
                    A = a
                # load flight path average for runs
                r = np.load(PATH + folder + '/flight_path_run_median.npy')
                if R.size:
                    R = np.append(R, r.reshape((r.shape[0], r.shape[1], 1)), axis=2)
                else:
                    R = r.reshape((r.shape[0], r.shape[1], 1))
                # load flight path average across subject
                if S.shape[0] == 0:
                    S = np.load(PATH + folder + '/flight_path_subj_median.npy')
                # load track progression
                if P.shape[0] == 0:
                    P = np.load(PATH + folder + '/flight_path_track_progression.npy')
        # plot the flight paths
        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, projection='3d')
        # plot valid flight paths
        for i in range(A.shape[2]):
            ax.plot(A[:, 0, i], A[:, 1, i], A[:, 2, i], 'k-', lw=1)
        # plot run averages
        for i in range(R.shape[2]):
            ax.plot(R[:, 0, i], R[:, 1, i], R[:, 2, i], 'b-', lw=1)
        # plot subject average
        ax.plot(S[:, 0], S[:, 1], S[:, 2], 'r-', lw=1)
        ax.set_xlim((-20, 20))
        ax.set_ylim((-20, 20))
        ax.set_zlim((-20, 20))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

def plot_flight_paths_group(PATH, subjs, track):
    P = np.array([])  # progression
    A = np.array([])  # position all laps
    R = np.array([])  # position run averages
    S = np.array([])  # position subject averages
    G = np.array([])  # position subject grand average
    for subj in subjs:
        subjpath = PATH + 's%03d/' % subj
        has_subj_avg = False
        folders = []
        for walker in os.walk(subjpath):
            if walker[0] == subjpath:
                folders = walker[1]
        folders = sorted(folders)
        if len(folders) > 0:
            for folder in [name for name in folders if (name.find(track) != -1)]:
                # print('..loading data: {}'.format(subjpath + folder + '/'))
                # check if flight path position data is available
                if os.path.isfile(subjpath + folder + '/flight_path_run.npy'):
                    # # load flightpath for all laps
                    # a = np.load(subjpath + folder + '/flight_path_run.npy')
                    # if A.size:
                    #     A = np.append(A, a, axis=2)
                    # else:
                    #     A = a
                    # # load flight path average for runs note
                    # r = np.load(subjpath + folder + '/flight_path_run_median.npy')
                    # if R.size:
                    #     R = np.append(R, r.reshape((r.shape[0], r.shape[1], 1)), axis=2)
                    # else:
                    #     R = r.reshape((r.shape[0], r.shape[1], 1))
                    # load flight path average across subject
                    if (has_subj_avg == False):
                        s = np.load(subjpath + folder + '/flight_path_subj_median.npy')
                        if S.size:
                            S = np.append(S, s.reshape((s.shape[0], s.shape[1], 1)), axis=2)
                        else:
                            S = s.reshape((s.shape[0], s.shape[1], 1))
                        has_subj_avg = True
                    # load track progression
                    if P.shape[0] == 0:
                        P = np.load(subjpath + folder + '/flight_path_track_progression.npy')
    #make grand average
    G = np.nanmean(S, axis=2)
    # plot the flight paths
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='3d')
    # # plot valid flight paths
    # for i in range(A.shape[2]):
    #     ax.plot(A[:, 0, i], A[:, 1, i], A[:, 2, i], 'k-', lw=1)
    # # plot run averages
    # for i in range(R.shape[2]):
    #     ax.plot(R[:, 0, i], R[:, 1, i], R[:, 2, i], 'g-', lw=1)
    # plot subject flight path
    for i in range(S.shape[2]):
        ax.plot(S[:, 0, i], S[:, 1, i], S[:, 2, i], 'b-', lw=1)
    # plot grand average flight path
    ax.plot(G[:, 0], G[:, 1], G[:, 2], 'r-', lw=1)
    ax.set_xlim((-20, 20))
    ax.set_ylim((-20, 20))
    ax.set_zlim((-20, 20))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def collect_logfiles_group(PATH, subjs):
    '''
    ### Edinburgh Handedness Inventory - Short Form
    __Coding:__
    Immer Rechts = 100
    Meistens Rechts = 50
    Beide Gleichviel = 0
    Meistens Links = -50
    Immer Links = -100
    __Laterality Quotient:__
    Mean across scores to the four questions on [1. writing, 2. throwing, 3. toothbrush, 4. spoon]
    Right handers: 61 to 100
    Mixed handers: -60 to 60
    Left handers: -100 to -61
    '''
    rootpath = PATH.split('process')[0]
    #group dataframe
    E = pd.DataFrame()
    #loop over subjects
    for subj in subjs:
        #path to data
        subjpath = rootpath + 'raw/s%03d/' % subj
        # import raw data
        if (subj == 4) or (subj == 5):
            curr_E = pd.read_csv(subjpath + 'Eingangsfragebogen.csv')
        else:
            curr_E = pd.read_csv(subjpath + 'FB_Eingangsfragebogen.csv')
        # add subject number
        curr_E = curr_E.drop(columns=['Probandennummer'])
        curr_E['Subject'] = subj
        # timestamp experiment day
        curr_E['Timestamp'] = pd.Timestamp(curr_E['Timestamp'].iloc[0].split('GMT')[0], tz='Europe/Paris')
        # Age in years
        ymd = [val for val in reversed([int(val) for val in curr_E['Geburtsdatum'].iloc[0].split('.')])]
        bday = pd.Timestamp(ymd[0], ymd[1], ymd[2]).tz_localize('Europe/Paris')
        curr_E['Age [Years]'] = (curr_E['Timestamp'] - bday).iloc[0].days / 365
        # Sex
        curr_E['Sex [Male]'] = (curr_E['Geschlecht'] == 'Männlich').astype(int)
        # Handedness: Laterality quotient
        ehi_dict = {'Immer Rechts': 100, 'Meistens Rechts': 50, 'Beide gleich viel': 0,
                    'Meistens Links': -50, 'Immer Links': -100}
        vals = []
        for n in ['Schreiben', 'Werfen', 'Zahnbürste', 'Löffel']:
            name = [val for val in curr_E.columns if (val.find(n) != -1)][0]
            vals.append(ehi_dict[curr_E[name].iloc[0]])
        curr_E['Laterality_Quotient [-100,100]'] = np.mean(np.array(vals))
        # Dominat eye
        curr_E['Dominant_Eye [Right]'] = (curr_E['Dominantes Auge'] == 'Rechts').astype(int)
        # FPV Age
        ymd = [val for val in
               reversed([int(val) for val in curr_E['Wann mit FPV begonnen [Datum]?'].iloc[0].split('.')])]
        fday = pd.Timestamp(ymd[0], ymd[1], ymd[2]).tz_localize('Europe/Paris')
        curr_E['FPV_Age [Years]'] = (curr_E['Timestamp'] - fday).iloc[0].days / 365
        # Flight time
        curr_E['Total_Flight_Time [Hours]'] = curr_E['Wieviele Stunden FPV insgesamt? [Stunden]']
        curr_E['Recent_Flight_Time [Hours/Week]'] = curr_E[
            'Wieviele Stunden FPV pro Woche in den letzten 3 Monaten? [Stunden/Woche]']
        # Proportion Disciplines
        rfa = [float(val.strip(' %')) for val in
               curr_E['Verteilung der FPV Stunden nach Disziplin: Race, Freestyle, Andere? [%] '].iloc[0].split(',')]
        rfa = np.array(rfa)
        rfa = rfa / np.sum(rfa)
        curr_E['Prop_Race'] = rfa[0]
        curr_E['Prop_Freestyle'] = rfa[1]
        curr_E['Prop_Other'] = rfa[2]
        # Number of drone races
        curr_E['Num_Drone_Races'] = curr_E['An wieviel Drohnenrennen teilgenommen?']
        # Rank at drone races
        curr_E['Rank_Drone_Races [1,100]'] = curr_E[
            'Drohnenrennen: Durchschnittlicher Rank über die letzten 10 Rennen? [1=Beste]']
        # Gaming
        curr_E['Gaming_Time [Hours/Week]'] = curr_E[
            'Wieviel Stunden Computerspiele pro Woche in den letzten 3 Monaten? [Stunden/Woche]']
        # sort the columns to a convenient format
        idx = np.argmax((curr_E.columns == 'Subject').astype(int))
        names = [curr_E.columns[0]]
        names.extend(curr_E.columns[np.arange(idx, curr_E.shape[1], 1)])
        names.extend(curr_E.columns[np.arange(1, idx, 1)])
        curr_E = curr_E.loc[:, names]
        # append current data to the group table
        if E.size:
            E = E.append(curr_E)
        else:
            E = curr_E
    # reset indices
    E = E.reset_index().drop(columns=['index'])
    # save the table
    outpath = rootpath + 'process/group/'
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    print('..saving {}'.format(outpath + 'subject_logfiles.csv'))
    E.to_csv(outpath + 'subject_logfiles.csv', index=False)

def collect_ratings_group(PATH, subjs):
    '''
    ### FPV Questionnaire (9-point scale)
    1. Wie bewerten Sie Ihre Leistung insgesamt? [Sehr schwach, Sehr stark]
    2. Wie bewerten Sie Ihre Geschwindigkeit? [Sehr niedrig, Sehr hoch]
    3. Wie bewerten Sie Ihre Genauigkeit? [Sehr gering, Sehr hoch]
    4. Wie bewerten Sie Ihre Konzentration? [Sehr niedrig, Sehr hoch]
    5. Wie stark war Ihr Gefühl von Kontrolle über die Bewegungen der Drohne? [Sehr schwach, Sehr stark]
    ### NASA TLX German Verion (9-point scale)
    6. Wie viel geistige Anforderung war bei der Aufgabe erforderlich? [Sehr gering, Sehr hoch]
    7. Wie viel körperliche Anforderung war bei der Aufgabe erforderlich? [Sehr gering, Sehr hoch]
    8. Wie viel Zeitdruck empfanden Sie während der Aufgabe? [Sehr gering, Sehr hoch]
    9. Wie zufrieden waren Sie mit Ihrer Leistung im Zusammenhang mit der Aufgabe? [Gar nicht, Sehr]
    (Note: for Question 9, the scale was presented to subjects as [Sehr, Gar Nicht], and is flipped here in the function)
    10. Insgesamt betrachtet: Wie gross war die von Ihnen empfundene Anstrengung bei der Aufgabe? [Sehr gering, Sehr hoch]
    11. Wie frustriert fühlten Sie sich während der Aufgabe? [Sehr gering, Sehr hoch]
    '''
    rootpath = PATH.split('process')[0]
    S = pd.DataFrame()
    #loop over subjects
    for subj in subjs:
        #set subject path to data
        subjpath = rootpath + 'raw/s%03d/' % subj
        # import raw data
        if (subj == 4) or (subj == 5):
            df1 = pd.read_csv(subjpath + 'Selbsteinschätzung.csv')
            df2 = pd.read_csv(subjpath + 'Task Load Index.csv')
            curr_S = df1.merge(df2, how='outer', left_on='PN', right_on='PN')
            curr_S = curr_S.drop(columns=[n for n in curr_S.columns if (n.find('Unnamed') != -1)])
            curr_S = curr_S.drop(columns=['Timestamp_y']).rename(columns={'Timestamp_x': 'Timestamp'})
        else:
            curr_S = pd.read_csv(subjpath + 'FB_v2_Selbsteinschätzung.csv')
        # fix some column names
        corr_dict = {
            'Wie bewerten Sie Ihre Leistung insgesamt:': 'Wie bewerten Sie Ihre Leistung insgesamt?',
            'Wie bewerten Sie ihre Geschwindigkeit?': 'Wie bewerten Sie Ihre Geschwindigkeit?',
            'Wie bewerten Sie ihre Genauigkeit?': 'Wie bewerten Sie Ihre Genauigkeit?'
        }
        for key, value in corr_dict.items():
            if key in curr_S.columns:
                curr_S = curr_S.rename(columns={key: value})
        # invert the scale on TLX question 9:
        curr_S['Wie zufrieden waren Sie mit Ihrer Leistung im Zusammenhang mit der Aufgabe?'] = 10 - curr_S[
            'Wie zufrieden waren Sie mit Ihrer Leistung im Zusammenhang mit der Aufgabe?']
        # add question labels
        label_dict = {
            'Wie bewerten Sie Ihre Leistung insgesamt?': '[Sehr Schwach, Sehr Stark]',
            'Wie bewerten Sie Ihre Geschwindigkeit?': '[Sehr Niedrig, Sehr Hoch]',
            'Wie bewerten Sie Ihre Genauigkeit?': '[Sehr Gering, Sehr Hoch]',
            'Wie bewerten Sie Ihre Konzentration?': ' [Sehr Niedrig, Sehr Hoch]',
            'Wie stark war Ihr Gefühl von Kontrolle über die Bewegungen der Drohne?': '[Sehr Schwach, Sehr Stark]',
            'Wie viel geistige Anforderung war bei der Aufgabe erforderlich?': '[Sehr Gering, Sehr Hoch]',
            'Wie viel körperliche Anforderung war bei der Aufgabe erforderlich?': '[Sehr Gering, Sehr Hoch]',
            'Wie viel Zeitdruck empfanden Sie während der Aufgabe?': '[Sehr Gering, Sehr Hoch]',
            'Wie zufrieden waren Sie mit Ihrer Leistung im Zusammenhang mit der Aufgabe?': '[Gar nicht, Sehr]',
            'Insgesamt betrachtet: Wie gross war die von Ihnen empfundene Anstrengung bei der Aufgabe?': '[Sehr gering, Sehr hoch]',
            'Wie frustriert fühlten Sie sich während der Aufgabe?': '[Sehr gering, Sehr hoch]'
        }
        new_colnames = []
        for name in curr_S.columns:
            if name in list(label_dict.keys()):
                curr_S = curr_S.rename(columns={name: name + ' ' + label_dict[name]})
        # subject number
        curr_S['Subject'] = subj
        # flight track
        ind = curr_S['PN'].str.contains('w')
        curr_S['Track'] = 'flat'
        curr_S.loc[ind, ('Track')] = 'wave'
        # run number
        is_flat = True
        count = 0
        run = []
        for i in range(curr_S.shape[0]):
            if (curr_S['Track'].iloc[i] == 'flat') and (is_flat == False):
                count = 0
                is_flat = True
            if (curr_S['Track'].iloc[i] == 'wave') and (is_flat == True):
                count = 0
                is_flat = False
            count += 1
            run.append(count)
        curr_S['Run'] = run
        # sort the columns to a convenient format
        idx = np.argmax((curr_S.columns == 'Subject').astype(int))
        names = [curr_S.columns[0]]
        names.extend(curr_S.columns[np.arange(idx, curr_S.shape[1], 1)])
        names.extend(curr_S.columns[np.arange(1, idx, 1)])
        curr_S = curr_S.loc[:, names]
        # convert ratings to 0-100 range
        idx = np.argmax(
            (curr_S.columns == 'Wie bewerten Sie Ihre Leistung insgesamt? [Sehr Schwach, Sehr Stark]').astype(int))
        for i in range(idx, curr_S.shape[1]):
            curr_S.loc[:, curr_S.columns[i]] = 100 * ((curr_S.loc[:, curr_S.columns[i]] - 1) / 8)
        # append current data to the group table
        if S.size:
            S = S.append(curr_S)
        else:
            S = curr_S
    # reset indices
    S = S.reset_index().drop(columns=['index'])
    # save the tables
    outpath = rootpath + 'process/group/'
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    print('..saving {}'.format(outpath + 'questionnaire_ratings.csv'))
    S.to_csv(outpath + 'questionnaire_ratings.csv', index=False)

def collect_flight_performance_group(PATH, subjs=np.arange(4, 25, 1), min_duration=100., min_valid_laps=1):
    print('--------------------------------------------')
    print('using run exclusion criteria:')
    print('minimum duration: {} sec'.format(min_duration))
    print('minimum number of valid laps: {} sec'.format(min_valid_laps))
    print('--------------------------------------------')
    #make the output path
    outpath = PATH + 'group/'
    if os.path.isdir(outpath) is False:
        os.mkdir(outpath)
    A = pd.DataFrame()
    for subj in subjs:
        curr_path = PATH + 's%03d/' % subj
        #walk through subfolders for the different runs
        for walker in os.walk(curr_path):
            #if relevant files are available
            if ('flight_performance.csv' in walker[2]) :
                print('                                                                                     ', end='\r')
                print('..collecting flight performance: {}'.format(walker[0]), end='\r')
                df = pd.read_csv(walker[0] + '/flight_performance.csv')
                #add some relevant information
                df['subject'] = subj
                df['recording'] = int(walker[0].split('/')[-1].split('_')[0])
                condition = walker[0].split('/')[-1].split('_')[1]
                df['condition'] = condition
                df['run'] = np.nan
                df['ts_start_run'] = df.ts_start.min()
                df['ts_end_run'] = df.ts_end.max()
                df['filepath'] = walker[0] + '/flight_performance.csv'
                #apply rejection criteria
                #by default current run is considered valid
                is_valid_run = True
                #check rejection criteria only for the racing task
                if condition in ['flat', 'wave']:
                    # ..check minimum duration of the run criterion
                    if min_duration is not None:
                        if (df.ts_end.max() - df.ts_start.min()) < min_duration:
                            is_valid_run = False
                    #..check minimum number of valid laps criterion
                    if min_valid_laps is not None:
                        if df.is_valid.sum() < min_valid_laps:
                            is_valid_run = False
                #if the current run is valid, append it's data to the output dataframe
                if is_valid_run:
                    if A.size:
                        A = A.append(df)
                    else:
                        A = df
    #add consecutive run numbers for each condition
    for subj in A.subject.unique():
        for cond in A.condition.unique():
            ind = (A.subject == subj) & (A.condition == cond)
            rec = A.loc[ind, ('recording')].values
            rec_id = np.unique(rec)
            run = np.empty(rec.shape)
            run[:] = np.nan
            for i in range(rec_id.shape[0]):
                run[rec == rec_id[i]] = i + 1
            A.loc[ind, ('run')] = run
    A['run'] = A['run'].astype(int)
    A.to_csv(outpath + 'flight_performance.csv', index=False)

def check_valid_laps(A):
    for cond in A.condition.unique():
        print('===========================================================================')
        print('cond, subj, rec, run, maxtime, is_valid_run, num_valid_laps, num_total_laps')
        print('===========================================================================')
        for subj in A.subject.unique():
            print('--------------')
            for rec in A.recording.unique():
                ind = ((A.condition == cond) & (A.subject == subj) & (A.recording == rec))
                if np.sum(ind.astype(int)) > 0:
                    run = A.loc[ind, ('run')].iloc[0]
                    valid_laps = A.loc[ind, ('is_valid')].sum()
                    total_laps = A.loc[ind, ('is_valid')].shape[0]
                    maxtime = A.loc[ind, ('ts_end')].max()
                    has_mintime = maxtime > 100.
                    has_minvalid = valid_laps > 1
                    is_valid_run = has_mintime & has_minvalid
                    print(cond, subj, rec, run, np.round(maxtime, 2), is_valid_run, valid_laps, total_laps)

def compute_performance_ranked(A):
    P = pd.DataFrame()
    # loop over tracks
    for track_name in ['flat', 'wave']:
        # make copy for the current data
        D = A.copy()
        # select current track
        D = D.loc[D.condition == track_name]
        # remove spurious laps, with 2 or less events e.g., [0 10]
        D = D.loc[D.num_events > 2]
        # count total number of laps and number of valid laps per subject
        D['num_laps_total'] = [D.loc[D.subject == D.subject.iloc[i]].shape[0] for i in range(D.shape[0])]
        D['num_laps_valid'] = [D.loc[D.subject == D.subject.iloc[i]].is_valid.sum() for i in range(D.shape[0])]
        D['collisions_total'] = [D.loc[D.subject == D.subject.iloc[i]].num_collisions.sum() for i in range(D.shape[0])]
        # select valid laps
        D = D.loc[D.is_valid == 1]
        # median and best laptimes
        D['lt_median'] = [D.loc[D.subject == D.subject.iloc[i]].lap_time.median() for i in range(D.shape[0])]
        D['lt_best'] = [D.loc[D.subject == D.subject.iloc[i]].lap_time.min() for i in range(D.shape[0])]
        D['is_best'] = (D.lap_time.values == D['lt_best'].values).astype(int)
        D['collisions_valid_laps'] = [D.loc[D.subject == D.subject.iloc[i]].num_collisions.sum() for i in
                                      range(D.shape[0])]
        # rank subjects by best lap time
        D['Rank'] = np.nan
        vals = sorted(D.lt_best.unique())
        for val in vals:
            D.Rank.loc[D.lt_best == val] = np.where(vals == val)[0][0] + 1
        D['Rank'] = D['Rank'].astype(int)
        D
        ## Performance
        # TODO, add:
        # - number of crashes (total)
        # - number of crashes (in valid laps)
        # - TLX scores
        # - link this to subject descriptives: age, experience, subjective ratings
        #Subject ranking
        for rk in sorted(D.Rank.unique()):
            num_laps_total = D.loc[(D.Rank == rk), ('num_laps_total')].iloc[0]
            num_laps_valid = D.loc[(D.Rank == rk), ('num_laps_valid')].iloc[0]
            subject = D.loc[(D.Rank == rk), ('subject')].iloc[0]
            lt_valid = D.loc[(D.Rank == rk), ('lap_time')].values
            lt_best = D.loc[(D.Rank == rk), ('lt_best')].iloc[0]
            lt_median = D.loc[(D.Rank == rk), ('lt_median')].iloc[0]
            collisions_total = D.loc[(D.Rank == rk), ('collisions_total')].iloc[0]
            collisions_valid_laps = D.loc[(D.Rank == rk), ('collisions_valid_laps')].iloc[0]

            df = pd.DataFrame({'track': track_name, 'Rank': rk, 'lt_best': lt_best, 'lt_median': lt_median,
                               'lt_valid': [lt_valid], 'num_laps_valid': num_laps_valid, 'num_laps_total': num_laps_total,
                               'prop_valid_laps': num_laps_valid / num_laps_total, 'num_collisions_total': collisions_total,
                               'num_collisions_valid_laps': collisions_valid_laps, 'subject': subject}, index=[0])

            if P.size:
                P = P.append(df)
            else:
                P = df
    return P

def plot_scatter_laptimes(P, PATH=None):
    for track_name in ['flat', 'wave']:
        df = P.copy()[P.track == track_name]
        plt.figure(figsize=(14, 7))
        for i in sorted(df.Rank.unique()):
            vals = df[df.Rank == i].lt_valid.values[0]
            plt.plot(np.ones(vals.shape) * i, vals, 'ok')
            val = df[df.Rank == i].lt_median.values
            plt.plot(np.array([-0.25, 0.25]) + i, [val, val], 'r-', lw=3)
        plt.legend(['Lap', 'Median Lap Time'], loc='upper left')
        plt.gca().set_xticks(P.Rank.unique())
        plt.xlabel('Subjects Ranked')
        plt.ylabel('Lap Time [s]')
        plt.title('LAPTIME - Track: {}'.format(track_name.upper()))
        plt.ylim((0, 72))
        if PATH is not None:
            plt.savefig(PATH + 'LapTimes_' + track_name + '.png')

def plot_bar_number_of_laps(P, PATH=None):
    for track_name in ['flat', 'wave']:
        df = P.copy()[P.track == track_name]
        total_laps = df.num_laps_total.values
        valid_laps = df.num_laps_valid.values
        invalid_laps = total_laps - valid_laps
        plt.figure(figsize=(15, 5))
        plt.bar(df.Rank, valid_laps, color='k')
        plt.bar(df.Rank, invalid_laps, color='gray', bottom=valid_laps)
        plt.plot([0.5, 21.5], np.array([1, 1]) * np.mean(valid_laps), 'r-', lw=4)
        plt.ylabel('Laps')
        plt.gca().set_xticks(df.Rank)
        plt.xlabel('Rank')
        plt.title('NUMBER OF LAPS ' + track_name.upper())
        plt.legend(['Avg Valid laps', 'Valid Laps', 'Invalid Laps'], loc='upper right')
        plt.ylim((0, 60))
        if PATH is not None:
            plt.savefig(PATH + 'NumberOfLaps_' + track_name + '.png')

def plot_bar_proportion_valid_laps(P, PATH=None):
    for track_name in ['flat', 'wave']:
        df = P.copy()[P.track == track_name]
        plt.figure(figsize=(15, 5))
        plt.bar(df.Rank, df.prop_valid_laps, color='k')
        plt.plot([0.5, 21.5], np.array([1, 1]) * np.mean(df.prop_valid_laps), 'r-', lw=4)
        plt.ylim((0, 1))
        plt.ylabel('Proportion Valid Laps')
        plt.gca().set_xticks(df.Rank)
        plt.xlabel('Rank')
        plt.title('PROPORTION VALID LAPS ' + track_name.upper())
        if PATH is not None:
            plt.savefig(PATH + 'ProportionValidLaps_' + track_name + '.png')

def plot_bar_number_of_collisions(P, PATH=None):
    for track_name in ['flat', 'wave']:
        df = P.copy()[P.track == track_name]
        total_laps = df.num_collisions_total.values
        valid_laps = df.num_collisions_valid_laps.values
        invalid_laps = total_laps - valid_laps
        plt.figure(figsize=(15, 5))
        plt.bar(df.Rank, valid_laps, color='k')
        plt.bar(df.Rank, invalid_laps, color='gray', bottom=valid_laps)
        plt.plot([0.5, 21.5], np.array([1, 1]) * np.mean(valid_laps), 'r-', lw=4)
        plt.ylabel('Collisions')
        plt.gca().set_xticks(df.Rank)
        plt.xlabel('Rank')
        plt.title('NUMBER OF COLLISIONS ' + track_name.upper())
        plt.legend(['Avg Valid laps', 'Valid Laps', 'Invalid Laps'], loc='upper right')
        plt.ylim((0, 360))
        if PATH is not None:
            plt.savefig(PATH + 'NumberOfCollisions_' + track_name + '.png')

def plot_return_laps_in_run_time(y, t=120., lims=(0,10), plot_histogram=False):
    ind = y < t
    if plot_histogram:
        count_in = np.sum((ind == True).astype(int))
        count_out = np.sum((ind == False).astype(int))
        count_total = count_in + count_out
        plt.figure(figsize=(15, 5))
        plt.hist(y, bins=1000)
        plt.plot([t, t], lims, 'r-')
        plt.title('Valid/Total Runs [%]: {}/{} [{:.2f}%]'.format(count_in, count_total, 100 * count_in / count_total))
        plt.xlabel('Time [s]')
        plt.ylabel('Count')
        plt.ylim(lims)
    return ind

def compute_shortest_path_length(T, make_plot=False):
    #the gates to consider
    relevant_gates = np.arange(0, 10, 1) #all gates
    #podium position
    p_podium = np.array([13.049704, -12.342312, 0.940298])
    #gate positions
    p_gates = T.loc[:, ('pos_x', 'pos_y', 'pos_z')].values
    # compute the minimum path length passing through the center of the gates,
    #...for the first lap consider as start position the podium
    p0 = np.vstack((p_podium, p_gates[relevant_gates, :]))
    #...for the other laps consider as start position the start/finish gate
    p1 = np.vstack((p_gates[-1, :], p_gates[relevant_gates, :]))
    #compute path lengths for first lap (l0) and other laps(l1)
    l0 = np.sum(np.linalg.norm(np.diff(p0, axis=0), axis=1))
    l1 = np.sum(np.linalg.norm(np.diff(p1, axis=0), axis=1))
    #plot the paths
    if make_plot:
        plt.figure(figsize=(10, 10))
        plt.subplot(221)
        plt.plot(p0[:, 0], p0[:, 1], 'bo-')
        plt.plot(p1[:, 0], p1[:, 1], 'ro--')
        plt.title('Minimum Path Length\nFirst Lap: {:.2f} m, Other Laps: {:.2f} m'.format(l0, l1))
        plt.legend(('First Laps', 'Other Laps'))
        plt.xlabel('Position X [m]')
        plt.ylabel('Position Y [m]')
        plt.xlim((-28, 28))
        plt.ylim((-28, 28))
        plt.subplot(222)
        plt.plot(p0[:, 1], p0[:, 2], 'bo-')
        plt.plot(p1[:, 1], p1[:, 2], 'ro--')
        plt.legend(('First Laps', 'Other Laps'))
        plt.xlabel('Position Y [m]')
        plt.ylabel('Position Z [m]')
        plt.xlim((-28, 28))
        plt.ylim((-28, 28))
        plt.subplot(223)
        plt.plot(p0[:, 0], p0[:, 2], 'bo-')
        plt.plot(p1[:, 0], p1[:, 2], 'ro--')
        plt.legend(('First Laps', 'Other Laps'))
        plt.xlabel('Position X [m]')
        plt.ylabel('Position Z [m]')
        plt.xlim((-28, 28))
        plt.ylim((-28, 28))
    #return first and other laps path lengths
    return (l0, l1)

def compute_path_length(p, t=None, t_start=None, t_end=None):
    if t is not None:
        if t_start is not None:
            ind = (t >= t_start)
            t = t[ind]
            p = p[ind, :]
        if t_end is not None:
            ind = (t <= t_end)
            t = t[ind]
            p = p[ind, :]
    l = np.nansum(np.linalg.norm(np.diff(p, axis=0), axis=1))
    return l

def questionnaire_fix_run_number(R, F):
    ## Flight Performance data
    # Correct run numbers and drone data finishing timestamps
    # related to the data used for flight performance
    DroneTimes = pd.DataFrame()
    for subj in F.subject.unique():
        for track in F.condition.unique():
            for run in F.run.unique():
                ind = (F.subject == subj) & (F.condition == track) & (F.run == run)
                if np.sum(ind.astype(int)) > 0:
                    curr_duration = F.loc[ind, ('ts_end_run')].iloc[0]
                    curr_path = F.loc[ind, ('filepath')].iloc[0]
                    curr_path_drone = '/'.join(curr_path.split('/')[:-1]) + '/drone.csv'
                    curr_D = pd.read_csv(curr_path_drone, nrows=1)
                    curr_starttime = curr_D.utc_timestamp.iloc[0]
                    curr_endtime = curr_starttime + curr_duration
                    df = pd.DataFrame({'subj': subj, 'run': run, 'track': track,
                                       'timestamp_start': curr_starttime,
                                       'timestamp_end': curr_endtime,
                                       'filepath': curr_path_drone}, index=[0])
                    if DroneTimes.size:
                        DroneTimes = DroneTimes.append(df)
                    else:
                        DroneTimes = df
    DroneTimes = DroneTimes.reset_index().drop(columns=['index'])
    ## Questionnaire Ratings
    # add start timestamp
    ts_start = []
    for i in range(R.shape[0]):
        val = pd.Timestamp(R.Timestamp.iloc[i].split('GMT')[0]).tz_localize('Europe/Paris').timestamp()
        ts_start.append(val)
    R['timestamp_start'] = ts_start
    R
    # add new run number
    run = []
    time_delta = []
    for i in range(R.shape[0]):
        curr_ts = R.timestamp_start.iloc[i]
        curr_subj = R.Subject.iloc[i]
        curr_track = R.Track.iloc[i]
        ind = ((DroneTimes.subj == curr_subj) &
               (DroneTimes.track == curr_track) &
               (DroneTimes.timestamp_end < curr_ts))
        if np.sum(ind.astype(int)) > 0:
            curr_run = DroneTimes.loc[ind, ('run')].iloc[-1]
            curr_delta = curr_ts - DroneTimes.loc[ind, ('timestamp_end')].iloc[-1]
        else:
            curr_run = np.nan
            curr_delta = np.nan
        run.append(curr_run)
        time_delta.append(curr_delta)
    R['new_run'] = run
    R['timestamp_delta'] = time_delta
    # remove new run number double entries
    run = []
    time_delta = []
    for i in range(R.shape[0]):
        curr_subj = R.Subject.iloc[i]
        curr_track = R.Track.iloc[i]
        curr_run = R.new_run.iloc[i]
        curr_delta = R.timestamp_delta.iloc[i]
        ind = ((R.Subject == curr_subj) &
               (R.Track == curr_track) &
               (R.new_run == curr_run))
        if (np.sum(ind.astype(int)) > 1) and (R.loc[ind, ('timestamp_delta')].min() < curr_delta):
            curr_run = np.nan
            curr_delta = np.nan
        run.append(curr_run)
        time_delta.append(curr_delta)
    R['new_run'] = run
    R['timestamp_delta'] = time_delta
    # remove empty rows
    R = R.loc[(np.isnan(R.new_run) == False), :]
    R['Run'] = R.new_run.values.astype(int)
    R = R.drop(columns=['new_run', 'timestamp_start', 'timestamp_delta'])
    return R

def make_performance_table_group(F, R):
    #Questionnaire dictionnary
    quest_dict = {
        'SEQ_Performance': 'Wie bewerten Sie Ihre Leistung insgesamt? [Sehr Schwach, Sehr Stark]',
        'SEQ_Speed': 'Wie bewerten Sie Ihre Geschwindigkeit? [Sehr Niedrig, Sehr Hoch]',
        'SEQ_Accuracy': 'Wie bewerten Sie Ihre Genauigkeit? [Sehr Gering, Sehr Hoch]',
        'SEQ_Concentration': 'Wie bewerten Sie Ihre Konzentration?  [Sehr Niedrig, Sehr Hoch]',
        'SEQ_Agency': 'Wie stark war Ihr Gefühl von Kontrolle über die Bewegungen der Drohne? [Sehr Schwach, Sehr Stark]',
        'TLX_Mental_Demand': 'Wie viel geistige Anforderung war bei der Aufgabe erforderlich? [Sehr Gering, Sehr Hoch]',
        'TLX_Physical_Demand': 'Wie viel körperliche Anforderung war bei der Aufgabe erforderlich? [Sehr Gering, Sehr Hoch]',
        'TLX_Temporal_Demand': 'Wie viel Zeitdruck empfanden Sie während der Aufgabe? [Sehr Gering, Sehr Hoch]',
        'TLX_Performance': 'Wie zufrieden waren Sie mit Ihrer Leistung im Zusammenhang mit der Aufgabe? [Gar nicht, Sehr]',
        'TLX_Effort': 'Insgesamt betrachtet: Wie gross war die von Ihnen empfundene Anstrengung bei der Aufgabe? [Sehr gering, Sehr hoch]',
        'TLX_Frustration': 'Wie frustriert fühlten Sie sich während der Aufgabe? [Sehr gering, Sehr hoch]',
    }
    ## Collect Performance x Run Table
    T = pd.DataFrame()
    for cond in ['flat', 'wave']:
        for run in np.arange(1, 6, 1):

            out_dict = {'Track': cond, 'Run': run}

            ## Flight Performance

            ind = (F.condition == cond) & (F.run == run)
            df = F.copy().loc[ind, :]

            # Number of Subjects
            out_dict['Num_Subjects'] = df.subject.unique().shape[0]

            # Number of Laps in Total
            vals = []
            for subj in df.subject.unique():
                ind = (df.subject == subj)
                vals.append(np.sum(ind.astype(int)))
            out_dict['Num_Laps_Mean'] = np.nanmean(np.array(vals))
            out_dict['Num_Laps_Std'] = np.nanstd(np.array(vals))

            # Number of Valid Laps
            vals = []
            for subj in df.subject.unique():
                ind = (df.subject == subj) & (df.is_valid == 1)
                vals.append(np.sum(ind.astype(int)))
            out_dict['Num_Valid_Laps_Mean'] = np.nanmean(np.array(vals))
            out_dict['Num_Valid_Laps_Std'] = np.nanstd(np.array(vals))

            # Number of Invalid Laps
            vals = []
            for subj in df.subject.unique():
                ind = (df.subject == subj) & (df.is_valid == 0)
                vals.append(np.sum(ind.astype(int)))
            out_dict['Num_Invalid_Laps_Mean'] = np.nanmean(np.array(vals))
            out_dict['Num_Invalid_Laps_Std'] = np.nanstd(np.array(vals))

            # Median Laptime
            vals = []
            for subj in df.subject.unique():
                ind = (df.subject == subj) & (df.is_valid == 1)
                val = np.nanmedian(df.loc[ind, ('lap_time')].values)
                vals.append(val)
            out_dict['Median_Lap_Time_Valid_Laps_Mean'] = np.nanmean(np.array(vals))
            out_dict['Median_Lap_Time_Valid_Laps_Std'] = np.nanstd(np.array(vals))

            # Fastest Laptime
            vals = []
            for subj in df.subject.unique():
                ind = (df.subject == subj) & (df.is_valid == 1)
                val = np.nanmin(df.loc[ind, ('lap_time')].values)
                vals.append(val)
            out_dict['Fastest_Lap_Time_Valid_Laps_Mean'] = np.nanmean(np.array(vals))
            out_dict['Fastest_Lap_Time_Valid_Laps_Std'] = np.nanstd(np.array(vals))

            # Number of Collisions in Total
            vals = []
            for subj in df.subject.unique():
                ind = (df.subject == subj)
                val = np.nansum(df.loc[ind, ('num_collisions')].values)
                vals.append(val)
            out_dict['Num_Collisions_Mean'] = np.nanmean(np.array(vals))
            out_dict['Num_Collisions_Std'] = np.nanstd(np.array(vals))

            # Number of Collisions in Valid Laps
            vals = []
            for subj in df.subject.unique():
                ind = (df.subject == subj) & (df.is_valid == 1)
                val = np.nansum(df.loc[ind, ('num_collisions')].values)
                vals.append(val)
            out_dict['Num_Collisions_Valid_Laps_Mean'] = np.nanmean(np.array(vals))
            out_dict['Num_Collisions_Valid_Laps_Std'] = np.nanstd(np.array(vals))

            # Number of Collisions in Invalid Laps
            vals = []
            for subj in df.subject.unique():
                ind = (df.subject == subj) & (df.is_valid == 0)
                val = np.nansum(df.loc[ind, ('num_collisions')].values)
                vals.append(val)
            out_dict['Num_Collisions_Invalid_Laps_Mean'] = np.nanmean(np.array(vals))
            out_dict['Num_Collisions_Invalid_Laps_Std'] = np.nanstd(np.array(vals))

            # Mean Velocity
            vals = []
            for subj in df.subject.unique():
                ind = (df.subject == subj) & (df.is_valid == 1)
                val = np.nanmean(df.loc[ind, ('Velocity_Mean [m/s]')].values)
                vals.append(val)
            out_dict['Mean_Velocity_Valid_Laps_Mean'] = np.nanmean(np.array(vals))
            out_dict['Mean_Velocity_Valid_Laps_Std'] = np.nanstd(np.array(vals))

            # Max Velocity
            vals = []
            for subj in df.subject.unique():
                ind = (df.subject == subj) & (df.is_valid == 1)
                val = np.nanmean(df.loc[ind, ('Velocity_Max [m/s]')].values)
                vals.append(val)
            out_dict['Max_Velocity_Valid_Laps_Mean'] = np.nanmean(np.array(vals))
            out_dict['Max_Velocity_Valid_Laps_Std'] = np.nanstd(np.array(vals))

            # Path Length
            vals = []
            for subj in df.subject.unique():
                ind = (df.subject == subj) & (df.is_valid == 1)
                val = np.nanmean(df.loc[ind, ('Path_Length [m]')].values)
                vals.append(val)
            out_dict['Path_Length_Valid_Laps_Mean'] = np.nanmean(np.array(vals))
            out_dict['Path_Length_Valid_Laps_Std'] = np.nanstd(np.array(vals))

            # Mean Path Deviation
            vals = []
            for subj in df.subject.unique():
                ind = (df.subject == subj) & (df.is_valid == 1)
                val = np.nanmean(df.loc[ind, ('Path_Deviation_Mean [m]')].values)
                vals.append(val)
            out_dict['Mean_Path_Deviation_Valid_Laps_Mean'] = np.nanmean(np.array(vals))
            out_dict['Mean_Path_Deviation_Valid_Laps_Std'] = np.nanstd(np.array(vals))

            # Max Path Deviation
            vals = []
            for subj in df.subject.unique():
                ind = (df.subject == subj) & (df.is_valid == 1)
                val = np.nanmean(df.loc[ind, ('Path_Deviation_Max [m]')].values)
                vals.append(val)
            out_dict['Max_Path_Deviation_Valid_Laps_Mean'] = np.nanmean(np.array(vals))
            out_dict['Max_Path_Deviation_Valid_Laps_Std'] = np.nanstd(np.array(vals))

            # Mean Z Tilt Angle
            vals = []
            for subj in df.subject.unique():
                ind = (df.subject == subj) & (df.is_valid == 1)
                val = np.nanmean(df.loc[ind, ('Z_Tilt_Angle_Mean [rad]')].values)
                vals.append(val)
            out_dict['Mean_Z_Tilt_Angle_Valid_Laps_Mean'] = np.nanmean(np.array(vals))
            out_dict['Mean_Z_Tilt_Angle_Valid_Laps_Std'] = np.nanstd(np.array(vals))

            # Max Z Tilt Angle
            vals = []
            for subj in df.subject.unique():
                ind = (df.subject == subj) & (df.is_valid == 1)
                val = np.nanmean(df.loc[ind, ('Z_Tilt_Angle_Max [rad]')].values)
                vals.append(val)
            out_dict['Max_Z_Tilt_Angle_Valid_Laps_Mean'] = np.nanmean(np.array(vals))
            out_dict['Max_Z_Tilt_Angle_Valid_Laps_Std'] = np.nanstd(np.array(vals))

            ## Questionnaire ratings

            ind = (R.Track == cond) & (R.Run == run)
            df2 = R.copy().loc[ind, :]

            for label, question in quest_dict.items():
                out_dict[label + '_Mean'] = np.nanmean(df2.loc[:, question].values)
                out_dict[label + '_Std'] = np.nanstd(df2.loc[:, question].values)

            ## Output dataframe

            df_out = pd.DataFrame(out_dict, index=[0])
            if T.size:
                T = T.append(df_out)
            else:
                T = df_out
    T = T.reset_index().drop(columns=['index'])
    return(T)

def make_performance_table_subjects(F, R):
    #Questionnaire dictionnary
    quest_dict = {
        'SEQ_Performance': 'Wie bewerten Sie Ihre Leistung insgesamt? [Sehr Schwach, Sehr Stark]',
        'SEQ_Speed': 'Wie bewerten Sie Ihre Geschwindigkeit? [Sehr Niedrig, Sehr Hoch]',
        'SEQ_Accuracy': 'Wie bewerten Sie Ihre Genauigkeit? [Sehr Gering, Sehr Hoch]',
        'SEQ_Concentration': 'Wie bewerten Sie Ihre Konzentration?  [Sehr Niedrig, Sehr Hoch]',
        'SEQ_Agency': 'Wie stark war Ihr Gefühl von Kontrolle über die Bewegungen der Drohne? [Sehr Schwach, Sehr Stark]',
        'TLX_Mental_Demand': 'Wie viel geistige Anforderung war bei der Aufgabe erforderlich? [Sehr Gering, Sehr Hoch]',
        'TLX_Physical_Demand': 'Wie viel körperliche Anforderung war bei der Aufgabe erforderlich? [Sehr Gering, Sehr Hoch]',
        'TLX_Temporal_Demand': 'Wie viel Zeitdruck empfanden Sie während der Aufgabe? [Sehr Gering, Sehr Hoch]',
        'TLX_Performance': 'Wie zufrieden waren Sie mit Ihrer Leistung im Zusammenhang mit der Aufgabe? [Gar nicht, Sehr]',
        'TLX_Effort': 'Insgesamt betrachtet: Wie gross war die von Ihnen empfundene Anstrengung bei der Aufgabe? [Sehr gering, Sehr hoch]',
        'TLX_Frustration': 'Wie frustriert fühlten Sie sich während der Aufgabe? [Sehr gering, Sehr hoch]',
    }
    ## Collect Performance x Run Table
    T = pd.DataFrame()
    for cond in ['flat', 'wave']:
        for run in F.run.unique():
            for subj in F.subject.unique():

                #pointer to data in the flight performance dataframe
                ind = (F.condition == cond) & (F.run == run) & (F.subject == subj)

                if np.sum(ind.astype(int)) > 0:

                    ## Flight Performance

                    df = F.copy().loc[ind, :]

                    out_dict = {'Track': cond, 'Run': run, 'Subject': subj}
                    out_dict['Num_Subjects'] = 1
                    out_dict['Num_Laps'] = df.shape[0]
                    out_dict['Num_Valid_Laps'] = np.sum((df.is_valid == 1).astype(int))
                    out_dict['Num_Invalid_Laps'] = np.sum((df.is_valid == 0).astype(int))
                    out_dict['Median_Lap_Time_Valid_Laps'] = np.nanmedian(df.loc[(df.is_valid == 1), ('lap_time')].values)
                    out_dict['Fastest_Lap_Time_Valid_Laps'] = np.nanmin(df.loc[(df.is_valid == 1), ('lap_time')].values)
                    out_dict['Num_Collisions'] = np.nansum(df.loc[:, ('num_collisions')].values)
                    out_dict['Num_Collisions_Valid_Laps'] = np.nansum(df.loc[(df.is_valid == 1), ('num_collisions')].values)
                    out_dict['Num_Collisions_Invalid_Laps'] = np.nansum(df.loc[(df.is_valid == 0), ('num_collisions')].values)
                    out_dict['Mean_Velocity_Valid_Laps'] = np.nanmean(df.loc[(df.is_valid == 1), ('Velocity_Mean [m/s]')].values)
                    out_dict['Max_Velocity_Valid_Laps'] = np.nanmean(df.loc[(df.is_valid == 1), ('Velocity_Max [m/s]')].values)
                    out_dict['Path_Length_Valid_Laps'] = np.nanmean(df.loc[(df.is_valid == 1), ('Path_Length [m]')].values)
                    out_dict['Mean_Path_Deviation_Valid_Laps'] = np.nanmean(df.loc[(df.is_valid == 1), ('Path_Deviation_Mean [m]')].values)
                    out_dict['Max_Path_Deviation_Valid_Laps'] = np.nanmean(df.loc[(df.is_valid == 1), ('Path_Deviation_Max [m]')].values)
                    out_dict['Mean_Z_Tilt_Angle_Valid_Laps'] = np.nanmean(df.loc[(df.is_valid == 1), ('Z_Tilt_Angle_Mean [rad]')].values)
                    out_dict['Max_Z_Tilt_Angle_Valid_Laps'] = np.nanmean(df.loc[(df.is_valid == 1), ('Z_Tilt_Angle_Max [rad]')].values)

                    ## Questionnaire ratings

                    ind = (R.Track == cond) & (R.Run == run) & (R.Subject == subj)
                    df2 = R.copy().loc[ind, :]

                    for label, question in quest_dict.items():
                        out_dict[label] = df2.loc[:, question].values

                    ## Output dataframe

                    df_out = pd.DataFrame(out_dict, index=[0])
                    if T.size:
                        T = T.append(df_out)
                    else:
                        T = df_out
    T = T.reset_index().drop(columns=['index'])
    return(T)

def make_performance_table_with_stats(TS, col_index=None):
    if col_index is None:
        col_index = np.arange(4, TS.shape[1], 1)
    ## Make Results table with Run-wise Mean and Std, and stats
    P = pd.DataFrame()
    for track in TS.Track.unique():
        for varname in TS.columns[col_index]:
            out_dir = {}
            out_dir['Track'] = track
            out_dir['Varname'] = varname
            #for each run
            for run in TS.Run.unique():
                vals = TS.loc[((TS.Run == run) & (TS.Track == track)), varname].values

                out_dir['R%d' % run] = '{:.2f} ({:.2f})'.format(np.nanmean(vals), np.nanstd(vals))
            #overall
            vals = TS.loc[(TS.Track == track), varname].values
            out_dir['Avg'] = '{:.2f} ({:.2f})'.format(np.nanmean(vals), np.nanstd(vals))
            #stats
            curr_df = compute_glme(TS.loc[(TS.Track == track), :], varname, 'Run', 'Subject', 0.05).drop(columns=[
                'dep_varname', 'fe_varname', 're_varname'])
            out_dir['Coefficient [95%CI]'] = '{:.2f} [{:.2f}, {:.2f}]'.format(
            curr_df.coefficient.iloc[0], curr_df.ci_low.iloc[0], curr_df.ci_high.iloc[0])
            out_dir['T'] = '{:.2f}'.format(curr_df.tvalue.iloc[0])
            out_dir['P'] = '{:.4f}'.format(curr_df.pvalue.iloc[0])
            #significance
            for alpha in [0.05, 0.01, 0.001, 0.0001]:
                out_dir['P<{:.4f}'.format(alpha)] = (curr_df.pvalue.iloc[0] < alpha).astype(int)
            #output
            out_df = pd.DataFrame(out_dir, index=[0])
            if P.size:
                P = P.append(out_df)
            else:
                P = out_df
    P = P.reset_index().drop(columns=['index'])
    return P

def compute_glme(data, y_name='y', x_name='x', g_name='g', alpha=0.05):
    '''
    input dataframe should contain the columnnames:
        y : dependent variable
        x : fixed_effect_predictor
        g : random_effect_predictor
    '''
    #select data of interest
    data = data.loc[:, (y_name, x_name, g_name)]
    data.columns = ['y', 'x', 'g']
    #fit the GLME model
    md = smf.mixedlm("y ~ x", data, groups=data['g'], re_formula="~x")
    mdf = md.fit(method=["lbfgs"])
    #extract model parameters
    coefficient = mdf.fe_params.x
    ci = mdf.conf_int(alpha=alpha).loc[('x'), :].values
    tvalue = mdf.tvalues.x
    pvalue = mdf.pvalues.x
    df = pd.DataFrame({'dep_varname': y_name, 'fe_varname' : x_name, 're_varname' :  g_name, 'coefficient' : coefficient,
                       'ci_low' : ci[0], 'ci_high' : ci[1], 'tvalue' : tvalue, 'pvalue' : pvalue}, index=[0])
    return df

def correlation_heatmap(data, output="heatmap.png"):
    mask = np.zeros_like(data, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Want diagonal elements as well
    mask[np.diag_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(22, 22))
    # Generate a custom diverging colormap
    cmap = sn.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns_plot = sn.heatmap(data, mask=mask, cmap=cmap, vmin=-1., vmax=1., center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    # save to file
    fig = sns_plot.get_figure()
    fig.savefig(output)

def clean_flight_performance(df):
    ## Keep only racing laps from flat and wave tracks
    ind = (df.condition == 'flat') | (df.condition == 'wave')
    df = df.loc[ind, :]
    ## Remove spurious laps with less than 4 gate passing events
    ind = (df.num_unique > 3)
    df = df.loc[ind, :]
    ## Remove laps that started after the 120s mark
    #  Basically, subjects were allowed to complete their lap when it was started within 120s after run start
    y = df.ts_start.values - df.ts_start_run.values
    t = 120.
    ind = plot_return_laps_in_run_time(y, t=t, lims=(0, 10))
    df = df.loc[ind, :]
    ## Remove invalid laps
    ind = df.is_valid==1
    df = df.loc[ind, :]
    return df

def add_far_fixation(G):
    gates = np.arange(0, 10, 1)
    labels = ['gate%d' % i for i in gates]
    #distances to gates
    D_gates = G.loc[:, (['intersect_gate%d_distance' % i for i in gates])].values
    D_floor = G.loc[:, ('intersect_floor_distance')].values
    #new output variables
    far_intersect_name = []
    far_intersect_distance = []
    far_intersect_pos_x = []
    far_intersect_pos_y = []
    far_intersect_pos_z = []
    #loop over samples
    for i in range(D_gates.shape[0]):
        #if at least some valid gate fixations are present
        if np.sum(np.isnan(D_gates[i, :])) < D_gates.shape[1]:
            #pointer to the currently fixated gate that is farthest away
            j = np.nanargmax(D_gates[i, :])
            #add fixated object label
            far_intersect_name.append(labels[j])
            #add distance to the fixated object
            far_intersect_distance.append(D_gates[i, j])
            #add position of the fixated object
            far_intersect_pos_x.append(G['intersect_gate%d_pos_x' % j].iloc[i])
            far_intersect_pos_y.append(G['intersect_gate%d_pos_y' % j].iloc[i])
            far_intersect_pos_z.append(G['intersect_gate%d_pos_z' % j].iloc[i])
        #if no gate fixations at all
        else:
            #if the floor is fixated
            if np.isnan(D_floor[i]) == False:
                # add fixated object label
                far_intersect_name.append('floor')
                # add distance to the fixated object
                far_intersect_distance.append(D_floor[i])
                # add position of the fixated object
                far_intersect_pos_x.append(G['intersect_floor_pos_x'].iloc[i])
                far_intersect_pos_y.append(G['intersect_floor_pos_y'].iloc[i])
                far_intersect_pos_z.append(G['intersect_floor_pos_z'].iloc[i])
            #if no object is fixated
            else:
                # add fixated object label
                far_intersect_name.append('None')
                # add distance to the fixated object
                far_intersect_distance.append(np.nan)
                # add position of the fixated object
                far_intersect_pos_x.append(np.nan)
                far_intersect_pos_y.append(np.nan)
                far_intersect_pos_z.append(np.nan)
    #save the output variables
    G['far_intersect_name'] = far_intersect_name
    G['far_intersect_distance'] = far_intersect_distance
    G['far_intersect_pos_x'] = far_intersect_pos_x
    G['far_intersect_pos_y'] = far_intersect_pos_y
    G['far_intersect_pos_z'] = far_intersect_pos_z
    return G

def add_fast_medium_slow_condition_labels(F, print_summary=False):
    # determine condtion speed labels based on a median-like split of laptimes in 3 brackets
    condition_speed = np.zeros((F.shape[0], 1))
    for subject in F.subject.unique():
        for track in F.condition.unique():
            # select valid laptimes of current subject
            ind = (F.subject == subject) & (F.condition == track)
            lap_times = np.sort(F.loc[ind, ('lap_time')])
            # cutoff laptimes between slow, medium and fast conditions (median like)
            cutoff_medium_fast = lap_times[int(lap_times.shape[0] / 3)]
            cutoff_slow_medium = lap_times[int(2 * lap_times.shape[0] / 3)]
            # make condition labels
            ind = (F.subject == subject) & (F.condition == track) & (F.lap_time <= cutoff_medium_fast)
            condition_speed[ind] = 1
            ind = (F.subject == subject) & (F.condition == track) & (F.lap_time > cutoff_medium_fast) & (F.lap_time <= cutoff_slow_medium)
            condition_speed[ind] = 2
            ind = (F.subject == subject) & (F.condition == track) & (F.lap_time > cutoff_slow_medium)
            condition_speed[ind] = 3
    # add condition labels to the data array
    condition_speed_labels = []
    for val in condition_speed:
        if val == 1:
            condition_speed_labels.append('fast')
        elif val == 2:
            condition_speed_labels.append('medium')
        elif val == 3:
            condition_speed_labels.append('slow')
        else:
            condition_speed_labels.append('NONE')
    F['condition_speed_fms'] = condition_speed_labels
    # print number of trials, median laptime and SD per condition per subject
    if print_summary:
        for subject in F.subject.unique():
            for track in F.condition.unique():
                labels = sorted(F.condition_speed_fms.unique())
                num_laps = []
                median_laptime = []
                iqr_laptime = []
                # select valid laptimes of current subject
                for label in labels:
                    ind = (F.subject == subject) & (F.condition == track) & (F.condition_speed_fms == label)
                    num_laps.append(np.sum(ind.astype(int)))
                    median_laptime.append(np.nanmedian(F.loc[ind, ('lap_time')].values))
                    iqr_laptime.append(iqr(F.loc[ind, ('lap_time')].values))
                print('===========')
                print('subject {}'.format(subject))
                print('track {}'.format(track))
                print(labels)
                print('Num Laps:       {}'.format(num_laps))
                print('Laptime Median: {}'.format(median_laptime))
                print('Laptime IQR:    {}'.format(iqr_laptime))
    return F

def add_fast_slow_condition_labels(F, print_summary=False):
    # determine condtion speed labels based on a median-like split of laptimes in 3 brackets
    condition_speed = np.zeros((F.shape[0], 1))
    for subject in F.subject.unique():
        for track in F.condition.unique():
            # select valid laptimes of current subject
            ind = (F.subject == subject) & (F.condition == track)
            lap_times = np.sort(F.loc[ind, ('lap_time')])
            # cutoff laptimes between slow,fast conditions (median like)
            cutoff_slow_fast = np.nanmedian(lap_times)
            # make condition labels
            ind = (F.subject == subject) & (F.condition == track) & (F.lap_time <= cutoff_slow_fast)
            condition_speed[ind] = 1
            ind = (F.subject == subject) & (F.condition == track) & (F.lap_time > cutoff_slow_fast)
            condition_speed[ind] = 2
    # add condition labels to the data array
    condition_speed_labels = []
    for val in condition_speed:
        if val == 1:
            condition_speed_labels.append('fast')
        elif val == 2:
            condition_speed_labels.append('slow')
        else:
            condition_speed_labels.append('NONE')
    F['condition_speed_fs'] = condition_speed_labels
    # print number of trials, median laptime and SD per condition per subject
    if print_summary:
        for subject in F.subject.unique():
            for track in F.condition.unique():
                labels = sorted(F.condition_speed_fs.unique())
                num_laps = []
                median_laptime = []
                iqr_laptime = []
                # select valid laptimes of current subject
                for label in labels:
                    ind = (F.subject == subject) & (F.condition == track) & (F.condition_speed_fs == label)
                    num_laps.append(np.sum(ind.astype(int)))
                    median_laptime.append(np.nanmedian(F.loc[ind, ('lap_time')].values))
                    iqr_laptime.append(iqr(F.loc[ind, ('lap_time')].values))
                print('===========')
                print('subject {}'.format(subject))
                print('track {}'.format(track))
                print(labels)
                print('Num Laps:       {}'.format(num_laps))
                print('Laptime Median: {}'.format(median_laptime))
                print('Laptime IQR:    {}'.format(iqr_laptime))
    return F

def gate_fixations_from_flight_performance(F, outfilepath, gate_ids=np.arange(0, 10, 1), min_dur=0.100):
    #header for the ouput file
    header = ['subject', 'track', 'run', 'lap', 'gate', 'ts', 'filepath', 'p2d_x', 'p2d_y', 'distance', 'lap_time',
              'condition_speed_fms', 'condition_speed_fs']
    #write the header
    with open(outfilepath, 'w', newline='') as logfile:
        writer = csv.writer(logfile)
        writer.writerow(header)
        #name of the past filename
        past_filepath = ''
        #loop over valid laps (of all runs of all subjects)
        for row in range(F.shape[0]):
            print('valid lap {}/{}'.format(row, F.shape[0]), end='\r')
            #collect gaze trace data, or make sure the currently loaded on matches the desired one
            curr_filepath = '/'.join(F.filepath.iloc[row].split('/')[:-1]) + '/gaze_traces.csv'
            if curr_filepath != past_filepath:
                G = pd.read_csv(curr_filepath)
            #update past filename
            past_filepath = curr_filepath
            #loop over gates
            for gate_id in gate_ids:
                #get gate AOI hits, considering all intersections (close or far), removing small segments
                time = G.ts.values
                hits = on_target_from_dataframe(G, 'gate%d' % gate_id)
                if min_dur is not None:
                    hits = reject_small_segments(hits, min_dur, time)
                #select timepoints from the current lap and of gate fixation
                ind = (G.ts > F.ts_start.iloc[row]) & (G.ts <= F.ts_end.iloc[row]) & (hits == 1)
                #number of samples
                samples = np.sum(ind.astype(int)).astype(int)
                #if some samples were found, proceed
                if samples > 0:
                    #subject
                    subject = np.ones((samples)) * F.subject.iloc[row]
                    #track
                    track = [F.condition.iloc[row] for rep in range(samples)]
                    #run
                    run = np.ones((samples)) * F.run.iloc[row]
                    #lap
                    lap = np.ones((samples)) * F.lap.iloc[row]
                    #laptime
                    lap_time = np.ones((samples)) * F.lap_time.iloc[row]
                    #gate
                    gate = np.ones((samples)) * gate_id
                    #timestamp
                    timestamp = G.loc[ind, ('ts')].values
                    #filepath
                    filepath = [curr_filepath for rep in range(samples)]
                    #gaze interaction with gate in 2d, coordinates
                    p2d_x = G.loc[ind, ('intersect_gate%d_2d_x' % gate_id)].values
                    p2d_y = G.loc[ind, ('intersect_gate%d_2d_y' % gate_id)].values
                    #distance
                    distance = G.loc[ind, ('intersect_gate%d_distance' % gate_id)].values
                    #condition_speed_fms (fast medium slow)
                    condition_speed_fms = [F.condition_speed_fms.iloc[row] for rep in range(samples)]
                    # condition_speed_fs (fast slow)
                    condition_speed_fs = [F.condition_speed_fs.iloc[row] for rep in range(samples)]
                    #make output dataframe for the current lap
                    df = pd.DataFrame({'subject' : subject, 'track' : track, 'run' : run, 'lap' : lap, 'gate' : gate, 'ts' : timestamp,
                                       'filepath' : filepath, 'p2d_x' : p2d_x, 'p2d_y' : p2d_y, 'distance' : distance, 'lap_time': lap_time,
                                       'condition_speed_fms' : condition_speed_fms, 'condition_speed_fs' : condition_speed_fs})
                    #append data frame to the big output csv file
                    for i in range(df.shape[0]):
                        line = df.iloc[i].values
                        writer.writerow(line)

def on_target_from_dataframe(df, name):
    return (np.isnan(df.loc[:, ('intersect_{}_distance'.format(name))].values)==False).astype(int)

def reject_small_segments(signal, min_seg=None, time=None):
    #make sure the signal is one dimensional
    signal = signal.flatten()
    #set time vector
    if time is None:
        time = np.arange(0, signal.shape[0], 1)
    #set minimum segements vector
    if min_seg is None:
        min_seg = np.nanmedian(np.diff(time))
    #determine timestamps of onsets
    if signal[0] > 0:
        init_val = True
    else:
        init_val = False
    ind = np.hstack((init_val, np.diff(signal) > 0))
    onsets = time[ind]
    #determine timestamps of offsets
    if signal[-1] > 0:
        end_val = True
    else:
        end_val = False
    ind = np.hstack((np.diff(signal) < 0, end_val))
    offsets = time[ind]
    #remove small segments
    ind = (offsets-onsets) >= min_seg
    onsets = onsets[ind]
    offsets = offsets[ind]
    #make output vector
    y = np.zeros(signal.shape)
    for i in range(onsets.shape[0]):
        ind = (time >= onsets[i]) & (time <= offsets[i])
        y[ind] = 1.
#     print(np.vstack((np.vstack((onsets, offsets)), (offsets-onsets))).T )
    return(y)

def plot_gaze_fixations(PATH, single_laps=True, average_across_laps=True, group_average=True):
    #load the data
    O = pd.read_csv(PATH + 'gaze_fixations.csv')
    resolution = 100  # image resolution
    sigma = 20  # currently use a fixed sigma value
    ##================
    # Single lap plots
    if single_laps:
        for track in O.track.unique():
            for gate in O.gate.unique().astype(int):
                for subject in O.subject.unique().astype(int):
                    #make output folder
                    outpath = PATH + 'gaze_fixations/track-{}/gate-{}/single_laps/'.format(track, '%02d' % gate)
                    if os.path.exists(outpath) == False:
                        make_path(outpath)
                    #get the number of runs and laps per subject
                    runs = O.loc[(O.subject == subject), ('run')].unique().astype(int)
                    laps = O.loc[(O.subject == subject), ('lap')].unique().astype(int)
                    #make a new figure
                    fig = plt.figure(figsize=(20,15))
                    panels = [7, 7]
                    #loop over valid laps across runs
                    i=0
                    for run in runs:
                        for lap in laps:
                            #pointer to valid datapoints
                            ind = ((O.subject == subject) & (O.track == track) &
                                   (O.gate == gate) & (O.run == run) & (O.lap == lap))
                            #if some data was found, proceed
                            if np.sum(ind.astype(int))>0:
                                i = i+1
                                ax = plt.subplot(panels[0], panels[1], i)
                                p = O.loc[ind, ('p2d_x', 'p2d_y')].values #fixation points are normalized
                                #make the heatmap
                                heatmap = make_heatmap(p, resolution, sigma)
                                ax.imshow(heatmap)
                                plt.title('run-{}, lap-{}'.format(run, lap))
                                ax.set_xticks([])
                                ax.set_yticks([])
                    #save the figure
                    fig.savefig(outpath + 'track-{}_gate-{}_subj-{}.png'.format(track, '%02d' % gate, '%02d' % subject))
                    plt.clf()
                    fig = None
    ##===================
    # Average across laps
    if average_across_laps:
        for track in O.track.unique():
            for gate in O.gate.unique().astype(int):
                for subject in O.subject.unique().astype(int):
                    # make output folder
                    outpath = PATH + 'gaze_fixations/track-{}/gate-{}/average_laps/'.format(track, '%02d' % gate)
                    if os.path.exists(outpath) == False:
                        make_path(outpath)
                    # get the number of runs and laps per subject
                    runs = O.loc[(O.subject == subject), ('run')].unique().astype(int)
                    laps = O.loc[(O.subject == subject), ('lap')].unique().astype(int)
                    # make a new figure
                    fig = plt.figure(figsize=(20, 15))
                    panels = [7, 7]
                    # loop over valid laps across runs compute mean fixation point per lap
                    p = np.empty((0, 2))
                    for run in runs:
                        for lap in laps:
                            # pointer to valid datapoints
                            ind = ((O.subject == subject) & (O.track == track) &
                                   (O.gate == gate) & (O.run == run) & (O.lap == lap))
                            # if some data was found, proceed
                            if np.sum(ind.astype(int)) > 0:
                                p = np.vstack((p, np.nanmedian(O.loc[ind, ('p2d_x', 'p2d_y')].values, axis=0)))  # fixation points are normalized

                    # make the heatmap
                    heatmap = make_heatmap(p, resolution, sigma)
                    ax = plt.subplot(1,1,1)
                    ax.imshow(heatmap)
                    plt.title('gate-{}, subject-{}'.format(gate, subject))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # save the figure
                    fig.savefig(outpath + 'track-{}_gate-{}_subj-{}.png'.format(track, '%02d' % gate, '%02d' % subject))
                    plt.clf()
                    fig = None
    ##=======================
    # Average across subjects
    if group_average:
        for track in O.track.unique():
            for gate in O.gate.unique().astype(int):
                p_group = np.empty((0, 2))
                for subject in O.subject.unique().astype(int):
                    # make output folder
                    outpath = PATH + 'gaze_fixations/track-{}/gate-{}/group_average/'.format(track, '%02d' % gate)
                    if os.path.exists(outpath) == False:
                        make_path(outpath)
                    # get the number of runs and laps per subject
                    runs = O.loc[(O.subject == subject), ('run')].unique().astype(int)
                    laps = O.loc[(O.subject == subject), ('lap')].unique().astype(int)
                    # make a new figure
                    fig = plt.figure(figsize=(20, 15))
                    panels = [7, 7]
                    # loop over valid laps across runs compute mean fixation point per lap
                    p = np.empty((0, 2))
                    for run in runs:
                        for lap in laps:
                            # pointer to valid datapoints
                            ind = ((O.subject == subject) & (O.track == track) &
                                   (O.gate == gate) & (O.run == run) & (O.lap == lap))
                            # if some data was found, proceed
                            if np.sum(ind.astype(int)) > 0:
                                p = np.vstack((p, np.nanmedian(O.loc[ind, ('p2d_x', 'p2d_y')].values, axis=0)))  # fixation points are normalized
                    p_group = np.vstack((p_group, p))
                # make the heatmap
                heatmap = make_heatmap(p_group, resolution, sigma)
                ax = plt.subplot(1,1,1)
                ax.imshow(heatmap)
                plt.title('gate-{}'.format(gate))
                ax.set_xticks([])
                ax.set_yticks([])
                # save the figure
                fig.savefig(outpath + 'track-{}_gate-{}.png'.format(track, '%02d' % gate))
                plt.clf()
                fig = None

def aoi_hit_event_plot(M, t=None, labels=None):
    if t is None:
        t = np.arange(0, M.shape[0], 1)
    plt.figure(figsize=(15, 9))
    for i in range(M.shape[0]):
        plt.eventplot(t[M[:, i] == 1], lineoffsets=-i, linelengths=1, color='k')
    plt.gca().set_yticks(-np.arange(0, M.shape[0]))
    if labels is not None:
        plt.gca().set_yticklabels(labels)
    plt.xlabel('time [s]')
    plt.ylim((-0.5, M.shape[0] + 0.5))

def collect_activation_maps(O, grid_size=(5,5)):
    #output matrix with activation maps
    M = np.empty((0, grid_size[0], grid_size[1]))
    #output variables
    subject = []
    track = []
    run = []
    lap = []
    gate = []
    condition_speed_fms = []
    condition_speed_fs = []
    #trial counter
    i = 0
    #loop over all condition combinations
    for t in O.track.unique():
        for s in O.subject.unique():
            for r in O.run.unique():
                for l in O.lap.unique():
                    for g in O.gate.unique():
                        #check if this combination of parameters exist in the data table
                        ind = ((O.gate.values == g) & (O.lap.values == l) & (O.run.values == r) &
                               (O.subject.values == s) & (O.track.values == t))
                        #if at least two trials were found
                        if np.sum(ind) > 1:
                            #print current trial number to prompt
                            print('trial {}'.format(i), end='\r')
                            # get the intersection points for the current trial
                            p = O.loc[ind, ('p2d_x', 'p2d_y')].values
                            # make activation maps
                            m = gridmaps(p, grid_size).reshape((1, grid_size[0], grid_size[1]))
                            #append activation maps to output matrix
                            M = np.vstack((M, m))
                            # other condition information
                            subject.append(int(O.loc[ind, ('subject')].iloc[0]))
                            track.append(O.loc[ind, ('track')].iloc[0])
                            run.append(int(O.loc[ind, ('run')].iloc[0]))
                            lap.append(int(O.loc[ind, ('lap')].iloc[0]))
                            gate.append(int(O.loc[ind, ('gate')].iloc[0]))
                            condition_speed_fms.append(O.loc[ind, ('condition_speed_fms')].iloc[0])
                            condition_speed_fs.append(O.loc[ind, ('condition_speed_fs')].iloc[0])
                            # raise the trial counter
                            i += 1
    #make output dataframe
    df = pd.DataFrame({'subject' : subject, 'track' : track, 'run' : run, 'lap' : lap, 'gate' : gate,
                       'condition_speed_fms' : condition_speed_fms, 'condition_speed_fs' : condition_speed_fs})
    #return the ouput matrix and the trial information dataframe
    return M, df

def activation_map_stats(M, info, track='flat', gate=0, ivar='condition_speed_fs'):
    #select data for the current track and gate
    ind = (info.track == track) & (info.gate == gate)
    curr_info = info.copy().loc[ind, :]
    curr_M = M[ind, :, :]
    #output arrays for the statistical results
    coef = np.empty((curr_M.shape[1], curr_M.shape[2]))
    coef[:] = np.nan
    tval = np.empty((curr_M.shape[1], curr_M.shape[2]))
    tval[:] = np.nan
    pval = np.empty((curr_M.shape[1], curr_M.shape[2]))
    pval[:] = np.nan
    ci_low = np.empty((curr_M.shape[1], curr_M.shape[2]))
    ci_low[:] = np.nan
    ci_high = np.empty((curr_M.shape[1], curr_M.shape[2]))
    ci_high[:] = np.nan
    #loop over grid rows and columns
    for row in range(curr_M.shape[1]):
        for col in range(curr_M.shape[2]):
            #dependent measure
            y = curr_M[:, row, col].flatten()
            #only proceed if dependent measures vary
            if np.sum(np.diff(y)) > 0:
                #dataframe to be used for the glm
                df = curr_info.copy()
                #add dependetn measure
                df['y'] = y
                #add independent variable
                cond = np.zeros((df.shape[0],))
                cond_labels = sorted(df.loc[:, (ivar)].unique())
                for i in range(len(cond_labels)):
                    cond[df.loc[:, (ivar)].values == cond_labels[i]] = i
                df['cond'] = cond
                #compute the GLMM, using a subjects random effect
                stats = compute_glme(df, 'y', 'cond', 'subject')
                #save the outputs for the current map
                coef[row, col] = stats.coefficient.iloc[0]
                tval[row, col] = stats.tvalue.iloc[0]
                pval[row, col] = stats.pvalue.iloc[0]
                ci_low[row, col] = stats.ci_low.iloc[0]
                ci_high[row, col] = stats.ci_high.iloc[0]
    #return the statistical results
    stats = {'tvalue' : tval, 'pvalue' : pval, 'coefficient' : coef, 'ci_low' : ci_low, 'ci_high' : ci_high}
    return stats

def activation_map_condition_averages(M, info, track='flat', gate=0, ivar='condition_speed_fs'):
    #pre-select data for the current track and gate
    ind = (info.track == track) & (info.gate == gate)
    curr_info = info.copy().loc[ind, :]
    curr_M = M[ind, :, :]
    #relevant variables for output data
    grid_size = (M.shape[1], M.shape[2])
    subjects = np.sort(curr_info.subject.unique())
    conditions = np.sort(curr_info[ivar].unique())
    #output map
    avg_M = np.empty((0, grid_size[0], grid_size[1]))
    #loop over conditions and subjects
    for condition in conditions:
        m = np.empty((0, grid_size[0], grid_size[1]))
        for subject in subjects:
            ind = (curr_info[ivar] == condition) & (curr_info.subject == subject)
            m = np.vstack((m, np.nanmean(curr_M[ind, :, :], axis=0).reshape((1, grid_size[0], grid_size[1]))))
        avg_M = np.vstack((avg_M, np.nanmean(m, axis=0).reshape((1, grid_size[0], grid_size[1]))))
    out = {'data' : avg_M, 'labels' : conditions}
    return out

def make_aoi_speed_average_and_stats_plot(M, info, PATH, track='flat', factor_name='condition_speed_fs', gates=np.arange(0, 10, 1)):
    # todo : add saving of glmm results to csv
    grid_size = (M.shape[1], M.shape[2])
    factor_levels = sorted(info[factor_name].unique())
    nlevels = len(factor_levels)
    ngates = gates.shape[0]
    # make output path
    outpath = PATH + 'activation_maps_stats/' + 'track-{}/'.format(track) + '{}/'.format(factor_name)
    if os.path.exists(outpath) is False:
        make_path(outpath)
    # make the figure
    fig = plt.figure(figsize=(16, 6))
    i = 0
    alpha = 0.05
    for gate in gates:
        # show condition average across subjects
        cond_avg = activation_map_condition_averages(M, info, track=track, gate=gate, ivar=factor_name)
        i = i + 1
        #IMPORTANT NOTE: imshow shows (x=down, y=right, origin=top left), however pixel coordinates are (x=right, y=down, origin=top left)
        #THUS: Transpose before plotting
        for j in range(nlevels):
            plt.subplot(nlevels + 2, ngates, i + (j * ngates))
            plt.imshow(np.squeeze(cond_avg['data'][j, :, :]).T,
                       cmap='hot',
                       vmin=np.nanmin(cond_avg['data'].flatten()),
                       vmax=np.nanmax(cond_avg['data'].flatten()))
            plt.title('gate {}: {}'.format(gate, cond_avg['labels'][j]))
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
        # show p-value
        stats = activation_map_stats(M, info, track=track, gate=gate, ivar='condition_speed_fs')
        plt.subplot(nlevels + 2, ngates, i + (nlevels * ngates))
        plt.imshow(stats['tvalue'].T,
                   cmap='bwr',
                   vmin=-3.,
                   vmax=3.)
        plt.title('gate {}: tval'.format(gate))
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        # show statistical difference
        plt.subplot(nlevels + 2, ngates, i + ((nlevels + 1) * ngates))
        plt.imshow((stats['pvalue'] < alpha).T,
                   cmap='hot',
                   vmin=0.,
                   vmax=1.)
        plt.title('gate {}: stat diff'.format(gate))
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
    plt.tight_layout()
    plt.savefig(outpath + 'glmm_grid-%03d' % grid_size[0] + 'x%03d' % grid_size[1] + '.jpg')

def make_aoi_turn_average_and_stats_plot(M, info, PATH, track='flat', factor_name='condition_turn_lr',
                                         gates_left=[1, 2, 3, 4, 5], gates_right=[6, 7, 8, 9, 0]):
    #determine size of the grid
    grid_size = (M.shape[1], M.shape[2])
    #add condition labels
    condition_labels = np.zeros((info.shape[0]))
    condition_labels[:] = -1
    # update gate numbers
    gate_labels = np.zeros((info.shape[0]))
    gate_labels[:] = -1
    #update stuff
    for i in range(len(gates_left)):
        ind = info.gate.values == gates_left[i]
        condition_labels[ind] = 1
        gate_labels[ind] = i
    for i in range(len(gates_right)):
        ind = info.gate.values == gates_right[i]
        condition_labels[ind] = 2
        gate_labels[ind] = i
    #update these variables
    info[factor_name] = condition_labels.astype(int)
    info['gate'] = gate_labels.astype(int)
    #now perform the processing pipeline
    gates = np.sort(np.array(info.gate.unique()))
    ngates = gates.shape[0]
    factor_levels = np.sort(info[factor_name].unique())
    nlevels = len(factor_levels)
    # make output path
    outpath = PATH + 'activation_maps_stats/' + 'track-{}/'.format(track) + '{}/'.format(factor_name)
    if os.path.exists(outpath) is False:
        make_path(outpath)
    # make the figure
    fig = plt.figure(figsize=(16, 8))
    i = 0
    alpha = 0.05 / (grid_size[0] * grid_size[1])
    for gate in gates:
        # show condition average across subjects
        cond_avg = activation_map_condition_averages(M, info, track=track, gate=gate, ivar=factor_name)
        i = i + 1
        #IMPORTANT NOTE: imshow shows (x=down, y=right, origin=top left), however pixel coordinates are (x=right, y=down, origin=top left)
        #THUS: Transpose before plotting
        num_actual_levels = cond_avg['data'].shape[0]
        for j in range(num_actual_levels):
            plt.subplot(nlevels + 2, ngates, i + (j * ngates))
            plt.imshow(np.squeeze(cond_avg['data'][j, :, :]).T,
                       cmap='hot',
                       vmin=np.nanmin(cond_avg['data'].flatten()),
                       vmax=np.nanmax(cond_avg['data'].flatten()))
            plt.title('gate {}: {}'.format(gate, cond_avg['labels'][j]))
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
        if num_actual_levels > 1:
            # show p-value
            stats = activation_map_stats(M, info, track=track, gate=gate, ivar=factor_name)
            plt.subplot(nlevels + 2, ngates, i + (nlevels * ngates))
            plt.imshow(stats['tvalue'].T,
                       cmap='bwr',
                       vmin=-3.,
                       vmax=3.)
            plt.title('gate {}: tval'.format(gate))
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            # show statistical difference
            plt.subplot(nlevels + 2, ngates, i + ((nlevels + 1) * ngates))
            plt.imshow((stats['pvalue'] < alpha).T,
                       cmap='hot',
                       vmin=0.,
                       vmax=1.)
            plt.title('gate {}: stat diff'.format(gate))
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
    plt.tight_layout()
    plt.savefig(outpath + 'glmm_grid-%03d' % grid_size[0] + 'x%03d' % grid_size[1] + '.jpg')

def collect_first_fixations(O):
    #pointers to the output data
    ind_ff = np.zeros((O.shape[0], 1))
    ind_lf = np.zeros((O.shape[0], 1))
    #some relevant variables to use for selecting trials
    ts = O.ts.values
    tracks = O.track.values
    subjects = O.subject.values
    gates = O.gate.values
    runs = O.run.values
    laps = O.lap.values
    #loop over conditions
    for track in O.track.unique():
        for subject in np.unique(subjects):
            for gate in np.unique(gates):
                ind = (tracks == track) & (subjects == subject) & (gates == gate)
                for run in np.unique(runs[ind]):
                    ind = (tracks == track) & (subjects == subject) & (gates == gate) & (runs == run)
                    for lap in np.unique(laps[ind]):
                        #print the current trial number
                        print('trial {}'.format(np.sum(ind_ff)), end='\r')
                        #pointer to all gate fixations in the current lap
                        ind = (tracks == track) & (subjects == subject) & (gates == gate) & (runs == run) & (laps == lap)
                        #first fixations
                        first_fixation = np.nanmin(ts[ind])
                        ind_ff[ind & (ts == first_fixation)] = 1
                        #last fixations
                        last_fixation = np.nanmax(ts[ind])
                        ind_lf[ind & (ts == last_fixation)] = 1
    #return data of the the first fixations
    out = O.copy().loc[(ind_ff == 1), :]
    return(out)

def add_first_fixation_features(FF, O):
    # some relevant variables to use for selecting trials
    ts = O.ts.values
    tracks = O.track.values
    subjects = O.subject.values
    gates = O.gate.values
    runs = O.run.values
    laps = O.lap.values
    #output variables
    ts_first = FF.loc[:, ('ts')].values
    ts_last = np.empty((FF.shape[0],))
    ts_last[:] = np.nan
    hits = np.empty((FF.shape[0],))
    hits[:] = np.nan
    lap_time = np.empty((FF.shape[0],))
    lap_time[:] = np.nan
    ts_gate = np.empty((FF.shape[0],))
    ts_gate[:] = np.nan
    # loop over conditions
    filepath_previous = ''
    for i in range(FF.shape[0]):
        print('{}/{}'.format(i, FF.shape[0]), end='\r')
        #pointer to current lap in O
        ind = ((tracks == FF.track.iloc[i]) &
               (subjects == FF.subject.iloc[i]) &
               (gates == FF.gate.iloc[i]) &
               (runs == FF.run.iloc[i]) &
               (laps == FF.lap.iloc[i]))
        #get some features
        ts_last[i] = np.nanmax(O.loc[ind, ('ts')].values)
        hits[i] = np.sum(ind)
        lap_time[i] = O.loc[ind, ('lap_time')].iloc[0]
        #also find the gate passing time from the events file of the subject
        filepath_current = '/'.join(O.loc[ind, ('filepath')].iloc[0].split('/')[:-1]) + '/flight_performance.csv'
        if filepath_previous != filepath_current:
            df = pd.read_csv(filepath_current)
            filepath_previous = filepath_current
        ind = (df.lap.values == FF.lap.iloc[i])
        _gate_id = np.fromstring(df.loc[ind, 'gate_id'].values[0].strip('[]'), dtype=int, sep=' ')
        _gate_ts = np.fromstring(df.loc[ind, 'gate_timestamps'].values[0].strip('[]'), dtype=np.float, sep=' ')
        vals = [_gate_ts[j] for j in range(_gate_id.shape[0]) if (_gate_id[j] == (FF.gate.iloc[i] + 1))]
        if len(vals) == 0:
            vals = np.nan
        else:
            vals = vals[0]
        ts_gate[i] = vals
    #save output data
    FF['lap_time'] = lap_time
    FF['ts_first_fixation'] = ts_first
    FF['ts_last_fixation'] = ts_last
    FF['ts_gate'] = ts_gate
    FF['td_first_last'] = ts_last - ts_first
    FF['td_first_gate'] = ts_gate - ts_first
    FF['td_gate_last'] = ts_last - ts_gate
    FF['num_hits'] = hits
    FF['fixation_duration'] = hits / 500 #duration of total fixation assuming fixed 500 Hz sampling rate
    return FF

def multicolor_line(x, y, d, fig=None, ax=None, linewidth=10, cmap='viridis', clims=None):
    ind = np.isnan(x)==False
    x = x[ind].flatten()
    y = y[ind].flatten()
    d = d[ind].flatten()
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    if (fig == None) and (ax == None):
        fig = plt.figure()
        ax = plt.subplot(111)
    # Create a continuous norm to map from data points to colors
    if clims is None:
        clims = (np.nanmin(d), np.nanmax(d))
    norm = plt.Normalize(clims[0], clims[1])
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    # Set the values used for colormapping
    lc.set_array(d)
    lc.set_linewidth(linewidth)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)

def collect_normalized_trajectories(PATH, win=(0, 200), res=1.):
    # load lap performance table
    F = pd.read_csv(PATH + 'flight_performance.csv')
    # select valid laps and those within 120 sec
    F = clean_flight_performance(F)
    # label conditions as fast/medium/slow according to laptime
    F = add_fast_medium_slow_condition_labels(F)
    # label conditions as fast/slow according to laptime
    F = add_fast_slow_condition_labels(F)
    # extract and compute normalized trajectories
    filepath_previous = ''
    O = pd.DataFrame()
    for i in range(F.shape[0]):
        print('line {}/{}'.format(i, F.shape[0]), end='\r')
        filepath_current = '/'.join(F.filepath.iloc[i].split('/')[:-1]) + '/drone.csv'
        if filepath_current != filepath_previous:
            D = pd.read_csv(filepath_current)
            filepath_previous = filepath_current
        df = get_norm_trajectories(D, t0=F.ts_start.iloc[i], t1=F.ts_end.iloc[i], gate_id=F.gate_id.iloc[i],
                                   gate_ts=F.gate_timestamps.iloc[i], win=win, res=res)
        if df is not None:
            df['subject'] = F.subject.iloc[i]
            df['track'] = F.condition.iloc[i]
            df['run'] = F.run.iloc[i]
            df['lap'] = F.lap.iloc[i]
            df['condition_speed_fms'] = F.condition_speed_fms.iloc[i]
            df['condition_speed_fs'] = F.condition_speed_fs.iloc[i]
            O = O.append(df)
    #reset index
    O = O.reset_index().drop(columns=['index'])
    #make output folder
    outpath = PATH + 'trajectories/res-%04dmm/' % (int(res*1000))
    if os.path.exists(outpath) == False:
        make_path(outpath)
    #save normalized trajectories
    O.to_csv(outpath + 'normalized_trajectories_res-%04dmm.csv' %(int(res*1000)), index=False)

def plot_normalized_trajectories_grand_average(PATH, res):
    # input folder
    inpath = PATH + 'trajectories/res-%04dmm/' % (int(res * 1000))
    inpathfilename = inpath + 'normalized_trajectories_res-%04dmm.csv' % (int(res * 1000))
    #only continue if inputdata exists
    if os.path.isfile(inpathfilename):
        # load trajectories normalized by progress
        T = pd.read_csv(inpathfilename)
        ##========================
        ##Group average trajectories
        # loop over the tracks
        for track in ['flat', 'wave']:
            # load gate information
            gate_infos = pd.read_csv('/'.join(PATH.split('/')[:-2]) + '/s024/tracks/{}.csv'.format(track))
            gate_corners = [Gate(gate_infos.iloc[i], width=3.7, height=3.7).corners for i in range(gate_infos.shape[0])]
            # some relevant variables
            progress = np.sort(T.progress.unique())
            subjects = np.sort(T.subject.unique())
            # Position: progress x subject x axis
            P = np.empty((progress.shape[0], subjects.shape[0], 3))
            P[:] = np.nan
            # Norm velocity: progress x subject x 1
            V = np.empty((progress.shape[0], subjects.shape[0], 1))
            V[:] = np.nan
            # loop over subjects and progress variables
            for isubj in range(subjects.shape[0]):
                for iprog in range(progress.shape[0]):
                    # select relevant data
                    ind = ((T.subject.values == subjects[isubj]) &
                           (T.track.values == track) &
                           (T.lap.values != 0) &  # exclude the first lab because of podium start
                           (T.progress.values == progress[iprog]))
                    # get current values
                    curr_velocity = T.loc[ind, ('norm_velocity')].values
                    curr_position = T.loc[ind, ('position_x', 'position_y', 'position_z')].values
                    # only proceed if non-empty slice was found
                    if np.sum(np.isnan(curr_velocity)) < curr_velocity.shape[0]:
                        P[iprog, isubj, :] = np.nanmean(curr_position, axis=0)
                        V[iprog, isubj] = np.nanmean(curr_velocity, axis=0)
            # grand average position: progress x axis
            MP = np.squeeze(np.nanmedian(P, axis=1))
            # grand average norm velocity: progress
            MV = np.squeeze(np.nanmedian(V, axis=1))
            # loop over different views on the trajectory
            for view in ['x', 'y', 'z']:
                if view == 'z':
                    ax_labels = ['Position X [m]', 'Position Y [m]']
                    ax_idx = [0, 1]
                    ax_lims = [(-30, 30), (-15, 15)]
                elif view == 'y':
                    ax_labels = ['Position X [m]', 'Position Z [m]']
                    ax_idx = [0, 2]
                    ax_lims = [(-30, 30), (-13, 17)]
                elif view == 'x':
                    ax_labels = ['Position Y [m]', 'Position Z [m]']
                    ax_idx = [1, 2]
                    ax_lims = [(-15, 15), (-5.5, 9.5)]
                # make group average figure
                fig = plt.figure(figsize=(12, 5))
                ax = plt.subplot(111)
                # make a multicolor plot, use norm velocity as coloring variable
                multicolor_line(MP[:, ax_idx[0]], MP[:, ax_idx[1]], MV, fig=fig, ax=ax, linewidth=12, cmap='cividis',
                                clims=None)
                # plt.plot(np.squeeze(MP[:, 0]), np.squeeze(MP[:, 1]), 'r')
                for corner in gate_corners:
                    ax.plot(corner[ax_idx[0], :], corner[ax_idx[1], :], 'k', linewidth=6)
                ax.set_xlabel(ax_labels[0])
                ax.set_ylabel(ax_labels[1])
                ax.set_xlim(ax_lims[0])
                ax.set_ylim(ax_lims[1])
                #make outpath
                outpath = inpath + 'track-{}/'.format(track)
                if os.path.isdir(outpath) == False:
                    make_path(outpath)
                #save the figure
                plt.savefig(outpath + 'grand_average_trajectories_track-{}'.format(track) + '_view-{}'.format(
                    view[0]) + '_res-%04dmm.jpg' % (int(res * 1000)))

def plot_grand_average_drone_state(PATH, res, method='median'):
    # input folder
    inpath = PATH + 'trajectories/res-%04dmm/' % (int(res * 1000))
    inpathfilename = inpath + 'normalized_trajectories_res-%04dmm.csv' % (int(res * 1000))
    #only continue if inputdata exists
    if os.path.isfile(inpathfilename):
        # load trajectories normalized by progress
        T = pd.read_csv(inpathfilename)
        # correct for double labels for start finish gate 0 and 10
        # Note: gates count from 0=start to 10=finish
        ind = (T.progress < 5.) & (T.gates == 10.)
        T.loc[ind, ('gates')] = 0.
        # header = ['position_x', 'position_y', 'position_z',
        #        'deviation_shortest_path_x', 'deviation_shortest_path_y',
        #        'deviation_shortest_path_z', 'norm_deviation_shortest_path',
        #        'velocity_x', 'velocity_y', 'velocity_z', 'norm_velocity',
        #        'acceleration_x', 'acceleration_y', 'acceleration_z',
        #        'norm_acceleration', 'angularvelocity_x', 'angularvelocity_y',
        #        'angularvelocity2_z', 'norm_angularvelocity', 'throttle', 'roll',
        #        'pitch', 'yaw']
        header = ['norm_velocity', 'norm_acceleration', 'throttle', 'angularvelocity_x', 'angularvelocity2_z', 'angularvelocity_y']
        num_vars = len(header)

        progress = np.sort(T.progress.unique())
        num_bins = progress.shape[0]

        subjects = T.subject.unique()

        # loop over the tracks
        for track in ['flat', 'wave']:
            # collect subject data
            subject_average = np.empty((0, num_vars, num_bins))
            for subject in subjects:
                # collect trial data
                trial_data = np.empty((0, num_vars, num_bins))
                for run in T.run.unique():
                    for lap in T.lap.unique():
                        # pointer to current data
                        ind = ((T.subject == subject) & (T.track == track) &
                               (T.run == run) & (T.lap == lap) & (T.lap > 0))
                        # only proceed if at least some data was found
                        if np.sum(ind) > 2:
                            # velocity
                            trial_data = np.vstack((trial_data,
                                                    T.loc[ind, (header)].values.T.reshape((1, num_vars, num_bins))))
                # make subject averages
                if method == 'median':
                    vals = np.nanmedian(trial_data, axis=0).reshape((1, num_vars, num_bins))
                else:
                    vals = np.nanmean(trial_data, axis=0).reshape((1, num_vars, num_bins))
                subject_average = np.vstack((subject_average, vals))
            # compute group average
            grand_average = np.empty((0, num_vars, num_bins))
            if method == 'median':
                grand_average = np.nanmedian(subject_average, axis=0)
            else:
                grand_average = np.nanmean(subject_average, axis=0)
            ## plot the figure
            fig, axs = plt.subplots(num_vars, 1)
            fig.set_figwidth(20)
            fig.set_figheight(num_vars * 2)
            # sortindex
            idx = np.arange(0, progress.shape[0], 1)
            tlims = (np.nanmedian(T.loc[((T.gates == 1) & (T.track==track)), ('progress')].values),
                     np.nanmedian(T.loc[((T.gates == 10) & (T.track == track)), ('progress')].values))
            sort_idx = np.append(np.append(idx[(progress > tlims[0]) & (progress <= tlims[1])],
                                           idx[(progress <= tlims[0])]),
                                           idx[(progress > tlims[1])])
            # make subject & group average plot
            for i in range(num_vars):
                #set the axis limits
                if (header[i] == 'throttle'):
                    ylims = (0, 1)
                elif (header[i] == 'roll') or (header[i] == 'pitch') or (header[i] == 'yaw'):
                    ylims = (-0.5, 0.5)
                elif (header[i] == 'angularvelocity_x'):
                    ylims = (-np.pi, np.pi)
                elif (header[i] == 'angularvelocity_y') or (header[i] == 'angularvelocity2_z'):
                    ylims = (-0.75, 0.75)
                else:
                    ylims = (np.nanmin(np.squeeze(subject_average[:, i, :]).flatten()),
                             np.nanmax(np.squeeze(subject_average[:, i, :]).flatten()))
                xlims = (0., tlims[1])
                # subject traces
                for j in range(len(subjects)):
                    axs[i].plot(progress,
                                np.squeeze(subject_average[j, i, sort_idx]),
                                '0.75', linewidth=3)  # gray
                # group average traces
                axs[i].plot(progress,
                            grand_average[i, sort_idx],
                            'r', linewidth=6)
                # gate positions
                for gate in np.arange(0, 11, 1):
                    _p = np.nanmedian(T.loc[(((T.gates == gate) & (T.track==track))), ('progress')].values)
                    _progress = progress[sort_idx]
                    imin = np.nanargmin(np.abs(_progress - _p))
                    _p = progress[imin]
                    axs[i].plot([_p, _p], ylims, 'k')
                # set plot title
                axs[i].set_ylabel(header[i])
                axs[i].set_ylim(ylims)
                axs[i].set_xlim(xlims)

            #save the figure
            outfilepath = inpath + 'track-{}/'.format(track) + 'grand_average_drone_state_track-{}'.format(track) + '_method-{}'.format(
                method) + '_res-%04dmm.jpg' % (int(res * 1000))
            print('..saving ' + outfilepath)
            plt.savefig(outfilepath)