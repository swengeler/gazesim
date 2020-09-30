from ffprobe import FFProbe
from shutil import copyfile

try:
    from src.functions.pupil_utils import *
except:
    from functions.pupil_utils import *

try:
    from src.functions.laptracker_utils import *
except:
    from functions.laptracker_utils import *

try:
    from src.functions.TimestampFixer import *
except:
    from functions.TimestampFixer import *

def make_path(path):
    outpath = '/'
    folders = path.split('/')
    for fold in folders:
        if len(fold)>0:
            outpath += fold +'/'
            if os.path.isdir(outpath) == False:
                os.mkdir(outpath)

def plot_pose(P, R, ax=None, l=1.):
  '''
  plot_pose(P,R,ax=None,l=1.)

  Makes a 3d matplotlib plot of poses showing the axes direction:
  x=red, y=green, z=blue

  P : np.ndarray, position [poses x axes]
  R : np.ndarray, rotation [poses x euler angles]
  '''
  # make a figure if no axis was provided
  if ax is None:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
  # make sure input data is 2D, where axes are in the 2nd dimension
  P = P.reshape((-1, 3))
  R = R.reshape((-1, 3))
  # loop over poses
  for i in range(P.shape[0]):
    # current position
    p0 = P[i, :]
    # loop over dimensions and plot the axes
    for dim, col in zip(np.arange(0, 3, 1), ['r', 'g', 'b']):
      u = np.zeros((1, 3))
      u[0, dim] = 1.
      v = Rotation.from_euler('xyz', R[i, :]).apply(u)[0]
      p1 = p0 + l * v
      ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=col)

def transform_raw_to_world_frame(df):
    '''
    pose_zdown_to_zup(df)

    Converts Alphapilot Airr-Sim dronedata.csv raw data [x=forward, y=right, z=down]
    to z-is-up coordinates [x=forward, y=left, z=up]

    df: pandas dataframe
    - contains the columns PositionX, PositionY, PositionZ
    - contains the columns RotationX, RotationY, RotationZ
    '''

    is_dronedata = False
    for name in df.columns:
        if name=='PositionX':
            is_dronedata = True
    #if the dataframe contains drone data..
    if is_dronedata:
        #fix position
        vals = df.loc[:, ('PositionX', 'PositionY', 'PositionZ')].values
        tf = Rotation.from_euler('x', [180], degrees=True)
        df[['PositionX', 'PositionY', 'PositionZ']] = tf.apply(vals)
        #fix rotation
        vals = df.loc[:, ('RotationX', 'RotationY', 'RotationZ')].values
        tf = Rotation.from_euler('x', [180], degrees=True)
        df[['RotationX', 'RotationY', 'RotationZ']] = (tf * Rotation.from_euler('xyz', vals) * tf).as_euler('xyz') #as euler angles [rad]
        for name in ['rot_x_quat', 'rot_y_quat', 'rot_z_quat', 'rot_w_quat']:
            df[name] = np.nan
        df[['rot_x_quat', 'rot_y_quat', 'rot_z_quat', 'rot_w_quat']] = (tf * Rotation.from_euler('xyz', vals) * tf).as_quat() #as quaternion
        #fix velocity
        vals = df.loc[:, ('VelocityX', 'VelocityY', 'VelocityZ')].values
        tf = Rotation.from_euler('x', [180], degrees=True)
        df[['VelocityX', 'VelocityY', 'VelocityZ']] = tf.apply(vals)
        # fix angular
        vals = df.loc[:, ('AngularX', 'AngularY', 'AngularZ')].values
        tf = Rotation.from_euler('yx', [90, 180], degrees=True)
        df[['AngularX', 'AngularY', 'AngularZ']] = tf.apply(vals)
        #fix gyroscope
        df['GyroY'] = -df['GyroY'].values
        #fix accelerometer
        vals = df.loc[:, ('AccX', 'AccY', 'AccZ')].values
        tf = Rotation.from_euler('z', [180], degrees=True)
        df[['AccX', 'AccY', 'AccZ']] = tf.apply(vals)
        #fix control commands
        df['Throttle'] =  ((df['Throttle'].values / 992.) + 1.) / 2. #[0,1] range
        for name in ['Roll', 'Pitch', 'Yaw']:
            df[name] = df[name] / 992. #[-1,1] range
    #if dataframe contains track information..
    else:
        # fix position
        vals = df.loc[:, ('pos_x', 'pos_y', 'pos_z')].values
        tf = Rotation.from_euler('x', [180], degrees=True)
        df[['pos_x', 'pos_y', 'pos_z']] = tf.apply(vals)
        #fix rotation
        vals = df.loc[:, ('rot_x_rad', 'rot_y_rad', 'rot_z_rad')].values
        tf = Rotation.from_euler('x', [180], degrees=True)
        df[['rot_x_rad', 'rot_y_rad', 'rot_z_rad']] = (tf * Rotation.from_euler('xyz', vals) * tf).as_euler('xyz') #as euler angles [rad]
        df[['rot_x_deg', 'rot_y_deg', 'rot_z_deg']] = (tf * Rotation.from_euler('xyz', vals) * tf).as_euler('xyz', degrees=True)  #as euler angles [deg]
        df[['rot_x_quat', 'rot_y_quat', 'rot_z_quat', 'rot_w_quat']] = (tf * Rotation.from_euler('xyz', vals) * tf).as_quat()  #as quaternion
    return df

def track_from_json(fpn):
    '''
    #load track/gate information from
    #/airr_Data/StreamingAssets/Maps/linux/addition-arena/eight_horz_elev.json'
    #note that z-axis is directed upward and floor is at 0.0
    '''
    world = fpn.split('.')[0].split('/')[-2]
    track = fpn.split('.')[0].split('/')[-1]
    df = pd.DataFrame()
    a = {}
    with open(fpn) as f:
        lis = [line.split() for line in f]
        for i, x in enumerate(lis):
            if len(x)>0 and x[0].find('name') !=-1:
                if len(a)>0 and a['name'].find('gate-airr')!=-1:
                    #note the order in the json file is 'yzx', so flip it here to 'xyz'
                    pos = [a['pos'][2], a['pos'][0], a['pos'][1]]
                    quat = [a['rot'][2], a['rot'][0], a['rot'][1], a['rot'][3]]
                    r = Rotation.from_quat(quat)
                    euler = r.as_euler('xyz', degrees=True)
                    rad = r.as_euler('xyz', degrees=False)
                    dim = [a['dim'][2], a['dim'][0], a['dim'][1]]

                    df = df.append(pd.DataFrame({
                        'world': world,
                        'track': track,
                        'pos_x': [pos[0]],
                        'pos_y': [pos[1]],
                        'pos_z': [-(pos[2]+1.75)], #flip z-axis and move floor to 1.75 meters
                        'rot_x_quat': [quat[0]],
                        'rot_y_quat': [quat[1]],
                        'rot_z_quat': [quat[2]],
                        'rot_w_quat': [quat[3]],
                        'rot_x_rad': [rad[0]],
                        'rot_y_rad': [rad[1]],
                        'rot_z_rad': [rad[2]],
                        'rot_x_deg': [euler[0]],
                        'rot_y_deg': [euler[1]],
                        'rot_z_deg': [euler[2]],
                        'dim_x': [dim[0]],
                        'dim_y': [dim[1]],
                        'dim_z': [dim[2]]
                        }), ignore_index=True)
                a = {}
                a['name'] = x[0].split('"')[-2]
            if len(x)>0 and x[0].find('local-position') !=-1:
                a['pos'] = np.fromstring(''.join(x).split('":[')[-1].split(']')[0], dtype=float, sep=',')
            if len(x)>0 and x[0].find('local-rotation') !=-1:
                a['rot'] = np.fromstring(''.join(x).split('":[')[-1].split(']')[0], dtype=float, sep=',')
            if len(x)>0 and x[0].find('local-scale') !=-1:
                a['dim'] = np.fromstring(''.join(x).split('":[')[-1].split(']')[0], dtype=float, sep=',')
    df = df.reset_index().drop(columns=['index'], axis=1)
    return df

def adjust_gate_orientation(track, orientation_adjustment):
    r_deg = track.loc[:, ('rot_x_deg', 'rot_y_deg', 'rot_z_deg')].values
    r_rad = track.loc[:, ('rot_x_rad', 'rot_y_rad', 'rot_z_rad')].values
    r_quat = track.loc[:, ('rot_x_quat', 'rot_y_quat', 'rot_z_quat', 'rot_w_quat')].values
    for i in range(len(orientation_adjustment)):
        if orientation_adjustment[i] == 1:
            r = Rotation.from_quat(r_quat[i, :]) * Rotation.from_euler('z', np.pi)
            r_deg[i, :] = r.as_euler('xyz', degrees=True).flatten()
            r_rad[i, :] = r.as_euler('xyz').flatten()
            r_quat[i, :] = r.as_quat()
    track['rot_x_deg'] = r_deg[:,0]
    track['rot_y_deg'] = r_deg[:,1]
    track['rot_z_deg'] = r_deg[:,2]
    track['rot_x_rad'] = r_rad[:, 0]
    track['rot_y_rad'] = r_rad[:, 1]
    track['rot_z_rad'] = r_rad[:, 2]
    track['rot_x_quat'] = r_quat[:, 0]
    track['rot_y_quat'] = r_quat[:, 1]
    track['rot_z_quat'] = r_quat[:, 2]
    track['rot_w_quat'] = r_quat[:, 3]
    return track

def load_tracks(PATH):
    folder = PATH+'airr-sim/airr_Data/StreamingAssets/Maps/linux/addition-arena/'
    files = {'hairpin':['slalom.json', [3,9], [0,0]],
             'slalom': ['slalom.json', [11,0,1,2,3,4,5,6,7,8,9,10], [0,0,0,0,0,0,0,0,0,0,0,0]],
             'splitslarge': ['splits.json', [0,2,5,3], [0,0,0,0]],
             'splitssmall': ['splits.json', [0,1,4,3], [0,0,0,0]],
             'flat': ['flat.json', [5,0,7,8,9,6,1,4,3,2], [1,1,1,0,0,0,0,0,0,1]],
             'wave': ['wave.json', [5,0,7,8,9,6,1,4,3,2], [1,1,1,0,0,0,0,0,0,1]]}
    tracks = {}
    for key, value in files.items():
        filename = value[0]
        gate_order = value[1]
        orientation_adjustment = value[2]
        track = track_from_json(folder+filename).loc[gate_order].reset_index().drop(columns=['index'])
        track['gate_id']=np.arange(1, track.shape[0]+1, 1) #start counting from 1
        track = transform_raw_to_world_frame(track)
        track = adjust_gate_orientation(track, orientation_adjustment)
        tracks[key]=track
    return tracks

def get_pupil_timestamps(PATH, clocktime_offset=0., timezone='Europe/Paris'):
    pupilTimestamps = pd.DataFrame()
    pupilpaths = []
    for x in os.walk(PATH):
        for f in x[2]:
            if f=='export_info.csv':
                pupilpaths.append(x[0]+'/')
    for pupilpath in sorted(pupilpaths):
        infoPlayer = {}
        path = '/'.join(pupilpath.split('/')[:-3])+'/info.player.json'
        df = pd.read_csv(path,delimiter='\n',header=None)
        for i in range(1,df.shape[0]-1):
            s=df.iloc[i].values[0].split(':')
            key=s[0]
            value=' '.join(s[1:])
            infoPlayer[''.join(key.strip().split('"'))]=''.join((''.join(value.strip().split(',')).split('"')))
        startTimeSync = np.float(infoPlayer['start_time_synced_s']) #pupiltime
        startTimeSystem = np.float(infoPlayer['start_time_system_s']) + clocktime_offset #system time (Europe/Paris time)
        deltaTime = startTimeSystem - startTimeSync
        exportInfo = {}
        path = pupilpath+'export_info.csv'
        df = pd.read_csv(path,delimiter='\n',header=None)
        for i in range(1,df.shape[0]):
            key,value=df.iloc[i].values[0].split(',')
            exportInfo[key]=value
        f0=int(exportInfo['Frame Index Range:'].split(' ')[0])
        f1=int(exportInfo['Frame Index Range:'].split(' ')[-1])
        absTimeRange = (np.float(exportInfo['Absolute Time Range'].split(',')[-1].split(' ')[0]), \
                        np.float(exportInfo['Absolute Time Range'].split(',')[-1].split(' ')[2]))
        ts0 = pd.Timestamp(absTimeRange[0]+deltaTime, unit='s', tz=timezone)
        ts1 = pd.Timestamp(absTimeRange[1]+deltaTime, unit='s', tz=timezone)
        df = pd.DataFrame({'ts_start': ts0, 'ts_end': ts1, 'filename': 'world.mp4', 'filepath': pupilpath, \
                           'frame_start': f0, 'frame_end': f1}, index=[0])
        if pupilTimestamps.size:
            pupilTimestamps = pupilTimestamps.append(df, ignore_index=True)
        else:
            pupilTimestamps = df
    return pupilTimestamps

def get_dronedata_timestamps(PATH):
    ## dronedata timestamps
    droneTimestamps = pd.DataFrame()
    for x in os.walk(PATH):
        if x[0]==PATH:
            for file in sorted(x[2]):
                if file.find('dronedata')!=-1:
    #                 ymd=[int(val) for val in file.split('.')[0].split('_')[1].split('-')]
    #                 hms=[int(val) for val in file.split('.')[0].split('_')[2].split('-')]
    #                 ts_start = pd.Timestamp(ymd[0], ymd[1], ymd[2], hms[0], hms[1], hms[2])
                    with open(PATH+file, "r") as f:
                        lines = f.readlines()
                        ts0=pd.Timestamp(np.float(lines[1].split(',')[0]), unit='s', tz='Europe/Paris')
                        ts1=pd.Timestamp(np.float(lines[-1].split(',')[0]), unit='s', tz='Europe/Paris')
                    df = pd.DataFrame({'ts_start': ts0, 'ts_end': ts1, 'filename': file, 'filepath': PATH},index=[0])
                    if droneTimestamps.size:
                        droneTimestamps = droneTimestamps.append(df,ignore_index=True)
                    else:
                        droneTimestamps = df
    return droneTimestamps

def get_tracknames_from_dronedata(droneTimestamps, tracks, thresh=1.75):
    '''
    Rule-based assignment of tracknames to dronedata files based on drone and gate position data
    Gate passes (hits) are detected for drone position within a givent threshold thresh)
    Comparing the number of hits across different tracks there is a rule-based selection of tracks
    1. check if wave or flat track [track without zero passes]
    2. check if slalom
    3. check if hairpin [if no slalom yet]
    4. check if splits [if slalom was done already], and which splits [max number of gate passes]
    Note that each track has an individual selection of gates relevant for the track
    '''
    final_track_selection = []
    for j in range(droneTimestamps.shape[0]):
        droneTimestamp = droneTimestamps.iloc[j]
        df = pd.read_csv(droneTimestamp.filepath+droneTimestamp.filename)
        df = transform_raw_to_world_frame(df)
        p = df[['PositionX','PositionY','PositionZ']].values
        track_passes = {}
        for key,t in tracks.items():
            hit_list = []
            for i in range(t.shape[0]):
                g = t[['pos_x','pos_y','pos_z']].iloc[i].values.reshape((1,3))
                dist = np.linalg.norm(p-g,axis=1)
                hits=np.sum(np.diff((dist<thresh).astype(np.float))>0)
                hit_list.append(hits)
            track_passes[key]=hit_list
        track_name = []
        #first check if flat or wave track
        for name in ['flat', 'wave']:
            if np.sum(np.array(track_passes[name])==0)==0:
                track_name.append(name)
        #if no flat/wave, check which maneuver it is
        if len(track_name)==0:
            p_hair = np.array(track_passes['hairpin'])
            p_slalom = np.array(track_passes['slalom'])
            p_splitssmall = np.array(track_passes['splitssmall'])
            p_splitslarge = np.array(track_passes['splitslarge'])
            if (np.sum(p_slalom==0)<4):
                track_name.append('slalom')
            else:
                before_slalom = True
                for name in final_track_selection:
                    if name=='slalom':
                        before_slalom=False
                if before_slalom:
                    track_name.append('hairpin')
                else:
                    if np.sum(p_splitssmall)>np.sum(p_splitslarge):
                        track_name.append('splitssmall')
                    else:
                        track_name.append('splitslarge')
        final_track_selection.append(track_name[0])
    return final_track_selection

def get_screen_timestamps(PATH, timezone='Europe/Paris'):
    screenTimestamps = pd.DataFrame()
    for walker in os.walk(PATH):
        for filename in sorted(walker[2]):
            if filename.find('Kazam_screencast') != -1:
                filepath = walker[0]
                metadata = FFProbe(filepath + filename)
                ts_start = pd.Timestamp(metadata.metadata['creation_time']).tz_convert(timezone)
                HMS = [float(val) for val in metadata.metadata['Duration'].split(':')]
                duration_in_sec = 360 * HMS[0] + 60 * HMS[1] + HMS[2]
                ts_end = pd.Timestamp(ts_start.timestamp() + duration_in_sec, unit='s', tz=timezone)
                cap = cv2.VideoCapture(filepath+filename)
                f0 = 0
                f1 = int(cap.get(7))
                fps = cap.get(5)
                df = pd.DataFrame({'ts_start': ts_start, 'ts_end': ts_end, 'filename': filename, 'filepath': filepath, \
                                   'frame_start': f0, 'frame_end': f1, 'fps': fps}, index=[0])
                if screenTimestamps.size:
                    screenTimestamps = screenTimestamps.append(df,ignore_index=True)
                else:
                    screenTimestamps = df
    return screenTimestamps

def crop_video(inpath, outpath, firstframe, lastframe):
    cap = cv2.VideoCapture(inpath)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(5)
    # fourcc = int(cap.get(6))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    frames = int(cap.get(7))
    writer = cv2.VideoWriter(outpath, fourcc, fps, (width, height))
    if firstframe<0:
      firstframe=0
    if lastframe>frames:
      lastframe=frames
    for f in np.arange(int(firstframe), int(lastframe)+1, 1):
        print('..exporting {} [{:.2f} %]'.format(outpath, 100*f/lastframe), end='\r')
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        _, im = cap.read()
        writer.write(im)
    print('')
    writer.release()

def video_to_video_with_overlay(inpath_world, inpath_gaze, outpath_world, ts_index='ts', frame_index='frame',
                                x_index='norm_pos_x', y_index='norm_pos_y', gaze_is_normalized=True, max_frames=None):
    cap = cv2.VideoCapture(inpath_world)
    gaze = pd.read_csv(inpath_gaze)
    if gaze[frame_index].iloc[0] > 0:
        gaze[frame_index] = gaze[frame_index] - gaze[frame_index].iloc[0]
    make_path('/'.join(outpath_world.split('/')[:-1]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, image = cap.read()
    #get some important information about the video
    width = cap.get(3)      #image width
    height = cap.get(4)     #image height
    fps = cap.get(5)        #frames per second
    fourcc = cap.get(6)     #fourcc codec integer identifier
    num_frames = cap.get(7) #number of frames
    is_color = True            #whether to use color
    if max_frames is None:
        max_frames = int(num_frames)
    df = gaze.copy()
    df = df[[ts_index, frame_index, x_index, y_index]]
    df.columns = ['ts','frame','x','y']
    if gaze_is_normalized:
        df['x'] = df['x'] * width
        df['y'] = (1.0-df['y']) * height
    #video writer objecgt
    vidwriter = cv2.VideoWriter(
        outpath_world,
        int(fourcc),
        fps,
        (int(width),int(height)),
        is_color)
    #loop over video frames
    for frame_vid in range(max_frames+1):
    # for frame_vid in range(int(num_frames)):
        #read current frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_vid)
        success, image = cap.read()
        if success:
            #find gaze positions for this frame
            x = df[df['frame']==frame_vid]['x'].values
            y = df[df['frame']==frame_vid]['y'].values
            image_modified=image.copy()
            print('..exporting {} [{:.2f} %]'.format(outpath_world, 100 * frame_vid / (max_frames-1)), end='\r')
            if len(x)>0:
                #plot overlay
                #copy the raw image and draw an overlay
                for x_,y_ in zip(x,y):
                    if not np.isnan(x_):
                        cv2.circle(image_modified, (int(x_), int(y_)), 5, (255,0,0), -1)
            #write frame to output video
            vidwriter.write(image_modified)
    print()
    #close the video writer
    vidwriter.release()

def load_world_timestamps_fix_timestamps(PATH):
    #raw world_timestamps.csv : assume timestamp is false and world_index (i.e., dataframe index) is correct
    W = pd.read_csv(PATH).reset_index()
    W.columns = ['world_index', 'world_timestamp', 'pts']
    ts_video_start = W.world_timestamp.iloc[0]
    duration = W.world_timestamp.iloc[-1] - W.world_timestamp.iloc[0]
    frames = W.shape[0] - 1
    period = duration / frames
    W.world_timestamp = np.arange(0, W.shape[0], 1) * period + ts_video_start
    return W, period

def fetch_frame_by_timestamp(ts,cap,t,f):
    frame = f[t<=ts][-1]
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    _,im = cap.read()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

def fix_screen_timestamps(PATH, datatype='screen', plot_cost=False, plot_final=False):
    '''
    fix_screen_timestamps(PATH, plot_cost=False, plot_final=False)

    Fixes the 'ts' in screen_timestamps.csv, such that it synchronizes with 'ts' in drone.csv

    This is done by detecting gate passing events from drone.csv
    and then iteratively searching for the gate passing event trigger in the video frames [i.e. grey horizontal bar at
    the bottom of the screen]
    Specifically, the grey bar event trigger has homogenous color. whereas scene information shows more variance
    Thus. frames for all gate passing events are retrieved, a cost is computed based on image data (i.e. sum of mean of
    standard deviations across lower screen pixel information)
    Then the timestamp with the lowest cost is selected
    and the screen_timestamp.csv is updated and saved to the data folder

    PATH : string
    - path to data

    plot_cost : bool
    - whether to plot the cost and selected timestamp for each iteration

    plot_final : bool
    - whether to plot gate passing event frames after timestamp correction (for visual check if all worked well)

    Christian Pfeiffer
    27.06.2020
    '''
    print('=======================')
    print('PATH: {}'.format(PATH))
    print('Datatype: {}'.format(datatype))
    #read data
    print('..loading input data', end='\r')
    D = pd.read_csv(PATH+'drone.csv')
    T = pd.read_csv(PATH+'track.csv')
    if datatype == 'surface':
        S = pd.read_csv(PATH + 'world_timestamps.csv')
        cap = cv2.VideoCapture(PATH + 'surface.mp4')
    else:
        S = pd.read_csv(PATH + 'screen_timestamps.csv')
        cap = cv2.VideoCapture(PATH + 'screen.mp4')
    #Detect gate passing events
    print('..detecting gate passing events', end='\r')
    ts_ = D['ts'].values
    pos_ = D[['PositionX','PositionY','PositionZ']].values
    #only continue with some drone data available
    if ts_.size:
        gates = []
        events = []
        ts_events = np.array([])
        for i in range(T.shape[0]):
            gates.append(Gate(T.iloc[i]))
            events.append(detect_gate_passing(ts_, pos_, T.iloc[i]))
            if ts_events.size:
                ts_events = np.hstack((ts_events, events[-1]))
            else:
                ts_events = events[-1]
        print('..gate passing detected: {} events'.format(ts_events.shape[0]))
        #only continue processing if gate passing events were detected
        if ts_events.size:
            #apply initial correction of timestamps, set first frame to ts=0.0sec
            S.ts -= S.ts.iloc[0]
            #Iterative optimization to find adjustment time for timestamps
            fps = np.round(cap.get(5))  #frame rate of the video stream
            steps = fps * 1.25   #total number of steps per iteration
            center = 0.  #starting guess for the compensation time
            if datatype == 'surface':
                frame_spacings = [0.5, 0.15]
            else:
                frame_spacings = [3., 0.15] #how large the distance between sampled frames
            time = S.copy().ts.values
            iterations = np.arange(0,len(frame_spacings),1)
            for iter in iterations: #number of items per list determines number of iterations
                frame_spacing = frame_spacings[iter]
                precision = frame_spacing / fps
                width = precision*(steps/(iter+1))
                ts = np.arange(center-width/2, center+width/2, precision)
                C = []
                count = 0
                for ts_ in ts:
                    count+=1
                    # print()
                    time_ = time + ts_
                    #only consider timestamp shifts where that include all gate passing event timestamps
                    if (np.min(time) <= np.min(ts_events)) and (np.max(time) >= np.max(ts_events)):
                        # frames = np.array([A.iloc[np.argmin(np.abs(A.ts.values - val))].frame.astype(int) for val in ts_events])
                        frames = np.array([S.loc[time_ <= val].frame.iloc[-1].astype(int) for val in ts_events])
                        cost_per_event = []
                        for i in range(ts_events.shape[0]):
                            f_ = frames[i]
                            t_ = ts_events[i]
                            im = fetch_frame_by_timestamp(t_, cap, time_, S.frame.values)
                            #compute cost: take the lower part of the image and compute standard deviation across all pixels, then
                            #compute the mean across image channels, append this cost to an array for all events
                            if datatype == 'surface':
                                cost_per_frame = -np.median(np.median(im[520:580, 20:780, :].reshape(-1, 3), axis=0).flatten())
                            else:
                                cost_per_frame = np.mean(np.std(im[520:, :, :].reshape((-1, 3)), axis=0))
                            cost_per_event.append(cost_per_frame)
                        if datatype == 'surface':
                            cost_all_events = np.median(np.array(cost_per_event))
                        else:
                            cost_all_events = np.sum(np.array(cost_per_event))
                    else:
                        cost_all_events = np.nan
                    # print('center={:.8f}, width={:.4f}, precision={:.4f}, ts={:.4f}, cost={:.4f}'.format(center, width, \
                    # precision, ts_, cost_all_events))
                    print('..optimizing timeshift: iter={}/{} [{:.2f} %]: ts={:.4f}, cost={:.4f}'.format(iter+1, len(iterations),
                         100 * count / (ts.shape[0]), ts_, cost_all_events), end='\r')
                    C.append(cost_all_events)
                print('', end='\r')
                C = np.array(C)
                if plot_cost:
                    ncols = np.round(np.sqrt(ts_events.shape[0])).astype(int)
                    nrows = np.ceil(np.sqrt(ts_events.shape[0])).astype(int)
                    plt.figure(figsize=(15,5))
                    plt.plot(ts, C)
                    plt.plot(ts[np.argmin(C)], C[np.argmin(C)], 'ro')
                    plt.title('Selection: ts={:.6f}, cost={:.6f}'.format(ts[np.argmin(C)], C[np.argmin(C)]))
                    plt.xlabel('Timeshift [sec]')
                    plt.ylabel('Cost [sum mean std]')
                    plt.show()
                center = ts[np.argmin(C)]
                total_cost =  C[np.argmin(C)]
            print('..final selection: ts={:.6f}, cost={:.6f}'.format(center, total_cost))
            #plot the final adjustment time selection
            if plot_final:
                ts_adj = center
                # videopath = PATH+'screen.mp4'
                A = S.copy()
                # cap = cv2.VideoCapture(videopath)
                plt.figure(figsize=(15,15))
                A.ts += ts_adj
                frames = np.array([ A[A.ts<=val].frame.iloc[-1].astype(int) for val in ts_events])
                V = []
                for i in range(ts_events.shape[0]):
                    f_ = frames[i]
                    t_ = ts_events[i]
                    im = fetch_frame_by_timestamp(t_, cap, A.ts.values, A.frame.values)
                    V.append(np.mean(np.std(im[520:, :, :].reshape((-1, 3)), axis=0)))
                    plt.subplot(nrows, ncols, i+1)
                    plt.imshow(im)
                    plt.xticks([])
                    plt.yticks([])
                plt.show()
            #update and save screen timestamps
            if datatype == 'surface':
                print('skip surface save')
            else:
                S.ts += center
                print('..save {}'.format(PATH+'screen_timestamps.csv'))
                S.to_csv(PATH+'screen_timestamps.csv', index=False)
            print()

def process_subject(PATH, clocktime_offset=3.64, export_files=['all'], runs=None):
    '''
    Process Subject
    Automatically selects from raw data corresponding drone data, pupil data and kazam data files
    Based on timestamps and saves the relevant data to /process folder
    '''
    #load and save track info
    tracks = load_tracks(PATH)
    outpath = '/process/'.join(PATH.split('/raw/')) + 'tracks/'
    make_path(outpath)
    for key, value in tracks.items():
        value.to_csv(outpath + key + '.csv', index=False)
    #load and save timestamp info
    drone = get_dronedata_timestamps(PATH)
    if os.path.isfile(PATH+'tracknames.csv'):
        df = pd.read_csv(PATH+'tracknames.csv')
        drone['track_name'] = df.name
    else:
        drone['track_name'] = get_tracknames_from_dronedata(drone, tracks)
    pupil = get_pupil_timestamps(PATH, clocktime_offset)
    screen = get_screen_timestamps(PATH)
    outpath = '/process/'.join(PATH.split('/raw/')) + 'info/'
    make_path(outpath)
    drone.to_csv(outpath + 'drone.csv', index=False)
    pupil.to_csv(outpath + 'pupil.csv', index=False)
    screen.to_csv(outpath + 'screen.csv', index=False)
    #save data per recording run (loop over available dronedata files)
    if runs is None:
        runs = [val for val in range(drone.shape[0])]
    for run in runs:
        process_run(PATH, run, tracks, drone, pupil, screen, clocktime_offset=clocktime_offset, export_files=export_files)

def process_run(PATH, run, tracks, drone, pupil, screen, clocktime_offset=0., export_files=['all']):
    '''
        Process Run
        Imports the data of the current recording run
    '''
    #dronedata current run
    curr_drone = drone.copy().iloc[run] #dronedata current run
    ts_d0 = curr_drone.ts_start.timestamp() #dronedata start timestamp
    ts_d1 = curr_drone.ts_end.timestamp() #dronedata end timestamp
    ts_dm = ts_d0 + 0.5 * (ts_d1 - ts_d0) #dronedata timestamp in the middle
    #pupildata current run
    ind_p = []
    ts_p0 = [val.timestamp() for val in pupil.ts_start] #pupildata start timestamp
    ts_p1 = [val.timestamp() for val in pupil.ts_end] #pupildata end timestamps
    for j in range(len(ts_p0)):
        ind_p.append((ts_p0[j] <= ts_d0) & (ts_p1[j] >= ts_d1))
    curr_pupil = pupil.copy().iloc[ind_p, :].T.squeeze() #pupildata current run
    if isinstance(curr_pupil, pd.DataFrame):
        if curr_pupil.shape[0] > 1:
            curr_pupil = curr_pupil.iloc[-1]
    #screendata current run
    if screen.ts_start.iloc[0] is not None:
        ind_s = []
        ts_s0 = [val.timestamp() for val in screen.ts_start] #screendata start timestamps
        ts_s1 = [val.timestamp() for val in screen.ts_end] #screendata end timestamps
        for j in range(len(ts_s0)):
            ind_s.append((ts_s0[j] <= ts_dm) & (ts_s1[j] >= ts_dm))
        curr_screen = screen.copy().iloc[ind_s, :].T.squeeze() #screendata current run
    else:
        curr_screen = None
    if isinstance(curr_screen, pd.DataFrame):
        if curr_screen.shape[0] > 1:
            curr_screen = curr_screen.iloc[-1]
    #proceed only if pupil data available
    if curr_pupil.size and curr_screen.size:
        print('--------')
        #make output folder for the current run
        outpath = '/process/'.join(PATH.split('/raw/')) + '%02d' % run + '_' + curr_drone.track_name + '/'
        print('outpath:    {}'.format(outpath))
        print('pupilpath:  {}'.format(curr_pupil.filepath))
        print('dronepath:  {}'.format(curr_drone.filepath + curr_drone.filename))
        print('screenpath: {}'.format(curr_screen.filepath + curr_screen.filename))
        make_path(outpath)
        #save track data
        T = tracks[curr_drone.track_name]
        T.to_csv(outpath + 'track.csv', index=False)
        #save drone data
        D = pd.read_csv(curr_drone.filepath + curr_drone.filename)
        D = transform_raw_to_world_frame(D)
        ts_onset = detect_liftoff(D.CurrTime.values, D[['PositionX', 'PositionY', 'PositionZ']].values)
        D = D.loc[D.CurrTime>=ts_onset]
        D['utc_timestamp'] = D.CurrTime
        D['ts'] = D.utc_timestamp - ts_onset
        D = D[['ts', 'utc_timestamp', 'CurrTime', 'Seconds', 'NanoSeconds',
                 'PositionX', 'PositionY', 'PositionZ',
                 'RotationX', 'RotationY', 'RotationZ',
                 'rot_x_quat', 'rot_y_quat', 'rot_z_quat', 'rot_w_quat',
                 'VelocityX', 'VelocityY', 'VelocityZ',
                 'AngularX', 'AngularY', 'AngularZ',
                 'DroneVelocityX', 'DroneVelocityY', 'DroneVelocityZ',
                 'DroneAccelerationX', 'DroneAccelerationY', 'DroneAccelerationZ',
                 'DroneAngularX', 'DroneAngularY', 'DroneAngularZ',
                 'GyroX', 'GyroY', 'GyroZ',
                 'AccX', 'AccY', 'AccZ',
                 'Throttle', 'Roll', 'Pitch', 'Yaw' ]]
        D.to_csv(outpath + 'drone.csv', index=False)
        #Gate-passing event and lap tracking
        lap_tracking(outpath)
        #compute and add track progress variable to dronedata
        add_track_progress_to_drone_data(outpath)
        #save world timestamps
        W, period = load_world_timestamps_fix_timestamps(curr_pupil.filepath+'world_timestamps.csv')
        ts_pupil0 = W.world_timestamp.iloc[0]
        W['utc_timestamp'] = W.world_timestamp - ts_pupil0 + curr_pupil.ts_start.timestamp()
        ts0 = ts_onset  #timepoint just before quad liftoff in utc timestamp [it works]
        ts1 = D.utc_timestamp.iloc[-1] + clocktime_offset  # timepoint at the end of the Dronedata recording in utc timestamp
                                                           # + the clocktime difference between pupil and dronedata [weird why?]
        print('Start: {}'.format(pd.Timestamp(ts0, unit='s', tz='Europe/Paris')))
        print('End:   {}'.format(pd.Timestamp(ts1, unit='s', tz='Europe/Paris')))
        print('Duration: {:.4f} sec'.format(ts1-ts0))
        f0 = W[W.utc_timestamp >= ts0].world_index.iloc[0] #first frame of video to be exported
        f1 = W[W.utc_timestamp >= ts1].world_index.iloc[0] #last frame of video to be exported
        print('World_index range: {} - {} frames [{} frames]'.format(f0, f1, f1-f0))
        W = W.loc[(W.world_index >= f0) & (W.world_index <= f1)]
        W['ts'] = W.utc_timestamp - ts0
        W['frame'] = W.world_index - f0
        W = W[['ts', 'frame', 'world_timestamp', 'world_index', 'utc_timestamp']]
        W.to_csv(outpath + 'world_timestamps.csv', index=False)
        #save surface positions
        surface_frame_offset = 2
        S = pd.read_csv(curr_pupil.filepath + 'surfaces/surf_positions_Surface 1.csv')
        S.world_index = np.floor((S.world_timestamp.values - ts_pupil0) / period).astype(int) + surface_frame_offset
        S['utc_timestamp'] = S.world_timestamp.values - ts_pupil0 + curr_pupil.ts_start.timestamp()
        S = S.loc[(S.world_index >= f0) & (S.world_index <= f1)]
        S['ts'] = S.utc_timestamp - ts0
        S['frame'] = S.world_index - f0
        for c in [c for c in S.columns if 'trans' in c]:
            S[c] = S[c].apply(transform_string_to_array)
        S = S[['ts', 'frame', 'world_timestamp', 'world_index', 'utc_timestamp', 'img_to_surf_trans',
                 'surf_to_img_trans', 'num_detected_markers', 'dist_img_to_surf_trans',
                 'surf_to_dist_img_trans']]
        S.to_csv(outpath + 'surf_positions.csv', index=False)
        #save gaze data
        gaze_frame_offset = 1
        G = pd.read_csv(curr_pupil.filepath + 'gaze_positions.csv')
        G.world_index = np.floor((G.gaze_timestamp.values - ts_pupil0) / period).astype(int) + gaze_frame_offset
        G['utc_timestamp'] = G.gaze_timestamp.values - ts_pupil0 + curr_pupil.ts_start.timestamp()
        G = G.loc[(G.world_index >= f0) & (G.world_index <= f1)]
        G['ts'] = G.utc_timestamp - ts0
        G['frame'] = G.world_index - f0
        G = G[['ts', 'frame', 'gaze_timestamp', 'world_index', 'utc_timestamp', 'confidence', 'norm_pos_x',
                'norm_pos_y', 'base_data']]
        G.to_csv(outpath + 'gaze_positions.csv', index=False)
        #save pupil data
        P = pd.read_csv(curr_pupil.filepath + 'pupil_positions.csv')
        P.world_index = np.floor((P.pupil_timestamp.values - ts_pupil0) / period).astype(int)
        P['utc_timestamp'] = P.pupil_timestamp.values - ts_pupil0 + curr_pupil.ts_start.timestamp()
        P = P.loc[(P.world_index >= f0) & (P.world_index <= f1)]
        P['ts'] = P.utc_timestamp - ts0
        P['frame'] = P.world_index - f0
        P = P[['ts','frame','pupil_timestamp', 'world_index', 'utc_timestamp', 'eye_id', 'confidence', 'norm_pos_x',
               'norm_pos_y', 'diameter', 'method', 'ellipse_center_x',
               'ellipse_center_y', 'ellipse_axis_a', 'ellipse_axis_b', 'ellipse_angle',
               'diameter_3d', 'model_confidence', 'model_id', 'sphere_center_x',
               'sphere_center_y', 'sphere_center_z', 'sphere_radius',
               'circle_3d_center_x', 'circle_3d_center_y', 'circle_3d_center_z',
               'circle_3d_normal_x', 'circle_3d_normal_y', 'circle_3d_normal_z',
               'circle_3d_radius', 'theta', 'phi', 'projected_sphere_center_x',
               'projected_sphere_center_y', 'projected_sphere_axis_a',
               'projected_sphere_axis_b', 'projected_sphere_angle']]
        P.to_csv(outpath + 'pupil_positions.csv', index=False)
        #save screen timestamps
        period_screen = 1. / curr_screen.fps
        ts_screen = np.arange(curr_screen.ts_start.timestamp(), curr_screen.ts_end.timestamp() + period_screen, period_screen)
        SC = pd.DataFrame({'utc_timestamp': ts_screen, 'screen_index': np.arange(0, ts_screen.shape[0], 1)})
        f0_screen = SC[SC.utc_timestamp >= ts0-clocktime_offset].screen_index.iloc[0]  # first frame of video to be exported
        f1_screen = SC[SC.utc_timestamp >= ts1-clocktime_offset].screen_index.iloc[0]  # last frame of video to be exported
        SC = SC.loc[(SC.screen_index >= f0_screen) & (SC.screen_index <= f1_screen)]
        SC['ts'] = SC.utc_timestamp - ts0
        SC['frame'] = SC.screen_index - f0_screen
        SC = SC[['ts', 'frame', 'screen_index', 'utc_timestamp']]
        SC.to_csv(outpath + 'screen_timestamps.csv', index=False)
        #save world camera intinisics
        inpath = '/'.join(curr_pupil.filepath.split('/')[:-3]) + '/'
        with open(inpath+'world.intrinsics', "rb") as fh:
            world_intrinsics = msgpack.unpack(fh, raw=False)
        width = 800
        height = 600
        K = np.array(world_intrinsics['({}, {})'.format(width, height)]['camera_matrix']).reshape(3, 3)
        D = np.array(world_intrinsics['({}, {})'.format(width, height)]['dist_coefs'])
        copyfile(inpath+'world.intrinsics', outpath+'world.intrinsics')
        #map gaze onto surface and append to gaze data
        if (export_files.count('all')>0) or (export_files.count('gaze_on_surface')>0):
            G_surf = gaze_to_surface_mapping(W, S, G, W.frame.values, K, D, method='index', output='su', width=width,
                                         height=height, colname_index='frame', colname_timestamp='ts')
            G_surf.to_csv(outpath + 'gaze_on_surface.csv', index=False)
        #save world video
        if (export_files.count('all')>0) or (export_files.count('world')>0):
            crop_video(curr_pupil.filepath + 'world.mp4', outpath + 'world.mp4', f0, f1)
        #save world video with surface and gaze overlay
        if (export_files.count('all')>0) or (export_files.count('world_overlay')>0):
            cap = cv2.VideoCapture(outpath + 'world.mp4')
            make_video(cap, W, S, G, W.frame.values, K, D, outpath=outpath + 'world_overlay.mp4', method='index',
                       output='wdo', width=width, height=height, colname_index='frame', colname_timestamp='ts')
        #save surface video with gaze overlay
        if (export_files.count('all')>0) or (export_files.count('surface')>0):
            cap = cv2.VideoCapture(outpath+'world.mp4')
            make_video(cap, W, S, G, W.frame.values, K, D, outpath=outpath+'surface.mp4', method='index',
                       output='su', width=width, height=height, colname_index='frame', colname_timestamp='ts')
        #save screen/kazam video
        if (export_files.count('all')>0) or (export_files.count('screen')>0):
            crop_video(curr_screen.filepath + curr_screen.filename, outpath + 'screen.mp4', f0_screen, f1_screen)
        #fix timestamps
        # based on the pipeline so far, still drone.csv screen_timestamps.csv and world_timestamps.csv timestamps are
        # not perfectly synchronized. Therefore, use image-based detection of gate passing event triggers, and correlation
        # in order to determine the time offsets and correct for them in the ..._timestamps.csv and gaze data files.
        # requires the classes TimestamFixer() and GatePassingEventDetector()
        #only do this for splits, flat and wave tracks, where there are gate passing events
        if (outpath.find('splits') != -1) or (outpath.find('flat') != -1) or (outpath.find('wave') != -1):
            obj = TimestampFixer(outpath)
            obj.run()



