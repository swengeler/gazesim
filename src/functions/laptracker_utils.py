import pandas as pd
try:
    from src.functions.Gate import *
except:
    from functions.Gate import *


def detect_liftoff(t, p, distance_threshold=0.1, prepend_time=2.0, start_position=None):
    if start_position is None:
        start_position = np.nanmedian(p[:1000,:], axis=0)
    dist = np.linalg.norm(p-start_position.reshape((1,3)),axis=1)
    t_above=t[dist>distance_threshold]
    if t_above.size:
        return t[t>=(t_above[0]-prepend_time)][0]
    return t[0]

def detect_gate_passing(t, p, g, thresh=1.):
    gate = Gate(g)
    p_gate = gate.center.reshape((1,3)).astype(float)
    dist = np.linalg.norm(p-p_gate, axis=1)
    ts1 = t[dist<thresh]
    ts2 = []
    for t_ in ts1:
        p_ = p[t >= t_, :]
        if p_.shape[0] > 1:
            _, p3d = gate.intersect(p_[0, :], p_[1, :])
            if p3d is not None:
                ts2.append(t_)
    ts2 = np.array(ts2)
    return ts2

def detect_collision(t, acc, thresh=3):
    '''
    Detects collisions based on accelerometer data, a single event each onset time the accelerometer exceeds threshold
    t : np.array(float) sp x 1
        Timestamps
    acc : np.array(float) sp x 3
        Accelerometer data
    thresh : float
        threshold, standard deviation above normalized signal
    timestamps
    '''
    vals = np.max(np.abs(acc), axis=1)
    vals /= np.std(vals)
    vals -= np.median(vals)
    events = np.hstack((0, (np.diff((vals > thresh).astype(int)) > 0).astype(int)))
    timestamps = t[events==1]
    return timestamps

def lap_tracking(PATH):
    print('---------')
    print('.. loading ' + PATH + 'drone.csv')
    D = pd.read_csv(PATH + 'drone.csv')
    print('.. loading ' + PATH + 'track.csv')
    T = pd.read_csv(PATH + 'track.csv')
    start_finish_gate_id = T.gate_id.iloc[-1]  # ID of the start/finish gate, i.e. last gate in the track list
    start_gate_label = 0  # ID of lap start events "0"
    # determine required gates order based on the number of gates on the track
    # if 10 gates (flat or wave track)
    if T.shape[0] == 10:
        required_gates_order = [0, 1, 3, 4, 5, 6, 8, 9, 10]
    # else (splits)
    else:
        required_gates_order = [val for val in range(T.shape[0])]
    t = D['ts'].values  # time
    p = D[['PositionX', 'PositionY', 'PositionZ']].values  # quad positions
    acc = D[['AccX', 'AccY', 'AccZ']].values # accelerometer data
    ## Add gate passing events
    # liftoff as start event
    t_ = detect_liftoff(t, p, distance_threshold=0.1, prepend_time=0.)
    e_ = start_gate_label
    E = pd.DataFrame({'ts': t_, 'gate': e_}, index=[0])
    # add gate passing events
    for i in range(T.shape[0]):
        t_ = detect_gate_passing(t, p, T.iloc[i])
        e_ = (np.ones((len(t_),)) * T.gate_id.iloc[i]).astype(int)
        df = pd.DataFrame({'ts': t_, 'gate': e_})
        E = E.append(df)
        # if current gate is start_finish gate, add events a second time with start gate label
        if T.gate_id.iloc[i] == start_finish_gate_id:
            df.gate = start_gate_label
            E = E.append(df)
    # sort events
    E = E.sort_values(by=['ts', 'gate'], ascending=[True, False]).reset_index().drop(columns=['index'])
    ## Add lap number
    lap = []
    lap_number = -1
    for i in range(E.shape[0]):
        #previous index
        i0 = i-1
        if i0 < 0:
            i0 = 0
        #raise lap number if current gate has gate start label or if current gate is 1 and previous had a higher number
        if (E.gate.iloc[i] == start_gate_label) or \
           ((E.gate.iloc[i]==1) & (E.gate.iloc[i0] > E.gate.iloc[i])):
            lap_number += 1
        lap.append(lap_number)
    E['lap'] = lap
    ## Check if lap is valid
    E['is_valid'] = 0
    for lap in E.lap.unique():
        df = E.loc[E.lap == lap]
        t_ = []
        for g_ in required_gates_order:
            if df.loc[df.gate == g_].size:
                t_.append(df.loc[df.gate == g_].ts.values[0])
            else:
                t_.append(np.nan)
        t_ = np.array(t_)
        # determine if lap is valid using two criteria
        # 1.all required gates were passed
        # 2.gate passing order wrt required order is correct (non-negative timestamp differences)
        is_valid = True
        for val in np.diff(t_):
            if (np.isnan(val)) or (val < 0.):
                is_valid = False
        if is_valid:
            E.is_valid[E.lap == lap] = 1
    ## Detect collisions
    collisions = detect_collision(t, acc, thresh=3)
    ## Laptimes
    L = pd.DataFrame()
    for lap in E.lap.unique():
        df = E.copy().loc[E.lap == lap]
        collision_ts = collisions[(collisions > df.ts.min()) & (collisions <= df.ts.max())]
        if collision_ts.size:
            num_collisions = collision_ts.shape[0]
        else:
            num_collisions = 0
        df2 = pd.DataFrame({'lap': lap,
                            'ts_start': df.ts.iloc[0],
                            'ts_end': df.ts.iloc[-1],
                            'lap_time': df.ts.iloc[-1] - df.ts.iloc[0],
                            'is_valid': df.is_valid.iloc[0],
                            'gate_id': [df.gate.values],
                            'gate_timestamps': [df.ts.values],
                            'num_events': df.gate.values.shape[0],
                            'num_unique': np.unique(df.gate.values).shape[0],
                            'num_collisions':num_collisions,
                            'collision_ts': [collision_ts] }, index=[0])
        if L.size:
            L = L.append(df2)
        else:
            L = df2
    # remove the fake final lap (because gate was counted twice)
    L = L.loc[L.lap_time > 0.]
    ## Remove laps that exceed 2 min flight time
    # ts_start = L.ts_start.iloc[0]
    # dur = 120.
    # ts_end = ts_start + dur
    # L=L.loc[L.ts_end<=ts_end]
    # L
    ## Save events and laptimes
    print('.. saving ' + PATH + 'events.csv')
    E.to_csv(PATH + 'events.csv', index=False)
    print('.. saving ' + PATH + 'laptimes.csv')
    L.to_csv(PATH + 'laptimes.csv', index=False)

def closest_point_on_line_3d(a, b, p, use_endpoints=True):
    ap = p-a
    ab = b-a
    c = a + np.dot(ap,ab)/np.dot(ab,ab) * ab
    #if only points between a and b are considered valid points
    if use_endpoints:
        # check if closest point is on the line of the current segment.
        # first: find the line end point closest to the closest point..
        if np.linalg.norm(c - a) < np.linalg.norm(c - b):
            # then check if it lies between the points, if not set closests point
            # to the closest end point
            if np.linalg.norm(c - b) > np.linalg.norm(a - b):
                c = a
        else:
            if np.linalg.norm(c - a) > np.linalg.norm(b - a):
                c = b
    return c

def closest_point_on_multiline_3d(edges, p, curr_segment):
    cp = np.array([])
    for i in range(1, edges.shape[0]):
        curr_cp = closest_point_on_line_3d(edges[i - 1, :], edges[i, :], p)
        if cp.size:
            cp = np.vstack((cp, curr_cp))
        else:
            cp = curr_cp
    if edges.shape[0] > 2:
        # i = np.argmin(np.linalg.norm(cp - p, axis=1)) #points to the segment
        i = curr_segment
        cp = cp[i, :]
        if i == 0: #if first segment, no past progress
            past_progress = 0.
        else: #if any other segment, compute past progress
            past_progress =  np.sum(np.linalg.norm(np.diff(edges[:i+1, :], axis=0), axis=1))
        #compute the progress in this lap so far
        dist_from_lap_start = past_progress + np.linalg.norm(edges[i, :] - cp)
    else:
        dist_from_lap_start = np.linalg.norm(edges[0, :] - cp)
    return (cp, dist_from_lap_start)

def add_track_progress_to_drone_data(PATH, use_projection_shortest_path=True):
    # load relevant input data
    print('..loading {}'.format(PATH + 'drone.csv'))
    D = pd.read_csv(PATH + 'drone.csv')
    T = pd.read_csv(PATH + 'track.csv')
    E = pd.read_csv(PATH + 'events.csv')
    print('..processing data', end='\r')
    # ------------------------------------------------------------------------------------------------------------------
    # Computing Track Progress in terms of orthogonal projection onto the shortest path between subsequent gates
    # ------------------------------------------------------------------------------------------------------------------
    if use_projection_shortest_path:
        #remove zero events
        E = E.loc[(E.gate > 0), :]
        p = D.loc[:, ('PositionX', 'PositionY', 'PositionZ')].values  # quad position
        t = D.loc[:, ('ts')].values  # time
        next_gate_id = np.empty(t.shape)  # next gate ID
        next_gate_id[:] = np.nan
        past_gate_id = np.empty(t.shape)  # next gate ID
        past_gate_id[:] = np.nan
        lap_number = np.empty(t.shape)  # current lap number
        lap_number[:] = np.nan
        for i in range(E.shape[0]):
            if i == 0:
                ind = (t < E.ts.iloc[i])
            elif i == (E.shape[0] - 1):
                ind = (t >= E.ts.iloc[i - 1])
            else:
                ind = (t >= E.ts.iloc[i - 1]) & (t < E.ts.iloc[i])
            _next_gate_id = E.gate.iloc[i]
            if i == 0:
                _past_gate_id = _next_gate_id - 1
            else:
                _past_gate_id = E.gate.iloc[i - 1]
                if _past_gate_id == 10:
                    _past_gate_id = 0
            # _past_gate_id = _next_gate_id - 1
            next_gate_id[ind] = _next_gate_id
            past_gate_id[ind] = _past_gate_id
            lap_number[ind] = E.lap.iloc[i]
            # if i == 0:
            #     lap_number[ind] = 0
            # else:
            #     lap_number[ind] = E.lap.iloc[i-1]

        p_podium = np.array([13.049704, -12.342312, 0.940298])  # podium position
        p_gates = T.loc[:, ('pos_x', 'pos_y', 'pos_z')].values  # gate positions
        p_first_lap = np.vstack((p_podium, p_gates))  # checkpoints first lap (start from podium)
        p_other_laps = np.vstack((p_gates[-1], p_gates))  # checkpoints other laps (start from gate)
        len_first_lap = np.linalg.norm(np.diff(p_first_lap, axis=0), axis=1)  # length of segments in the first lap
        len_other_laps = np.linalg.norm(np.diff(p_other_laps, axis=0), axis=1)  # length of segments in the other laps
        past_gate_id = past_gate_id.astype(int)
        next_gate_id = next_gate_id.astype(int)
        # compute track progress
        total_progress = np.empty((D.shape[0], 1)) # progress variable
        total_progress[:] = np.nan
        p_prime = np.empty(p.shape) #projection points on the shortest path between subsequent gates
        p_prime[:] = np.nan
        #loop over data samples
        for i in range(t.shape[0]):
            _t = t[i]  # current time
            _p = p[i, :]  # current drone position
            _l = lap_number[i]  # current lap number
            _id0 = past_gate_id[i]  # id of the past gate (first gate == 1)
            _id1 = next_gate_id[i]  # id of the next gate (first gate == 1)
            _curr_segment = _id0 #number of the current segment (0-9)
            if _l == 0:  # for the first lap
                # _checkpoints = p_first_lap[_id0 : _id1+1, :]
                _checkpoints = p_first_lap
                _cp0 = p_first_lap[_id0, :]
                _cp1 = p_first_lap[_id1, :]
                _len_current_lap = np.sum(len_first_lap)
                _len_start_to_cp1 = np.sum(len_first_lap[:_id1])
            else:  # for other laps
                # _checkpoints = p_other_laps[_id0 : _id1+1, :]
                _checkpoints = p_other_laps
                _cp0 = p_other_laps[_id0, :]
                _cp1 = p_other_laps[_id1, :]
                _len_current_lap = np.sum(len_other_laps)
                _len_start_to_cp1 = np.sum(len_other_laps[:_id1])
            # _p_prime = closest_point_on_line_3d(_cp0, _cp1, _p)
            _p_prime, _len_start_to_p_prime = closest_point_on_multiline_3d(_checkpoints, _p, _curr_segment)
            p_prime[i, :] = _p_prime
            _len_p_prime_to_cp1 = np.linalg.norm(_cp1 - _p_prime)
            if _l == 0:
                _past_progress = 0.
            if _l == 1:
                _past_progress = np.sum(len_first_lap)
            else:
                _past_progress = np.sum(len_first_lap) + (_l - 1) * np.sum(len_other_laps)
            # _total_progress = _past_progress + _len_start_to_cp1 - _len_p_prime_to_cp1
            _total_progress = _past_progress + _len_start_to_p_prime
            total_progress[i] = _total_progress
        #add output data to drone data
        D['TrackProgress'] = total_progress
        D['ProjectionShortestPathX'] = p_prime[:, 0]
        D['ProjectionShortestPathY'] = p_prime[:, 1]
        D['ProjectionShortestPathZ'] = p_prime[:, 2]
    #-------------------------------------------------------------------------------------------------------------------
    # Computing Track Progress in terms of shortest distance to the next gate
    #-------------------------------------------------------------------------------------------------------------------
    else:
        #compute track segment information
        gate_id = T.loc[:, ('gate_id')].values
        w = T.loc[:, ('pos_x', 'pos_y', 'pos_z')].values
        segment_distance = []
        past_waypoint = []
        next_waypoint = []
        past_gate_id = []
        next_gate_id = []
        for i in range(w.shape[0]):
            if i == 0:
                w0 = w[-1, :]
                id0 = gate_id[-1]
            else:
                w0 = w[i - 1, :]
                id0 = gate_id[i - 1]
            w1 = w[i, :]
            id1 = gate_id[i]
            past_waypoint.append(w0)
            next_waypoint.append(w1)
            past_gate_id.append(id0)
            next_gate_id.append(id1)
            segment_distance.append(np.linalg.norm(w1 - w0))
        segment = np.arange(0, w.shape[0], 1)
        lap_distance = []
        dist = 0.
        for val in segment_distance:
            dist += val
            lap_distance.append(dist)
        W = pd.DataFrame({'segment': segment,
                          'segment_distance': segment_distance,
                          'lap_distance': lap_distance,
                          'GateID_past': past_gate_id,
                          'GateID_next': next_gate_id,
                          'WP_past_pos_x': [val[0] for val in past_waypoint],
                          'WP_past_pos_y': [val[1] for val in past_waypoint],
                          'WP_past_pos_z': [val[2] for val in past_waypoint],
                          'WP_next_pos_x': [val[0] for val in next_waypoint],
                          'WP_next_pos_y': [val[1] for val in next_waypoint],
                          'WP_next_pos_z': [val[2] for val in next_waypoint]
                          })
        # W.to_csv(PATH + 'track_segments.csv', index=False)
        #compute track progress
        p = D.loc[:, ('PositionX', 'PositionY', 'PositionZ')].values
        t = D.loc[:, ('ts')].values
        track_progress = np.empty((D.shape[0], 1))
        track_progress[:] = np.nan
        for i in range(1, E.shape[0]):
            if i == 1:
                past_ts = D.ts.iloc[0]
            else:
                past_ts = E.ts.iloc[i - 1]
            next_lap = E.lap.iloc[i]
            if ((i == (E.shape[0]-1)) and (E.gate.iloc[i] == 0)):
                next_gate = 1
                next_ts = D.ts.iloc[-1]
            else:
                next_gate = E.gate.iloc[i]
                next_ts = E.ts.iloc[i]
            pnt = (t >= past_ts) & (t < next_ts) #pointer
            curr_pos = p[pnt, :]
            next_waypoint_pos = W.loc[
                (W.GateID_next == next_gate), ('WP_next_pos_x', 'WP_next_pos_y', 'WP_next_pos_z')].values
            next_lap_distance = W.loc[(W.GateID_next == next_gate), ('lap_distance')].values
            lap_total_distance = np.sum(W.segment_distance.values)
            dist = np.linalg.norm(curr_pos.reshape(-1, 3) - next_waypoint_pos.reshape(-1, 3), axis=1)
            track_progress[pnt, 0] = next_lap * lap_total_distance + next_lap_distance - dist
        D['TrackProgress'] = track_progress
    # save dronedata ouput
    print('..saving {}'.format(PATH + 'drone.csv'))
    D.to_csv(PATH + 'drone.csv', index=False)

