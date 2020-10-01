import os
import numpy as np
import pandas as pd


class LapTracker(object):

    def __init__(self, path):
        self.gate_sequence_dict = {'wave': [4, 8, 11, 12, 13, 10, 6, 5, 4],
                                   'flat': [2, 5, 7, 8, 9, 6, 4, 3, 2]}
        self.subjpath = path
        self.subj = int(path.split('/')[-2].split('s')[-1])
        self.folders = self.get_folders(path)
        self.df = self.get_df()

    def save_df(self, foldername='merged'):
        df = self.df
        if not os.path.exists(self.subjpath + foldername):
            os.mkdir(self.subjpath + foldername)
        df.to_csv(self.subjpath + foldername + '/laps_all.csv', index=False)
        df = df.loc[df['valid_sequence']].sort_values(by=['subj', 'cond', 'lap'])
        df.to_csv(self.subjpath + foldername + '/laps_valid.csv', index=False)

    def get_df(self):
        out = pd.DataFrame()
        for track_name in self.folders.keys():
            for folder_name in self.folders[track_name]:

                run = int(folder_name.split('_')[0])
                cond = np.argmax(np.array([key == track_name for key in self.folders.keys()]))

                datapath = self.subjpath + folder_name + '/sync/resampled.csv'
                data = pd.read_csv(datapath, low_memory=False)
                trackpath = self.subjpath + folder_name + '/sync/track.csv'
                gates = pd.read_csv(trackpath, low_memory=False)

                events = self.get_gate_events(data, track_name)
                gates = self.select_gates(events, gates)
                laptimes = self.get_laptimes(events)
                lap_start_end = self.get_lap_timestamp(events)
                gate_event_timestamps = self.get_gate_event_timestamps(events)
                median_vel, max_vel = self.get_lapvelocities(data, events)
                gate_sequence = self.get_gate_sequence(events)
                valid_sequence = self.check_valid_sequence(gate_sequence)

                tracklength = self.get_tracklength(gates)

                curr_df = pd.DataFrame({
                    'subj':  np.ones(laptimes.shape, dtype=int) * self.subj,
                    'run': np.ones(laptimes.shape, dtype=int) * run,
                    'cond': np.ones(laptimes.shape, dtype=int) * cond,
                    'cond_name': [track_name for i in range(laptimes.shape[0])],
                    'gate_sequence': gate_sequence,
                    'valid_sequence': valid_sequence,
                    'lap': np.arange(0, laptimes.shape[0], 1),
                    'lap_ts_start': np.array([val[0] for val in lap_start_end]).reshape(laptimes.shape),
                    'lap_ts_end': np.array([val[1] for val in lap_start_end]).reshape(laptimes.shape),
                    'lap_time': laptimes,
                    'gate_event_ts': gate_event_timestamps,
                    'track_length': np.ones(laptimes.shape) * tracklength,
                    'median_velocity': median_vel,
                    'max_velocity': max_vel,
                    'data_path': datapath,
                    'track_path': trackpath})
                if out.size:
                    out = out.append(curr_df, ignore_index=True)
                else:
                    out = curr_df
        return out

    def get_folders(self, path):
        folders = {'flat': sorted(
            [x[0].split('/')[-1] for x in os.walk(path) if x[0].split('/')[-1].find('_eight-horz_') != -1]),
            'wave': sorted([x[0].split('/')[-1] for x in os.walk(path) if
                            x[0].split('/')[-1].find('_eight-horz-elev_') != -1])}
        # exclude folders that contain no data
        for key in folders.keys():
            folders[key] = [x for x in folders[key] if os.path.isfile(path + x + '/sync/resampled.csv')]
        return folders

    def get_gate_events(self, data, track_name):
        # find all the events by gate number
        ev = pd.DataFrame()
        for n in [n for n in data.columns if n[:5] == 'event']:
            timestamps = data['ts_sync'].loc[data[n] == 1].values
            gate_id = int(n.split('gate')[-1])
            if timestamps.size:
                df = pd.DataFrame({'ts_sync': timestamps, 'gate_old': np.ones(timestamps.shape) * gate_id})
                if ev.size:
                    ev = ev.append(df, ignore_index=True)
                else:
                    ev = df
        # sort by time
        ev = ev.sort_values(by=['ts_sync']).reset_index().drop(columns=['index'])
        # add start as gate event (assuming last gate during this run is first gate)
        t_start = data['ts_sync'].loc[np.abs(data['PositionX'].diff()) > 0.001].iloc[0]
        # intended gate sequence for the current track
        seq = self.gate_sequence_dict[track_name]
        # first gate ID
        id_start = seq[0]
        ev = ev.append(pd.DataFrame({'ts_sync': [t_start], 'gate_old': [id_start]}), ignore_index=True)
        # sort by time
        ev = ev.sort_values(by=['ts_sync']).reset_index().drop(columns=['index'])
        # fill first value (nan) with a large one, then remove second gate (i.e. shortly after first occurring entry)
        ev = ev.loc[(ev['ts_sync'].diff().fillna(999.).values < 1.) == False]
        # sort by time
        ev = ev.sort_values(by=['ts_sync']).reset_index().drop(columns=['index'])
        # relabel gates and add lap number
        new_gates = []
        new_laps = []
        lap_count = -1
        gate_count = -1
        reached_last = True
        for i in range(ev.shape[0]):
            if reached_last or (ev['gate_old'].iloc[i] == seq[0]):
                gate_count = -1
                lap_count += 1
                reached_last = False
            elif ev['gate_old'].iloc[i] == seq[-2]:
                reached_last = True
            gate_count += 1
            new_gates.append(gate_count)
            new_laps.append(lap_count)
        ev['gate'] = new_gates
        ev['lap'] = new_laps
        #add extra event for the last gate of the lap (based on first gate of next lap)
        df0 = ev.loc[ev['gate'] == 0].iloc[1:]
        df0['lap'] -= 1
        val = [ev.loc[ev['lap'] == lap]['gate'].max() + 1 for lap in ev['lap'].unique()[:-1]]
        df0['gate'] = val
        ev = ev.append(df0).sort_values(by=['ts_sync', 'lap', 'gate'])

        return ev

    def select_gates(self, events, gates):
        ind = events['gate_old'].loc[events['lap'] == 2].values  # take the order of the second lap
        # (it didnt work for the first)
        return gates.iloc[ind]

    def get_laptimes(self, df):
        return np.diff(df['ts_sync'].loc[df['gate'] == df['gate'].iloc[0]].values)

    def get_lap_timestamp(self, event):
        ts = []
        for i in np.arange(event['lap'].min(), event['lap'].max(), 1):
            t0 = event['ts_sync'].loc[event['lap'] == i].iloc[0]
            t1 = event['ts_sync'].loc[event['lap'] == i].iloc[-1]
            if t0.size:
                ts.append((t0, t1))
            else:
                ts.append(np.nan)
        return ts

    def get_gate_event_timestamps(self, event):
        ts = []
        for i in np.arange(event['lap'].min(), event['lap'].max(), 1):
            ts_ = event['ts_sync'].loc[event['lap'] == i].values
            if ts_.size:
                ts.append(ts_)
            else:
                ts.append(np.array([]))
        return ts

    def get_gate_sequence(self, event):
        seqs = []
        for i in np.arange(event['lap'].min(), event['lap'].max(), 1):
            vals = event['gate_old'].loc[event['lap'] == i].values
            if vals.size:
                seq = '_'.join([str(int(v)) for v in vals])
            else:
                seq = ''
            seqs.append(seq)
        return seqs

    def check_valid_sequence(self, seq):
        valid = []
        for s in seq:
            if (s == '_'.join([str(x) for x in self.gate_sequence_dict['wave']])) or \
               (s == '_'.join([str(x) for x in self.gate_sequence_dict['flat']])):
                valid.append(True)
            else:
                valid.append(False)
        return valid

    def get_tracklength(self, gates):
        x = gates[['pos_x', 'pos_y', 'pos_z']].values
        x = np.vstack((x, x[0, :]))
        delta = np.diff(x, axis=0)
        l = [np.linalg.norm(delta[i, :]) for i in range(delta.shape[0])]
        return np.sum(l)

    def get_lapvelocities(self, data, events):
        median_vel = np.array([])
        max_vel = np.array([])
        for i in np.arange(0, events['lap'].max(), 1):
            t0 = events['ts_sync'].loc[events['lap'] == i].iloc[0]
            t1 = events['ts_sync'].loc[events['lap'] == i + 1].iloc[0]
            ind = (data['ts_sync'].values >= t0) & (data['ts_sync'].values < t1)
            ts = data['ts_sync'].loc[ind].values
            pos = data[['PositionX', 'PositionY', 'PositionZ']].loc[ind].values
            delta = np.diff(pos, axis=0)
            dist = [np.linalg.norm(delta[i, :]) for i in range(delta.shape[0])]
            dt = np.nanmedian(np.diff(ts))
            vel = dist / dt
            if median_vel.size:
                median_vel = np.vstack((median_vel, np.nanmedian(vel)))
            else:
                median_vel = np.nanmedian(vel)
            if max_vel.size:
                max_vel = np.vstack((max_vel, np.nanmax(vel)))
            else:
                max_vel = np.nanmax(vel)
        return median_vel.flatten(), max_vel.flatten()
