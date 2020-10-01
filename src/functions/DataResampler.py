import os
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation


class DataResampler(object):

  def __init__(self, path, sr=500, overwrite=False):
    self.overwrite = overwrite
    self.set_valid_paths(path)
    self.sr = sr #sampling rate

  def run(self, indices=None):
    self.set_indices(indices)
    for i in self.indices:
      self.process(i)

  def process(self,i):
    inpath = self.valid_paths[i]['in']
    gatepath = '/'.join(inpath.split('/')[:-2])+'/track.json'
    outpath_data = self.valid_paths[i]['out']
    self.load_gates(gatepath)
    self.data = pd.read_csv(inpath)
    self.resample()
    self.add_gate_distances()
    self.add_gate_passing_events()
    print('..saving {}'.format(outpath_data))
    self.data.to_csv(outpath_data, index=False)
    outpath_gates = '/'.join(outpath_data.split('/')[:-1])+'/track.csv'
    print('..saving {}'.format(outpath_gates))
    self.gates.to_csv(outpath_gates, index=False)

  def add_gate_passing_events(self):
    for name in [n for n in self.data.columns if n[:5]=='dist_']:
      (v, _) = self.find_distance_minima(x=self.data['ts_sync'].values,
                                 y=self.data[name].values,
                                 dist=1.,
                                 dur=2.)
      self.data['event_'+name.split('_')[-1]]=v

  def find_distance_minima(self, x, y, dist=1., dur=2.):
      t=x[y<dist]
      tss = []
      if len(t)>0:
        ind = np.hstack((np.array([False]), np.diff(t)>2.))
        ts0 = np.hstack((t[0],t[ind]))
        ind = np.hstack((np.diff(t)>dur, np.array([False])))
        ts1 = np.hstack((t[ind],t[-1]))
        ts = np.vstack((ts0, ts1))
        for i in range(ts.shape[1]):
            ind = (x>=ts[0,i]) & (x<=ts[1,i])
            _x = x[ind]
            _y = y[ind]
            x_val = _x[np.argmin(_y)]
            tss.append(x_val)
      tss = np.array(tss)
      v = np.zeros(x.shape)
      for ts in tss:
          v[x==ts]=1.
      return (v, tss)

  def add_gate_distances(self):
    for i in range(self.gates.shape[0]):
      p0 = self.gates.iloc[i][['pos_x','pos_y','pos_z']].values.reshape((3,1))
      P = self.data[['PositionX', 'PositionY', 'PositionZ']].values.T
      d = np.empty((P.shape[1],1))
      for j in range(P.shape[1]):
          p = P[:, j:j+1]
          d[j,0]=np.linalg.norm(p-p0)
      self.data['dist_gate'+str(i)]=d

  def load_gates(self, path):
    gates = self.load_track_from_json(path)
    gates['pos_z'] = -(gates['pos_z'] + 1.75)  #reverse z-axis so it points down and add offset
    # gates = gates.drop(index=[0,2,7,9]) #drop bottom center gates
    # print(gates.head(60))
    # gates.index = [5,0,7,8,9,6,1,4,3,2] #reorder gate numbers
    # gates = gates.sort_values(by=['index']).reset_index().drop(columns='index')
    self.gates = gates

  def resample(self):
    sr = self.sr
    t0 = np.floor(self.data['ts_sync'].min())
    t1 = np.ceil(self.data['ts_sync'].max())
    x = np.arange(t0, t1+1/sr, 1/sr)
    out = pd.DataFrame()
    for name in self.data.columns:
        if ((name!='base_data') and
           (name.find('gaze_point_3d')==-1) and
           (name.find('eye_center')==-1) and
           (name.find('gaze_normal')==-1)):
            df = self.data.copy()[['ts_sync', name]].dropna()
            xp = df.values[:,0]
            yp = df.values[:,1]
            y = np.interp(x, xp, yp)
            out[name] = y
    for name in ['world_index',
                 'frame_screen',
                 'frame_surface',
                 'frame_sync_screen',
                 'frame_sync_surface']:
        out[name]=np.floor(out[name]).astype(int)
    self.data = out

  def set_valid_paths(self, path):
    paths = []
    for maindir in os.walk(path):
      if maindir[0]==path:
        for subdir in sorted(maindir[1]):
          inpath = ''.join([maindir[0],subdir,'/sync/merged.csv'])
          outpath = ''.join([maindir[0],subdir,'/sync/resampled.csv'])
          if ( os.path.isfile(inpath) and
              (self.overwrite or not os.path.isfile(outpath)) ):
            paths.append({'in': inpath, 'out': outpath})
    count=0
    if len(paths)>0:
      print('Processing options:')
      for p in paths:
        print('key [{}]: {}'.format(count, '/'.join(p['out'].split('/')[:-2])))
        count+=1
      print()
    else:
      print('Processing done.')
    self.valid_paths = paths

  def set_indices(self, indices):
    if len(self.valid_paths)==0:
      indices = None
    if indices is None:
      indices = [i for i in range(len(self.valid_paths))]
    if isinstance(indices,int):
        indices = [indices]
    self.indices = indices

  def load_track_from_json(self, fpn):
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
                        'pos_z': [pos[2]],
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

