import os
import cv2
import pandas as pd
import numpy as np
from scipy.stats import iqr


class DataSynchronizer(object):

  def __init__(self, path, overwrite=False):
    self.overwrite = overwrite
    self.set_valid_paths(path)

  def run(self, indices=None):
    self.set_indices(indices)
    if self.indices is not None:
      for i in self.indices:
        self.process(i)

  def process(self, i):
    paths = self.valid_paths[i]
    self.gaze = pd.read_csv(paths['gaze'])
    self.drone = pd.read_csv(paths['drone'])
    self.anchors = pd.read_csv(paths['anchors'])
    self.sync()
    self.crop()
    self.save(paths)

  def save(self, paths):
    self.make_outpath(paths['out'])
    outpath = paths['out']+'merged.csv'
    if not os.path.isfile(outpath):
      print('..save {}'.format(outpath))
      self.merged.to_csv(outpath, index=False)
    outpath = paths['out']+'surface.mp4'
    if not os.path.isfile(outpath):
      inpath = paths['surface']
      firstframe = self.merged['frame_surface'].min()
      lastframe = self.merged['frame_surface'].max()
      self.export_video(inpath, outpath, firstframe, lastframe)
    outpath = paths['out']+'world.mp4'
    if not os.path.isfile(outpath):
      inpath = paths['world']
      firstframe = self.merged['frame_surface'].min()
      lastframe = self.merged['frame_surface'].max()
      self.export_video(inpath, outpath, firstframe, lastframe)
    outpath = paths['out']+'screen.mp4'
    if not os.path.isfile(outpath):
      inpath = paths['screen']
      firstframe = self.merged['frame_screen'].min()
      lastframe = self.merged['frame_screen'].max()
      self.export_video(inpath, outpath, firstframe, lastframe)

  def export_video(self, inpath, outpath, firstframe, lastframe):
    cap = cv2.VideoCapture(inpath)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(5)
    fourcc = int(cap.get(6))
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

  def crop(self):
    t0 = self.merged[self.merged['world_index']==self.merged['world_index'].min()]['ts_drone'].values[0]
    t1 = self.merged[self.merged['world_index']==self.merged['world_index'].max()]['ts_drone'].values[0]
    self.merged = self.merged.loc[(self.merged['ts_drone']>=t0) &
                                  (self.merged['ts_drone']<=t1)]
    self.merged['frame_sync_screen'] = self.merged['frame_screen'] - self.merged['frame_screen'].min()
    self.merged['frame_sync_surface'] = self.merged['frame_surface'] - self.merged['frame_surface'].min()
    self.merged['ts_sync'] = self.merged['ts_drone'] - self.merged['ts_drone'].min()

  def add_ts_screen(self):
    '''
    Removes constant offset between ExpTime logs in dronedata.csv and
    displayed times in the screen.mp4, which are Seconds.NanoSeconds
    '''
    ts_screen_imprecise = (self.merged['Seconds'].astype(str)+'.'+self.merged['NanoSeconds'].astype(str)).astype(np.float64).values
    v = self.merged['ExpTime'] - ts_screen_imprecise
    m = np.median(v) # median deviation
    s = iqr(v)       # interquartile range
    f = 10.          # scaling factor
    v = v[(v>=m-f*s) & (v<=m+f*s)] #remove outliers
    constant_offset = np.median(v)
    self.merged['ts_screen'] = self.merged['ExpTime'] - constant_offset

  def sync(self):
    '''
    Synronizes drone and gaze data using anchors.
    Returns merged frame
    '''
    self.merged = self.drone.copy()
    self.anchors = self.anchors.sort_values(by=['frame_screen'])
    #add timestamps corresponding to the timestamps shown in screen video
    self.add_ts_screen()
    #add frame number in screen video
    self.merged['frame_screen'] = self.frame_from_anchor(
      t0 = self.anchors['timestamp_screen'].iloc[0],
      t1 = self.anchors['timestamp_screen'].iloc[-1],
      f0 = self.anchors['frame_screen'].iloc[0],
      f1 = self.anchors['frame_screen'].iloc[-1],
      ts = self.merged['ts_screen'].values)
    #add frame number of the surface video
    self.merged['frame_surface'] = self.frame_from_anchor(
      t0 = self.anchors['timestamp_screen'].iloc[0],
      t1 = self.anchors['timestamp_screen'].iloc[-1],
      f0 = self.anchors['frame_surface'].iloc[0],
      f1 = self.anchors['frame_surface'].iloc[-1],
      ts = self.merged['ts_screen'].values)
    #add drone_timestamps (UTC) to merged
    ddelta = np.median(self.merged['CurrTime'].values - self.merged['ts_screen'].values)
    self.merged['ts_drone'] = self.merged['ts_screen'] + ddelta
    #anchor point drone_timestamps
    dt0 = self.anchors['timestamp_screen'].iloc[0] + ddelta
    dt1 = self.anchors['timestamp_screen'].iloc[-1] + ddelta
    #add gaze timestamps to gaze
    self.gaze['ts_gaze'] = self.gaze['gaze_timestamp']
    #self.gaze['frame_surface'] = self.gaze['world_index']
    #anchor point surface frame
    gf0 = self.anchors['frame_surface'].iloc[0]
    gf1 = self.anchors['frame_surface'].iloc[-1]
    gt0 = self.gaze.loc[self.gaze['world_index']==gf0]['ts_gaze'].iloc[0]
    gt1 = self.gaze.loc[self.gaze['world_index']==gf1]['ts_gaze'].iloc[0]
    #add drone timestamps gaze
    gdelta = np.mean(np.array([dt0-gt0, dt1-gt1]))
    self.gaze['ts_drone'] =  self.gaze['ts_gaze']+gdelta
    #merge the dataframes
    self.merged = self.merged.merge(self.gaze,
                                       how='outer',
                                       left_on='ts_drone',
                                       right_on='ts_drone',
                                       sort=True).sort_values(by=['ts_drone'])

  def frame_from_anchor(self, t0, t1, f0, f1, ts):
    fps = (f1-f0) / (t1-t0)
    fn = []
    for t in ts:
      f = np.floor(f0 + ((t - t0) * fps))
      fn.append(f)
    return fn

  def make_outpath(self, path):
    outpath = '/'
    folders = path.split('/')
    for fold in folders:
        if len(fold)>0:
            outpath += fold +'/'
            if os.path.isdir(outpath) == False:
                os.mkdir(outpath)

  def set_valid_paths(self, path):
    paths = []
    for maindir in os.walk(path):
      if maindir[0]==path:
        for subdir in sorted(maindir[1]):
          path_surface = ''.join([maindir[0],subdir,'/gaze/surface_und_overlay.mp4'])
          path_world = ''.join([maindir[0],subdir,'/gaze/world.mp4'])
          path_screen = ''.join([maindir[0],subdir,'/screen.mp4'])
          path_anchors = ''.join([maindir[0],subdir,'/anchors.csv'])
          path_drone = ''.join([maindir[0],subdir,'/drone.csv'])
          path_gaze = ''.join([maindir[0],subdir,'/gaze.csv'])
          outpath = ''.join([maindir[0],subdir,'/sync/'])
          if (os.path.isfile(path_surface) and
              os.path.isfile(path_world) and
              os.path.isfile(path_screen) and
              os.path.isfile(path_anchors) and
              os.path.isfile(path_drone) and
              os.path.isfile(path_gaze) and
              (self.overwrite or not os.path.exists(outpath)) ):
            paths.append({
              'surface' : path_surface,
              'world' : path_world,
              'screen' : path_screen,
              'anchors' : path_anchors,
              'drone' : path_drone,
              'gaze' : path_gaze,
              'out' : outpath})
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
    else:
      if indices is None:
        indices = [i for i in range(len(self.valid_paths))]
      if isinstance(indices,int):
          indices = [indices]
    self.indices = indices

