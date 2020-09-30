import os
from shutil import copyfile, copytree
import pandas as pd


class DataLoader(object):

  def __init__(self, path):
    self.root = path
    self.df = pd.read_csv(path+'config.csv')
    self.set_paths(self.df['session'].iloc[0])

  def set_paths(self, i):
    df = self.df[self.df['session']==i]
    self.session = df['session'].values[0]
    self.world = df['world'].values[0]
    self.track = df['track'].values[0].replace('_', '-')
    self.gaze = df['gaze'].values[0]
    self.pupilpath = self.root+'init/'+df['pupildatapath'].values[0]
    self.dronedatafile = self.root+'init/'+df['dronedatafile'].values[0]
    self.screencapfile = self.root+'init/'+df['screencapfile'].values[0]
    self.jsonfile = (self.root+'init/'+df['airrsimpath'].values[0]+
                    'airr_Data/StreamingAssets/Maps/linux/'+
                    df['world'].values[0]+'/'+df['track'].values[0]+'.json')
    self.outpath = self.root+'_'.join(['%02d'%self.session, self.world, self.track, self.gaze])+'/'

  def make_outpath(self, path):
    outpath = '/'
    folders = path.split('/')
    for fold in folders:
        if len(fold)>0:
            outpath += fold +'/'
            if os.path.isdir(outpath) == False:
                os.mkdir(outpath)

  def run(self, sessions=None):
    if sessions == None:
      sessions = self.df['session'].values
    if isinstance(sessions, int):
      sessions = [sessions]
    for s in sessions:
      self.set_paths(s)
      self.make_outpath(self.outpath)
      print('..importing data to: {}'.format(self.outpath))
      inpath = self.dronedatafile
      outpath = self.outpath+'drone.csv'
      if os.path.isfile(inpath) and not os.path.isfile(outpath):
        copyfile(inpath, outpath)
      inpath = self.screencapfile
      outpath = self.outpath+'screen.mp4'
      if os.path.isfile(inpath) and not os.path.isfile(outpath):
        copyfile(inpath, outpath)
      inpath = self.jsonfile
      outpath = self.outpath+'track.json'
      if os.path.isfile(inpath) and not os.path.isfile(outpath):
        copyfile(inpath, outpath)
      inpath = self.pupilpath
      outpath = self.outpath+'gaze/'
      if os.path.exists(inpath) and not os.path.exists(outpath):
        copytree(inpath, outpath)
        inpath = self.pupilpath.split('exports/')[0]+'world.intrinsics'
        outpath = outpath+'world.intrinsics'
        copyfile(inpath, outpath)
