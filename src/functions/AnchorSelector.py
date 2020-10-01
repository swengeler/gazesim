import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets


class AnchorSelector(object):

  def __init__(self, path, figsize=(20, 20)):
    self.set_valid_paths(path)
    self.figsize = figsize

  def run(self, indices=None):
    self.set_indices(indices)
    if self.indices is not None:
      self.process()

  def process(self):
    if len(self.indices)!=0:
      self.initialize()
      self.set_widgets()
      self.show_widgets()

  def set_indices(self, indices):
    if len(self.valid_paths)==0:
      indices = None
    if indices is None:
      indices = [i for i in range(len(self.valid_paths))]
    if isinstance(indices,int):
        indices = [indices]
    self.indices = indices

  def initialize(self):
    i = self.indices.pop(0)
    inpath1, inpath2, outpath = self.valid_paths[i]
    self.path1 = inpath1+'frame%08d.jpg'
    self.n1 = int(sorted([x for x in os.walk('/'.join(self.path1.split('/')[:-1]))]
                 [0][2])[-1].split('.jpg')[0].split('frame')[-1])
    self.path2 = inpath2
    self.cap2 = cv2.VideoCapture(self.path2)
    self.n2 = int(self.cap2.get(7))
    self.anchors = []
    self.outpath = outpath
    self.lock = True

  def set_valid_paths(self, path):
    paths = []
    for maindir in os.walk(path):
      if maindir[0]==path:
        for subdir in sorted(maindir[1]):
          inpath1 = ''.join([maindir[0],subdir,'/gaze/surface_und_overlay/'])
          inpath2 = ''.join([maindir[0],subdir,'/screen.mp4'])
          outpath = ''.join([maindir[0],subdir,'/anchors.csv'])
          if (os.path.exists(inpath1) and 
             os.path.isfile(inpath2) and not 
             os.path.isfile(outpath)):
            paths.append((inpath1, inpath2, outpath))
    count=0
    if len(paths)>0:
      print('Processing options:')
      for p in paths:
        print('key [{}]: {}'.format(count, '/'.join(p[1].split('/')[:-1])))
        count+=1
      print()
    else:
      print('Processing done.')
    self.valid_paths = paths

  def transition(self):
    self.out.close()
    self.ui.close()
    print('Anchors: {}'.format(self.anchors))
    self.save_anchors()
    if len(self.indices)>0:
      self.process()

  def save_anchors(self):
    if len(self.anchors)>0:
      df = pd.DataFrame(self.anchors, columns=['frame_surface','frame_screen','timestamp_screen'])
      print('..saving '+self.outpath)
      df.to_csv(self.outpath,index=False)

  def set_widgets(self):
    self.w1 = widgets.IntSlider(description='Surface', 
                           min=0, max=self.n1, step=1, 
                           continuous_update=False, 
                           layout={'width': '700px'})
    self.w2 = widgets.IntSlider(description='Screen', 
                           min=0, max=self.n2, step=1, 
                           continuous_update=False, 
                           layout={'width': '700px'})
    self.t1 = widgets.Textarea(description='Timestamp', 
                         continuous_update=False, 
                         layout={'width': '610px','height': '30px'})
    self.t2 = widgets.Textarea(description='Anchors', 
                         continuous_update=False, 
                         layout={'width': '500px','height': '200px'})
    self.b1 = widgets.Button(description='Add/Update')
    self.b2 = widgets.Button(description='Remove Last')
    self.b3 = widgets.Button(description='Done')
    self.b1.on_click(self.on_click_update)
    self.b2.on_click(self.on_click_remove)
    self.b3.on_click(self.on_click_done)

  def show_widgets(self):
    self.out = widgets.interactive_output(self.show_frames, {'i1': self.w1, 'i2': self.w2})
    self.ui = widgets.VBox([self.w1, self.w2, self.t1,
              widgets.HBox([self.t2, 
              widgets.VBox([self.b1, self.b2, self.b3])])])
    widgets.display(self.out, self.ui)

  def on_click_update(self, x):
    if len(self.anchors)>0:
        a = [items for items in self.anchors if (items[0]==self.w1.value) or (items[1]==self.w2.value)]
        if len(a)>0:
            for _a in a:
                self.anchors.remove(_a)
    val = self.t1.value
    if len(val)==0:
      val=np.nan
    else:
      val = float(val)
    self.t1.value = ''
    self.anchors.append((self.w1.value, self.w2.value, val))
    self.anchors.sort()
    s=''
    for item in self.anchors:
        s+='{}\n'.format(item)
    self.t2.value = s

  def on_click_remove(self, x):
    if len(self.anchors)>0:
        self.anchors.pop()
    s=''
    for item in self.anchors:
        s+='{}\n'.format(item)
    self.t2.value = s

  def on_click_done(self, x):
    self.transition()

  def show_frames(self, i1, i2):
    fig = plt.figure(figsize=self.figsize)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    im1 = cv2.imread(self.path1%i1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    ax1.imshow(im1)
    ax1.set_title('SURFACE:\n{}'.format(self.path1%i1))
    ax1.set_xticks([])
    ax1.set_yticks([])
    self.cap2.set(cv2.CAP_PROP_POS_FRAMES, i2) 
    _, im2 = self.cap2.read()
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    ax2.imshow(im2)
    ax2.set_title('SCREEN:\n{}'.format(self.path2))
    ax2.set_xticks([])
    ax2.set_yticks([])
