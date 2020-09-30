import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import ipyvolume as ipv


def ipy_plot(ts, df, gates, resample=20, show_pose=True, show_trajectory=False, show_gates=True, show_floor=True, \
             colnames=['ts_sync', 'PositionX', 'PositionY', 'PositionZ', 'RotationX', 'RotationY', 'RotationZ']):

  if isinstance(ts, float):
    ts = [ts, ts]

  df = df.loc[(df[colnames[0]]>=ts[0]) & (df[colnames[0]]<=ts[1])]
  df = df.iloc[np.arange(0,df.shape[0],int(resample))]

  pos = df[colnames[1:4]].values.T
  rot = df[colnames[4:7]].values.T

  r = np.zeros(rot.shape)
  for i in range(rot.shape[1]):
      u = np.array([1.,0.,0.]).reshape(3,1)
      r[:,i:i+1] = Rotation.from_euler('xyz', rot[:,i]).apply(u.T).T

  d = np.vstack((pos, r))

  fig = ipv.figure(None,800,600)

  if show_floor:
    a = np.arange(-30, 30)
    U, V = np.meshgrid(a, a)
    X = U
    Y = V
    Z = np.ones(X.shape) * (-0.75)
    ipv.plot_surface(X,Y,Z, color="white")


  if show_gates:
    l = 1.
    g = np.array([[-l,-l, l, l,-l],
                  [ 0., 0., 0., 0., 0.],
                  [-l, l, l,-l,-l]])
    for i in range(gates.shape[0]):
        w = gates.iloc[i][['rot_x_quat', 
                           'rot_y_quat', 
                           'rot_z_quat', 
                           'rot_w_quat']].values
        t = gates.iloc[i][['pos_x', 
                           'pos_y', 
                           'pos_z']].values
        g_curr = ((Rotation.from_euler('z',[90],degrees=True).apply(
                  Rotation.from_quat(w).apply(g.T)).T + 
                  t.reshape(3,1)).astype(float))
        ipv.plot(g_curr[0,:],
                 g_curr[1,:],
                 g_curr[2,:], 
                 color="red")

  if show_trajectory:
    ipv.plot(d[0,:],
             d[1,:],
             d[2,:],
             color='black')

  if show_pose:
    for i in range(pos.shape[1]):
      l = 1.
      p = pos[:,i]
      u = np.array([1.,0.,0.]).reshape(3,1)
      r = Rotation.from_euler('xyz', rot[:,i]).apply(u.T)[0]
      ipv.plot([p[0], p[0] + l * r[0]],
               [p[1], p[1] + l * r[1]],
               [p[2], p[2] + l * r[2]],
               color='red', )  
      p = pos[:,i]
      u = np.array([0.,1.,0.]).reshape(3,1)
      r = Rotation.from_euler('xyz', rot[:,i]).apply(u.T)[0]
      ipv.plot([p[0], p[0] + l * r[0]],
               [p[1], p[1] + l * r[1]],
               [p[2], p[2] + l * r[2]],
               color='green')  
      p = pos[:,i]
      u = np.array([0.,0.,1.]).reshape(3,1)
      r = Rotation.from_euler('xyz', rot[:,i]).apply(u.T)[0]
      ipv.plot([p[0], p[0] + l * r[0]],
               [p[1], p[1] + l * r[1]],
               [p[2], p[2] + l * r[2]],
               color='blue')  

  ipv.show()
  w = 60
  h = -0.75
  ipv.xlim(-w/2, w/2)
  ipv.ylim(-w/2, w/2)
  ipv.zlim(-w/2 ,w/2)
  ipv.view(180,-45,1.5)
  ipv.style.axes_off()
  ipv.style.box_off()

#load track/gate information from #/airr_Data/StreamingAssets/Maps/linux/addition-arena/eight_horz_elev.json' #note that z-axis is directed upward and floor is at 0.0
def load_track_from_json(fpn):
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
                        'pos_z': [-(pos[2]+1.75)],
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

#load track/gate information from #/airr-sim/gates_positions.txt' #note that z-axis is directed downward and floor is at -1.75
def load_track_from_txt(fpn):
    df = pd.DataFrame()
    world = None
    track = None
    pos = None
    rot = None
    with open(fpn) as f:
        lis = [line.split() for line in f] 
        for i, x in enumerate(lis):            
            if len(x)>0 and x[0].lower().find('position') ==-1:
                world = x[0].split(':')[0]
                track = x[1]
            if len(x)>0 and x[0].lower().find('position') !=-1:
                pos = np.fromstring(x[1], dtype=float, sep=',')
                rot = np.fromstring(x[3], dtype=float, sep=',')
                df = df.append(pd.DataFrame({
                    'world': world,
                    'track': track,
                    'pos_x': [pos[0]],
                    'pos_y': [pos[1]],
                    'pos_z': [pos[2]],
                    'rot_x_rad': [rot[0]],
                    'rot_y_rad': [rot[1]],
                    'rot_z_rad': [rot[2]] }), ignore_index=True)
    df = df.reset_index().drop(columns=['index'], axis=1)   
    return df 

#returns dataframe info
def get_dronedata_info(df):
    fps = 1./np.nanmedian(df['ExpTime'].diff())
    period = np.nanmedian(df['ExpTime'].diff()) * 1000.
    t_start = df['ExpTime'].iloc[0]
    t_end = df['ExpTime'].iloc[-1]
    samples = df.shape[0]
    duration = df['ExpTime'].iloc[-1] - df['ExpTime'].iloc[0]
    # print('fps = {:.2f} Hz\nperiod = {:.2f} ms\nsamples = {}\nstart = {:.2f} sec\nend = {:.2f} sec\nduration = {:.2f} sec'.format(
    #     fps, period, samples, t_start, t_end, duration))
    info = {'fps':fps, 'period':period, 'samples':samples, 't_start':t_start, 't_end':t_end, 'duration':duration}
    return  info

def get_drawable_objects(ax):
    objs = {
    'uptilt': 30., 
    'fov': (100, 100/1.04),
    'center_view': True,
    'center_view_width': 10.,
    'show': ['traj',
             'body_axes', 
             'quad', 
             'los', 
             'camera'
            ],
    'len_los': 2000.,
    'len_body_axes': 2000.,
    'dist_camera': 2000.,
    'ax': ax,
    'traj': [ 
        ax.plot([0],[0],[0], c='k', lw=2)], #trajectory 
    'body_axes': [
        ax.plot([0],[0],[0], c='r', lw=4),  #x axis
        ax.plot([0],[0],[0], c='g', lw=4),  #y axis
        ax.plot([0],[0],[0], c='b', lw=4)], #z axis
    'quad': [
        ax.plot([0],[0],[0], c='k', lw=2, marker='o', markersize=0.)], #quad
    'los': [
        ax.plot([0],[0],[0], c='k', lw=2)], #line of sight / center ray
    'camera': [
        ax.plot([0],[0],[0], c='k', lw=2), #ray corner1
        ax.plot([0],[0],[0], c='k', lw=2), #ray corner2
        ax.plot([0],[0],[0], c='k', lw=2), #ray corner3
        ax.plot([0],[0],[0], c='k', lw=2), #ray corner4
        ax.plot([0],[0],[0], c='k', lw=2)] #frame
    }
    return objs

## make animation
def update_objects(num, data, objs):
    
    #current position
    p = data[0:3, num:num+1] 

    #rotation transform body to world
    tf_B_W = Rotation.from_euler('xyz',[data[3,num], data[4,num], data[5,num]])
    
    #rotation transform x-axis to los (in body frame)
    tf_x_los = Rotation.from_euler('y',[np.radians(objs['uptilt'])])
    
    #draw trajectory
    if len([x for x in objs['show'] if x=='traj']):
        objs['traj'][0][0].set_data(data[0:2,:num+1])
        objs['traj'][0][0].set_3d_properties(data[2,:num+1])
        
    #draw body axes
    if len([x for x in objs['show'] if x=='body_axes']):

        for i in range(3):
            #x-axis in body frame
            v = np.zeros((1,3))
            v[0, i]= objs['len_body_axes']
            v = tf_B_W.apply(v).T
            vals = np.hstack((p,p+v))
            objs['body_axes'][i][0].set_data(vals[0:2,:])
            objs['body_axes'][i][0].set_3d_properties(vals[2,:])
        
    #draw quad
    if len([x for x in objs['show'] if x=='quad']):
        objs['quad'][0][0].set_data(p[0:2,:])
        objs['quad'][0][0].set_3d_properties(p[2,:])
    
    #draw line of sight
    if len([x for x in objs['show'] if x=='los']):
        v = np.zeros((1,3))
        v[0,0]= objs['len_los']
        v = tf_B_W.apply(tf_x_los.apply(v)).T
        vals = np.hstack((p,p+v))
        objs['los'][0][0].set_data(vals[0:2,:])
        objs['los'][0][0].set_3d_properties(vals[2,:])
    
    #draw camera
    if len([x for x in objs['show'] if x=='camera']):
       
        fov_h = objs['fov'][0]
        fov_v = objs['fov'][1]

        l_ray_corner = (objs['dist_camera'] / np.cos(np.radians(fov_h/2))) / np.cos(np.radians(fov_v/2))

        rays = []
        for tpl in [(-1,-1), 
                    (-1, 1), 
                    ( 1, 1), 
                    ( 1,-1)]:
            u = np.array([1.,0.,0.]) * l_ray_corner
            r = Rotation.from_euler('zy', 
                             [tpl[0]*fov_h/2., tpl[1]*fov_v/2.], 
                             degrees=True)
            u = r.apply(u)
            rays.append(u)

        ray_frame = None
        for i in range(4):
            ray = rays[i]
            v = tf_B_W.apply(tf_x_los.apply(ray)).T
            if ray_frame is None:
                ray_frame = p+v
            else:
                ray_frame = np.hstack((ray_frame,p+v))
            vals = np.hstack((p,p+v))
            objs['camera'][i][0].set_data(vals[0:2,:])
            objs['camera'][i][0].set_3d_properties(vals[2,:])    

        #draw ray frame
        ray_frame = np.hstack((ray_frame,ray_frame[:,:1]))
        objs['camera'][4][0].set_data(ray_frame[0:2,:])
        objs['camera'][4][0].set_3d_properties(ray_frame[2,:])


    #center view on quad
    if objs['center_view']:
        w = objs['center_view_width']
        objs['ax'].set_xlim((p[0]-w/2, p[0]+w/2))
        objs['ax'].set_ylim((p[1]-w/2, p[1]+w/2))
        objs['ax'].set_zlim((p[2]-w/2, p[2]+w/2))
        objs['ax'].invert_zaxis()

        
     
    return (objs)

#project a point from world to image space
def world_to_image(x, R, t, K=None, D=None, tf=None, width=1., height=1.):
  '''
  Transforms a point in world coordinates to camera homogeneous coordinates
  
  Parameters
  ----------
  x : np.array [1,3]
      Point in world frame
  R : np.array [3,3]
      Camera rotation matrix in world frame
  t : np.array [1,3]
      Camera position in world frame
  K : np.array [3,3]
      Camera matrix (intrinsics)
      Defaults to alphapilot airr-sim values
  D : np.array [1,5]
      Camera distortion coefficients [currently not used]
      Defaults to alphapilot airr-sim values
  tf : np.array [3,3]
      Transform between world frame and camera frame axes
      Defaults to two rotation z=90deg followed by x=90deg
      to convert from world (x=forward,y=right,z=down) to
      camera frame (x=right,y=down,z=up)
  width : float
      Width of the image coordinates
      Defaults to 1. [i.e., 0-1 range]
  height : float
      Height of the image coordinates
      Defaults to 1. [i.e., 0-1 range]
  '''
  # default camera matrix and image width and height forAlphapilot Airr-Sim
  if K is None:
    K = np.array([ 5.5224483588700491e+02, 0., 612., 
                   0., 5.5224483588700491e+02, 512., 
                   0., 0., 1. ]).reshape(3,3) 
    width = 1224.
    height = 1024.
  # default distortion coefficients from Alphapilot Airr-Sim [not used at the moment]    
  if D is None: 
    D = np.array([ -3.1171088019991899e-01, 1.3848047745070519e-01, 0., 0., -3.3159518593208655e-02 ])        
  #transform from world (x=?,y=?,z=?) to camera frame (x=right,y=down,z=up)
  #default assumes a world frame orientation (x=forward,y=right,z=down)
  if tf is None:
    tf = Rotation.from_euler('zx',[-90,-90],degrees=True).as_matrix()
  #Processing steps:
  #Step 1. Translate the point to position relative to camera center 
  x = x.flatten() - t.flatten()
  #Step 2. Rotate the point to fit the camera orientation
  x = np.linalg.pinv(R) @ x
  #step 3. Convert world frame orientation to camera orientation
  x = tf @ x
  #Step 4. Convert point to image coordinates
  Rn = np.eye(3)
  p = K @ Rn @ x
  #Step 5. Convert image coordinates to homogeneous coordinates (z distance of 1)
  p = p[:2] / p[2]
  #Step 6. Normalize image coordinates to 0-1 range
  p = np.array(p) / np.array([width, height])

  return p

#project an image point to world coordinate to a chosen distance note there is some ambiguity if the point is behind, this is reflected in x and y values being huge
def image_to_world(p, d, R, t, K=None, D=None, tf=None, width=1., height=1.):
  '''
  Casts a pixel coordinate to world frame at a certain distance

  Parameters
  ----------
  p : np.array [1,2]
      Pixel coordinate (x,y) normalized to 0-1 range
  d : float
      Distance of the point from camera position
  R : np.array [3,3]
      Camera rotation matrix in world frame
  t : np.array [1,3]
      Camera position in world frame
  K : np.array [3,3]
      Camera matrix (intrinsics)
      Defaults to alphapilot airr-sim values
  D : np.array [1,5]
      Camera distortion coefficients [currently not used]
      Defaults to alphapilot airr-sim values
  tf : np.array [3,3]
      Transform between world frame and camera frame axes
      Defaults to two rotation z=90deg followed by x=90deg
      to convert from world (x=forward,y=right,z=down) to
      camera frame (x=right,y=down,z=up)
  width : float
      Width of the image coordinates
      Defaults to 1. [i.e., 0-1 range]
  height : float
      Height of the image coordinates
      Defaults to 1. [i.e., 0-1 range]
  '''
  # default camera matrix and image width and height forAlphapilot Airr-Sim
  if K is None: # default camera matrix from Alphapilot Airr-Sim
    K = np.array([ 5.5224483588700491e+02, 0., 612.,
                   0., 5.5224483588700491e+02, 512.,
                   0., 0., 1. ]).reshape(3,3)
    width = 1224.
    height = 1024.
  # default distortion coefficients from Alphapilot Airr-Sim [not used at the moment]
  if D is None:
    D = np.array([ -3.1171088019991899e-01, 1.3848047745070519e-01, 0., 0., -3.3159518593208655e-02 ])
  #transform from world (x=?,y=?,z=?) to camera frame (x=right,y=down,z=up)
  #default assumes a world frame orientation (x=forward,y=right,z=down)
  if tf is None:
    tf = Rotation.from_euler('zx',[-90,-90],degrees=True).as_matrix()
  #inverse of the transform
  tf_inv = np.linalg.pinv(tf)
  #Processing steps
  #Step 1. Undo normalization of pixel coordinates (to conform to the camera matrix K)
  p = np.array(p) * np.array([width, height])
  #Step 2. Since the pixels ar in homogeneous coordinates, add a 1. to the z axis
  p = np.array([p[0], p[1], 1.])
  #Step 3. Perform reverse projection. i.e. from image to world frame
  Rn = np.eye(3)
  x = np.linalg.pinv(K @ Rn) @ p
  #Step 4. Select the point (along the cated ray) at the desired distance from the camera
  x = (x / np.linalg.norm(x)) * d
  #Step 5. Convert from camera frame axis orientation to world frame orientation
  x = tf_inv @ x
  #Step 6. Rotate the point according to the camera rotation
  x = R @ x
  #Step 7. Translate the point according to camera position
  x = x.flatten() + t.flatten()

  return x
