#https://pythonmatplotlibtips.blogspot.com/2018/01/combine-3d-two-2d-animations-in-one-figure-timedanimation.html
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3D
from scipy.spatial.transform import Rotation

class Animation3D(animation.TimedAnimation):
    '''
    D = drone data sampled at 500 Hz
    gates = list of gate objects(shapely surfaces)
    camera = camera object with the frame coordinates
    C = camera position and rotation data sampled at 500 Hz, same timestamps as in D
    G = gaze 2D points and 3D rays, sampled at 200 Hz
    '''
    def __init__(self, D, C=None, G=None, Gate_objects=None, Camera_object=None, vector_names=None, Floor_object=None, fig=None, ax=0,
                 speed=1, fps=25, trail_time=1., axis_length=2., view=(90, -90), equal_lims=25, xlims=None, ylims=None, zlims=None,
                 config='gazesim'):
        #configuration: set names of inputdata according to configuration tag
        if config=='gazesim':
            self.config = {'time': 'ts',
                           'drone_position' : ['PositionX', 'PositionY', 'PositionZ'],
                           'drone_rotation' : ['rot_x_quat', 'rot_y_quat', 'rot_z_quat', 'rot_w_quat'],
                           'camera_position' : ['cam_pos_x', 'cam_pos_y', 'cam_pos_z'],
                           'camera_rotation' : ['cam_rot_x_quat', 'cam_rot_y_quat', 'cam_rot_z_quat', 'cam_rot_w_quat'],
                           'norm_ray' : ['norm_ray_x', 'norm_ray_y', 'norm_ray_z'],
                           'close_intersect_pos' : ['close_intersect_pos_x', 'close_intersect_pos_y', 'close_intersect_pos_z'],
                           'close_intersect_distance' : 'close_intersect_distance'}
        elif config=='liftoff':
            self.config = {'time': 'time-since-start [s]',
                           'drone_position' : ['position_x [m]', 'position_y [m]', 'position_z [m]'],
                           'drone_rotation' : ['rotation_x [quaternion]', 'rotation_y [quaternion]', 'rotation_z [quaternion]', 'rotation_w [quaternion]'],
                           'camera_position' : ['cam_pos_x', 'cam_pos_y', 'cam_pos_z'],
                           'camera_rotation' : ['cam_rot_x_quat', 'cam_rot_y_quat', 'cam_rot_z_quat', 'cam_rot_w_quat'],
                           'norm_ray' : ['norm_ray_x', 'norm_ray_y', 'norm_ray_z'],
                           'close_intersect_pos' : ['close_intersect_pos_x', 'close_intersect_pos_y', 'close_intersect_pos_z'],
                           'close_intersect_distance' : 'close_intersect_distance'}
        else:
            self.config = None
        #past time [sec] highlighting position data with a different color
        self.trail_time = trail_time
        #length [m] of quadrotor body axes
        self.axis_length = axis_length
        #speed of animation
        self.speed = speed
        #frames per second for the visualization
        self.fps = fps
        # interval/period in which the data is visualized
        interval = int(1000 / fps)  #in ms
        #downsample drone data to the visualization sampling rate
        D = self.downsample(D, fps / speed)
        #downsample camera data to the visualization sampling rate
        if C is not None:
            C = self.downsample(C, fps / speed)
        print('visualization sampling rate is {:.2f} Hz'.format(fps))
        #get the objects to visualize from hierarchical header
        Pose_objects = self.get_pose_objects(D, Gate_objects) #here add the drone(s) moving objects
        #add drawable lines for position, trail, body xyz to each object
        Pose_objects = self.add_lines_to_objects(Pose_objects)
        #make a figure
        if fig is None:
            fig = plt.figure(figsize=(10, 10))
            self.ax = fig.add_subplot(1, 1, 1,projection="3d")
        else:
            self.ax = fig.axes[ax]
        #add lines to current axis
        self.ax = self.add_lines_to_axis(self.ax, Pose_objects)
        # add static gates to axis
        if Gate_objects is not None:
            self.add_gates_to_axis(self.ax, Gate_objects)
        #add camera frame to axis
        if (Camera_object is not None) and (C is not None):
            #make a camera object with position information etc
            self.camera_object = {'name': 'camera',
                                  'time': C.loc[:, (self.config['time'])].values,
                                  'position': C.loc[:, (self.config['camera_position'])].values,
                                  'rotation': C.loc[:, (self.config['camera_rotation'])].values,
                                  'object' : Camera_object,
                                  'line': Line3D([], [], [], color='black')}
            #add camera object as line to the axis
            self.ax.add_line(self.camera_object['line'])
        else:
            #if no camera data available set this variable to None (important, will be checked in the draw method)
            self.camera_object = None
        #add gaze projection to axis
        if G is not None:
            ts_trace = G.loc[:, (self.config['time'])].values
            ts_drone = D.loc[:, (self.config['time'])].values
            idx = [np.argmin(np.abs(ts_trace - _t)) for _t in ts_drone]

            ts_trace = G.loc[:, (self.config['time'])].iloc[idx].values
            p_origin = G.loc[:, (self.config['camera_position'])].iloc[idx].values
            norm_ray = G.loc[:, (self.config['norm_ray'])].iloc[idx].values
            p_close_intersect = G.loc[:, (self.config['close_intersect_pos'])].iloc[idx].values
            ind = np.isnan(p_close_intersect[:,0]).flatten()
            p_endpoint = p_close_intersect
            p_endpoint[ind, :] = p_origin[ind, :] + norm_ray[ind, :] * 1000.
            ray_length = G.loc[:, (self.config['close_intersect_distance'])].iloc[idx]
            ray_length[ind] = 1000.
            # make a camera object with position information etc
            self.trace_object = {'name': 'camera',
                                  'time': ts_trace,
                                  'origin': p_origin,
                                  'endpoint': p_endpoint,
                                  'norm_ray': norm_ray,
                                  'length': ray_length,
                                  'line': Line3D([], [], [], color='cyan')}
            # add camera object as line to the axis
            self.ax.add_line(self.trace_object['line'])
        else:
            self.trace_object = None
        #add vector objects to axis
        if vector_names is not None:
            colors = ['c', 'm', 'y', 'k']
            self.vector_objects = []
            i = 0
            for name in vector_names:
                t = D.loc[:, (self.config['time'])].values
                p0 = D.loc[:, (self.config['drone_position'])].values
                v = D.loc[:, (name % 'x', name % 'y', name % 'z')].values
                length = np.linalg.norm(v, axis=1)
                vn = v / length.reshape((-1, 1))
                p1 = p0 + v
                self.vector_objects.append({  'name': name,
                                              'time': t,
                                              'origin': p0,
                                              'endpoint': p1,
                                              'norm_ray': vn,
                                              'length': length,
                                              'line': Line3D([], [], [], color=colors[i]) })
                self.ax.add_line(self.vector_objects[-1]['line'])
                i = i+1
        else:
            self.vector_objects = None
        #save objects as global variable
        self.Pose_objects = Pose_objects
        #save time as global variable
        self.t = self.Pose_objects[0]['time']
        #label the axes
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_zlabel('Z [m]')
        #set the view
        self.ax.view_init(elev=view[0], azim=view[1])
        #set axis limits
        #if equal axis limits requested
        if equal_lims is not None:
            if isinstance(equal_lims, tuple) or isinstance(equal_lims, list):
                lims = tuple(equal_lims)
            else:
                lims = (-equal_lims, equal_lims)
            self.ax.set_xlim(lims)
            self.ax.set_ylim(lims)
            self.ax.set_zlim(lims)
        #if individual axis limits or none are requested
        else:
            min_vals = np.nanmin(self.Pose_objects[0]['position'], axis=0)
            max_vals = np.nanmax(self.Pose_objects[0]['position'], axis=0)
            if xlims is None:
                xlims = (min_vals[0], max_vals[0])
            self.ax.set_xlim(xlims)
            if ylims is None:
                ylims = (min_vals[1], max_vals[1])
            self.ax.set_ylim(ylims)
            if zlims is None:
                zlims = (min_vals[2], max_vals[2])
            self.ax.set_zlim(zlims)
        #make the animation
        animation.TimedAnimation.__init__(self, fig, interval=interval, repeat=False, blit=False)

    def add_lines_to_axis(self, ax, objects):
        for i in range(len(objects)):
            for _, line in objects[i]['lines'].items():
                ax.add_line(line)
        return ax

    def add_gates_to_axis(self, ax, gates):
        for gate in gates:
            ax.add_line(Line3D(gate.corners[0, :], gate.corners[1, :], gate.corners[2, :], color='blue', linewidth=3))
        return ax

    def add_lines_to_objects(self, objects):
        for i in range(len(objects)):
            objects[i]['lines'] = {'position' : Line3D([], [], [], color='black'),
                                   'trail' : Line3D([], [], [], color='m', linewidth=2),
                                   'body_x' : Line3D([], [], [], color='red', linewidth=2),
                                   'body_y' : Line3D([], [], [], color='green', linewidth=2),
                                   'body_z' : Line3D([], [], [], color='blue', linewidth=2)}
        return objects

    def downsample(self, df, fps):
        print('downsampling for visualization..')
        #time in seconds
        t = df.loc[:, (self.config['time'])].values.flatten()
        sr_input = 1 / np.nanmedian(np.diff(t))
        print('input data sampling rate is {:.2f} Hz'.format(sr_input))
        # time steps to consider for visualization
        steps = int(sr_input / fps)
        ind = np.arange(0, t.shape[0], steps)
        #downsample the data
        df = df.iloc[ind]
        sr_output = 1 / np.nanmedian(np.diff(df.loc[:, (self.config['time'])].values.flatten()))
        print('output data sampling rate is {:.2f} Hz'.format(sr_output))

        return df

    def get_pose_objects(self, D, G=None, C=None):
        '''
        Creates pose objects with x y z axis visualized
        D: for drone data
        T: for gate positions
        C: for camera
        '''
        #Time variable is shared between all of these
        t = D.loc[:, (self.config['time'])].values.flatten()
        ###############################
        # Make a list of pose objects #
        ###############################
        #Each object consists of a tuple {name of object, time [sec], position [m], rotation [quaternion or helical]}
        objects = []
        #Drone pose object
        _name = 'drone'
        _P = D.loc[:, (self.config['drone_position'])].values # in meters
        _R = D.loc[:, (self.config['drone_rotation'])].values # rotation as quaternion
        objects.append({'name': _name, 'time': t, 'position': _P, 'rotation': _R})
        #Gate pose objects
        if G is not None:
            samples = D.shape[0]
            i = 0
            for gate in G:
                _name = 'gate%d' % i
                _P = np.tile(gate.center, (samples, 1)).astype(float)
                _R = np.tile(gate.rotation, (samples, 1)).astype(float)
                objects.append({'name': _name, 'time': t, 'position': _P, 'rotation': _R})
                i += 1
        #Camera pose objects
        if C is not None:
            _name = 'camera'
            _P = C.loc[:, (self.config['camera_position'])].values  # in meters
            _R = C.loc[:, (self.config['camera_rotation'])].values  # rotation as quaternion
            objects.append({'name': _name, 'time': t, 'position': _P, 'rotation': _R})
        return objects

    def _draw_frame(self, framedata):
        #index data
        i = framedata
        head = i - 1
        head_slice = (self.t > self.t[i] - self.trail_time) & (self.t < self.t[i])
        #update the lines
        for j in range(len(self.Pose_objects)):
            #past positions
            self.Pose_objects[j]['lines']['position'].set_data(self.Pose_objects[j]['position'][:i, 0], self.Pose_objects[j]['position'][:i, 1])
            self.Pose_objects[j]['lines']['position'].set_3d_properties(self.Pose_objects[j]['position'][:i, 2])
            #trail positions
            self.Pose_objects[j]['lines']['trail'].set_data(self.Pose_objects[j]['position'][head_slice, 0], self.Pose_objects[j]['position'][head_slice, 1])
            self.Pose_objects[j]['lines']['trail'].set_3d_properties(self.Pose_objects[j]['position'][head_slice, 2])
            #body axes
            p0 = self.Pose_objects[j]['position'][head, :]
            r = self.Pose_objects[j]['rotation'][head, :]
            for name, idx in zip(['body_x', 'body_y', 'body_z'], [0, 1, 2]):
                #if rotation contains NaNs, don't draw the axes
                if np.isnan(r).any():
                    p = np.vstack((p0, p0))
                #if rotation contains no NaNs, draw the axes
                else:
                    v = np.array([0., 0., 0.])
                    v[idx] = 1.
                    v = v * self.axis_length
                    if r.shape[0] == 4:
                        v = Rotation.from_quat(r).apply(v)
                    else:
                        v = Rotation.from_euler('xyz', r).apply(v)
                    p1 = p0 + v
                    p = np.vstack((p0, p1))
                self.Pose_objects[j]['lines'][name].set_data(p[:, 0], p[:, 1])
                self.Pose_objects[j]['lines'][name].set_3d_properties(p[:, 2])
        #update the camera frame
        if self.camera_object is not None:
            self.camera_object['object'].update(self.camera_object['position'][head, :], self.camera_object['rotation'][head, :])
            p = self.camera_object['object'].frame.T
            self.camera_object['line'].set_data(p[:, 0], p[:, 1])
            self.camera_object['line'].set_3d_properties(p[:, 2])
        #update the trace object
        if self.trace_object is not None:
            p0 = self.trace_object['origin'][head, :]
            p1 = self.trace_object['endpoint'][head, :]
            p = np.vstack((p0, p1))
            self.trace_object['line'].set_data(p[:, 0], p[:, 1])
            self.trace_object['line'].set_3d_properties(p[:, 2])
        #update the vector objects
        if self.vector_objects is not None:
            for obj in self.vector_objects:
                p0 = obj['origin'][head, :]
                p1 = obj['endpoint'][head, :]
                p = np.vstack((p0, p1))
                obj['line'].set_data(p[:, 0], p[:, 1])
                obj['line'].set_3d_properties(p[:, 2])
        #-----------------------------------
        # get the updated lines for drawing
        #-----------------------------------
        lines = []
        #..lines for the drone objects
        for j in range(len(self.Pose_objects)):
            for name in self.Pose_objects[j]['lines'].keys():
                lines.append(self.Pose_objects[j]['lines'][name])
        #..lines for the camera object
        if self.camera_object is not None:
            lines.append(self.camera_object['line'])
        # ..lines for the trace object
        if self.trace_object is not None:
            lines.append(self.trace_object['line'])
        #update the drawn artists
        self._drawn_artists = lines
        #show time in title
        self.ax.set_title(u'Speed: {}x, Time: {:.1f} sec'.format(self.speed, self.t[head]))

    def new_frame_seq(self):
        return iter(range(self.t.size))

    def _init_draw(self):
        lines = []
        for i in range(len(self.Pose_objects)):
            for _, line in self.Pose_objects[i]['lines'].items():
                lines.append(line)
        for l in lines:
            l.set_data([], [])

    def show(self):
        plt.show()
