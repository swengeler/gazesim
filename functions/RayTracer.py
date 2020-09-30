try:
    from src.functions.preprocessing_utils import *
    from src.functions.Animation3D import *
    from src.functions.Gate import *
    from src.functions.Camera import *
    from src.functions.Floor import *
    from src.functions.SurfaceProjection import *
except:
    from functions.preprocessing_utils import *
    from functions.Animation3D import *
    from functions.Gate import *
    from functions.Camera import *
    from functions.Floor import *
    from functions.SurfaceProjection import *

class RayTracer(object):

    def __init__(self, PATH, win=None, show_animation=False, save_animation=False):

        #some fixed settings
        self.camera_fov = 120  # deg
        self.camera_translation = np.array([0.2, 0., 0.1]) # meters
        self.camera_uptilt = 30  # deg
        self.camera_rot_quat = Rotation.from_euler('y', -self.camera_uptilt, degrees=True).as_quat()
        self.camera_matrix = np.array([[0.41, 0., 0.5], [0., 0.56, 0.5], [0., 0., 1.]])
        self.camera_dist_coefs = np.array([[0., 0., 0., 0.]])
        self.gate_dimensions = (3.7, 3.7)#(2.5, 2.5)  # meters

        #load data
        self.T = pd.read_csv(PATH + 'track.csv')
        self.D = pd.read_csv(PATH + 'drone.csv')
        self.G = pd.read_csv(PATH + 'gaze_on_surface.csv')

        #fix y image coordnate direction
        self.G['norm_pos_y'] = 1. - self.G['norm_pos_y']
        self.G['norm_y_su'] = 1. - self.G['norm_y_su']
        self.G['y_su'] = self.G['screen_height'] - self.G['y_su']

        #select data in a certain time window
        if win is not None:
            self.D = self.D.loc[((self.D.ts>=win[0]) & (self.D.ts<win[1])), :]
            self.G = self.G.loc[((self.G.ts>=win[0]) & (self.G.ts<win[1])), :]
            self.W = self.W.loc[((self.W.ts>=win[0]) & (self.W.ts<win[1])), :]

        # camera poses
        self.C = self.get_camera_pose(self.D)

        ######################################
        # Gates and Ground plane as surfaces #
        ######################################
        self.gates = [Gate(self.T.iloc[i], self.gate_dimensions[0], self.gate_dimensions[1]) for i in range(self.T.shape[0])]
        self.camera = Camera(self.camera_fov, self.camera_matrix, self.camera_dist_coefs)
        self.floor = Floor(x=(-30., 30.), y=(-15., 15.), z=0.)

        ###############
        # Ray Tracing #
        ###############
        #merge gaze and camera/drone data, because different sampling rates, mainly done for synchronization
        self.R = self.merge_gaze_cam_data(self.C, self.G)
        #ray tracing
        self.R = self.raytracing(self.R, 1000.)

        ##################################
        # Gaze intersection with objects #
        ##################################
        # Detect gaze intersections with any gates or the ground plane
        # return the first and subsequent intersection points, 3D location, and current distance from camera
        # return the closest object identity and fixation distance
        # todo : later use multiple rays (for a given gaze location confidence) to also detect close by objects of fixation
        self.R = self.intersections(self.R, self.gates, self.floor)

        #########################
        # Visualize the results #
        #########################
        if (save_animation or show_animation):
            #make animation of drone flying through gates
            anim = Animation3D(self.D, self.C, self.R, self.gates, self.camera, self.floor, speed=1.)
            # #make two animations, showing the back projection onto the video image too
            # self.W = pd.read_csv(PATH + 'world_timestamps.csv')
            # self.cap = cv2.VideoCapture(PATH + 'surface.mp4')
            # fig = plt.figure(figsize=(20, 10))
            # fig.add_subplot(2, 1, 1, projection="3d")
            # anim = Animation3D(self.D, self.C, self.R, self.gates, self.camera, self.floor, speed=1., fig=fig, ax=0)
            # fig.add_subplot(2, 1, 2)
            # anim2 = SurfaceProjection(self.cap, self.W, self.R, self.gates, fig=fig, ax=1)
            # plt.show()

        if show_animation:
            plt.show()

        if save_animation:
            # todo : save the animation as video
            None

        ########################
        # Save gaze trace data #
        ########################
        # output data frame with gaze fixations in 2d and 3d, identity of closest fixation object and distance, and all other fixation objects
        # save this output dataframe to csv
        if win is None:
            self.save(PATH)

    def save(self, PATH):
        make_path(PATH)
        self.R.to_csv(PATH + 'gaze_traces.csv', index=False)

    def trans_C_W(self, t_cam, r_cam, p_2d, d=1.):
        #=====
        # Doesnt work, figure out why
        # Important: this transforms the opencv world frame (x=right, y=down, z=forward) to our world frame (x=forward, y=left, z=up)
        # tf = Rotation.from_euler('xz', [np.pi / 2, np.pi / 2])
        # tf = Rotation.from_euler('zx', [-np.pi / 2, -np.pi / 2])
        # =====
        p = np.array([p_2d[0], p_2d[1], 1.]).reshape(3, 1)
        x = np.linalg.pinv(self.camera_matrix @ np.eye(3)) @ p
        x = x.flatten()
        x = np.array([x[2], -x[0], -x[1]]) #convert from Opencv (x=right, y=down, z=forward) to World (x=forward, y=left, z=up) coordinate frame
        x = (x / np.linalg.norm(x)) * d
        # x = tf.apply(x.flatten())
        x = Rotation.from_quat(r_cam).apply(x.flatten())
        p_3d = x.flatten() + t_cam.flatten()
        return p_3d

    def trans_W_C(self, t_cam, r_cam, p_3d):
        # =====
        # Doesnt work, figure out why
        # Important:this transforms the opencv world frame (x=right, y=down, z=forward) to our world frame (x=forward, y=left, z=up)
        # tf = Rotation.from_euler('xz', [np.pi / 2, np.pi / 2])
        # tf = Rotation.from_euler('zx', [-np.pi / 2, -np.pi / 2])
        # =====
        x = p_3d.flatten() - t_cam.flatten()
        x = Rotation.from_quat(r_cam).apply(x, inverse=True)
        # x = tf.apply(x.flatten(), inverse=True)
        x = x.flatten()
        x = np.array([-x[1], -x[2], x[0]])  # convert from World (x=forward, y=left, z=up) to Opencv (x=right, y=down, z=forward) coordinate frame
        rvec, _ = cv2.Rodrigues(np.identity(3))
        tvec = np.zeros((1, 3))
        p_2d = cv2.projectPoints(np.float32([x]), rvec, tvec, self.camera_matrix, self.camera_dist_coefs)[0]
        p_2d = p_2d.squeeze().flatten()
        return p_2d

    def raytracing(self, R, distance=1000.):
        print('..raytracing', end='\r')
        p = R.loc[:, ('cam_pos_x', 'cam_pos_y', 'cam_pos_z')].values
        r = R.loc[:, ('cam_rot_x_quat', 'cam_rot_y_quat', 'cam_rot_z_quat', 'cam_rot_w_quat')].values
        g = R.loc[:, ('norm_x_su', 'norm_y_su')].values
        #new version of the ray_tracing method
        ray_endpoint = np.empty((R.shape[0], 3))
        ray_endpoint[:] = np.nan
        for i in range(R.shape[0]):
            #perform ray tracing only if gaze points are valid, within screen limits
            if ((g[i, 0] >= 0.) & (g[i, 0] <= 1.) & (g[i, 1] >= 0.) & (g[i, 1] <= 1.)):
                ray_endpoint[i, :] = self.trans_C_W(p[i, :], r[i, :], g[i, :], distance)
        R['ray_origin_pos_x'] = p[:, 0]
        R['ray_origin_pos_y'] = p[:, 1]
        R['ray_origin_pos_z'] = p[:, 2]
        R['ray_endpoint_pos_x'] = ray_endpoint[:, 0]
        R['ray_endpoint_pos_y'] = ray_endpoint[:, 1]
        R['ray_endpoint_pos_z'] = ray_endpoint[:, 2]
        norm_ray = ray_endpoint - p
        norm_ray = norm_ray / np.linalg.norm(norm_ray, axis=1).reshape((-1, 1))
        R['norm_ray_x'] = norm_ray[:, 0]
        R['norm_ray_y'] = norm_ray[:, 1]
        R['norm_ray_z'] = norm_ray[:, 2]
        return R

    def intersections(self, R, gates=None, floor=None):
        if gates is not None:
            for j in range(len(gates)):
                print('..checking intersections with gate {}'.format(j), end='\r')
                gate = gates[j]
                M3d = np.empty((R.shape[0], 3))
                M3d[:] = np.nan
                M2d = np.empty((R.shape[0], 2))
                M2d[:] = np.nan
                for i in range(R.shape[0]):
                    p0 = R.loc[:, ('ray_origin_pos_x', 'ray_origin_pos_y', 'ray_origin_pos_z')].iloc[i].values
                    p1 = R.loc[:, ('ray_endpoint_pos_x', 'ray_endpoint_pos_y', 'ray_endpoint_pos_z')].iloc[i].values
                    p2d, p3d = gate.intersect(p0, p1)
                    if p2d is not None:
                        M2d[i, :] = p2d
                        M3d[i, :] = p3d
                R['intersect_gate%d_pos_x' % j] = M3d[:, 0]
                R['intersect_gate%d_pos_y' % j] = M3d[:, 1]
                R['intersect_gate%d_pos_z' % j] = M3d[:, 2]
                R['intersect_gate%d_2d_x' % j] = M2d[:, 0]
                R['intersect_gate%d_2d_y' % j] = M2d[:, 1]
                R['intersect_gate%d_distance' % j] = np.linalg.norm(M3d - R.loc[:, ('ray_origin_pos_x', 'ray_origin_pos_y', 'ray_origin_pos_z')].values, axis=1)
        if floor is not None:
            print('..checking intersections with floor', end='\r')
            M3d = np.empty((R.shape[0], 3))
            M3d[:] = np.nan
            M2d = np.empty((R.shape[0], 2))
            M2d[:] = np.nan
            for i in range(R.shape[0]):
                p0 = R.loc[:, ('ray_origin_pos_x', 'ray_origin_pos_y', 'ray_origin_pos_z')].iloc[i].values
                p1 = R.loc[:, ('ray_endpoint_pos_x', 'ray_endpoint_pos_y', 'ray_endpoint_pos_z')].iloc[i].values
                p2d, p3d = floor.intersect(p0, p1)
                if p2d is not None:
                    M2d[i, :] = p2d
                    M3d[i, :] = p3d
            R['intersect_floor_pos_x'] = M3d[:, 0]
            R['intersect_floor_pos_y'] = M3d[:, 1]
            R['intersect_floor_pos_z'] = M3d[:, 2]
            R['intersect_floor_2d_x'] = M2d[:, 0]
            R['intersect_floor_2d_y'] = M2d[:, 1]
            R['intersect_floor_distance'] = np.linalg.norm(M3d - R.loc[:, ('ray_origin_pos_x', 'ray_origin_pos_y', 'ray_origin_pos_z')].values, axis=1)
        #information about the closest object
        samples = R.shape[0]
        colnames = [name for name in R.columns if ((name.find('intersect')!=-1) & (name.find('distance')!=-1))]
        entitynames = [name.split('_')[1] for name in colnames]
        distances = R.loc[:, colnames].values
        num_intersections = np.sum((np.isnan(distances) == False), axis=1)
        has_intersection = num_intersections>0
        closest_distance = np.empty((samples, 1))
        closest_distance[:] = np.nan
        closest_entity = ['None' for i in range(samples)]
        close_intersect_pos = np.empty((samples, 3))
        close_intersect_pos[:] = np.nan
        for i in range(samples):
            if has_intersection[i]:
                j = np.nanargmin(distances[i, :])
                closest_distance[i] = distances[i, j]
                closest_entity[i] = entitynames[j]
                close_intersect_pos[i, 0] = R['intersect_' + entitynames[j] + '_pos_x'].iloc[i]
                close_intersect_pos[i, 1] = R['intersect_' + entitynames[j] + '_pos_y'].iloc[i]
                close_intersect_pos[i, 2] = R['intersect_' + entitynames[j] + '_pos_z'].iloc[i]
        R['num_intersections'] = num_intersections
        R['close_intersect_name'] = closest_entity
        R['close_intersect_distance'] = closest_distance
        R['close_intersect_pos_x'] = close_intersect_pos[:, 0]
        R['close_intersect_pos_y'] = close_intersect_pos[:, 1]
        R['close_intersect_pos_z'] = close_intersect_pos[:, 2]
        print(R.loc[:, ('close_intersect_name', 'close_intersect_distance')])
        return R

    # def ray_tracing(self, row):
    #     print('..old raytracing REMOVE', end='\r')
    #     #camera pose in World frame
    #     p_cam = np.array([row.cam_pos_x, row.cam_pos_y, row.cam_pos_z])
    #     r_cam = np.array([row.cam_rot_x_quat, row.cam_rot_y_quat, row.cam_rot_z_quat, row.cam_rot_w_quat])
    #     #gaze coordinates in Image frame
    #     gaze_coords = np.array([row.norm_x_su, row.norm_y_su])
    #     #tracing a point at far distance
    #     p0 = p_cam
    #     p1 = self.trans_C_W(p_cam, r_cam, gaze_coords, 1000.)
    #     #check for intersections with gates
    #     intersections = []
    #     for i in range(len(self.gates)):
    #         p_2d, p_3d = self.gates[i].intersect(p0, p1)
    #         if p_2d is not None:
    #             intersections.append({'name' : 'gate%d'%i,
    #                                   'p_2d' : p_2d,
    #                                   'p_3d' : p_3d,
    #                                   'distance' : np.linalg.norm(p_3d - p0)})
    #     #check for intersections with ground floor
    #     p_2d, p_3d = self.floor.intersect(p0, p1)
    #     if p_2d is not None:
    #         intersections.append({'name': 'floor',
    #                               'p_2d': p_2d,
    #                               'p_3d': p_3d,
    #                               'distance': np.linalg.norm(p_3d - p0)})
    #     #sort intersections by distance
    #     if len(intersections) > 0:
    #         distances = np.array([item['distance'] for item in intersections])
    #         sort_index = np.argsort(distances)
    #         closest_distance = intersections[sort_index[0]]['distance']
    #         closest_name = intersections[sort_index[0]]['name']
    #         closest_p_3d = intersections[sort_index[0]]['p_3d']
    #     else:
    #         closest_distance = np.nan
    #         closest_name = ''
    #         closest_p_3d = np.array([np.nan, np.nan, np.nan])
    #     #number of intersections
    #     num_intersections = len(intersections)
    #     #save intersection information
    #     row['num_intersections'] = num_intersections
    #     row['close_intersect_name'] = closest_name
    #     row['close_intersect_distance'] = closest_distance
    #     row['close_intersect_pos_x'] = closest_p_3d[0]
    #     row['close_intersect_pos_y'] = closest_p_3d[1]
    #     row['close_intersect_pos_z'] = closest_p_3d[2]
    #     #tracing a point at unit distance
    #     p_centered = self.trans_C_W(p_cam, r_cam, gaze_coords) - p_cam
    #     row['norm_ray_x'] = p_centered[0]
    #     row['norm_ray_y'] = p_centered[1]
    #     row['norm_ray_z'] = p_centered[2]
    #     return row

    def get_camera_pose(self, D):
        # add camera pose
        p_quad = D.loc[:, ('PositionX', 'PositionY', 'PositionZ')].values
        r_quad = D.loc[:, ('rot_x_quat','rot_y_quat','rot_z_quat','rot_w_quat')].values
        p_cam = Rotation.from_quat(r_quad).apply(np.tile(self.camera_translation, (p_quad.shape[0], 1))) + p_quad
        r_cam = (Rotation.from_quat(r_quad) * Rotation.from_quat(
            np.tile(self.camera_rot_quat, (p_quad.shape[0], 1)))).as_quat()
        pose_cam = np.hstack((p_cam, r_cam))
        names_cam = ['cam_pos_x', 'cam_pos_y', 'cam_pos_z', 'cam_rot_x_quat', 'cam_rot_y_quat', 'cam_rot_z_quat',
                     'cam_rot_w_quat']
        for i in range(len(names_cam)):
            D[names_cam[i]] = pose_cam[:, i]
        D = D.loc[:, ('ts', 'cam_pos_x', 'cam_pos_y', 'cam_pos_z', 'cam_rot_x_quat', 'cam_rot_y_quat', 'cam_rot_z_quat',
                     'cam_rot_w_quat')]
        return D

    def merge_gaze_cam_data(self, C, G):
        #make dataframe with gaze data
        R = G.loc[:, ('ts', 'norm_x_su', 'norm_y_su')].reset_index().drop(columns=['index'])
        #add camera pose
        names_C = ['ts', 'cam_pos_x', 'cam_pos_y', 'cam_pos_z', 'cam_rot_x_quat', 'cam_rot_y_quat', 'cam_rot_z_quat',
                     'cam_rot_w_quat']
        names_G = ['ts_drone', 'cam_pos_x', 'cam_pos_y', 'cam_pos_z', 'cam_rot_x_quat', 'cam_rot_y_quat', 'cam_rot_z_quat',
                     'cam_rot_w_quat']
        ts_C = C.ts.values
        ts_G= G.ts.values
        idx_C = [np.nanargmin(np.abs(ts_C - t)) for t in ts_G]
        C = C.iloc[idx_C].loc[:, names_C].reset_index().drop(columns=['index'])
        C.columns = names_G
        R = R.merge(C, how='outer', left_index=True, right_index=True)
        return R
