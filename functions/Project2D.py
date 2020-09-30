import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D

class Project2D(object):

    def __init__(self, data, gates, k=np.array([[0.41, 0., 0.5], [0., 0.56, 0.5], [0., 0., 1.0]]),
                 d=np.array([[0., 0., 0., 0.]]), t_cam=[0.2, 0., -0.1], uptilt=30., gate_width=2.5,
                 path_screen=None, path_surface=None):
        self.gates_df = gates
        self.gates = None
        self.gate_width = None
        self.set_gate_width(gate_width)
        self.uptilt = None
        self.set_uptilt(uptilt)
        self.t_cam = None
        self.set_t_cam(t_cam)
        self.K = None
        self.set_k(k)
        self.D = None
        self.set_d(d)
        self.data = data
        self.fov = 120.
        self.curr_gates = None
        self.curr_pose_cam = None
        self.curr_pose_quad = None
        self.curr_point_gaze = None
        self.curr_point_cam_center = None
        self.curr_point_gate_center = None
        self.curr_point_heading = None
        self.angles = None
        self.vectors_3d = None
        if path_screen is not None:
            self.cap_screen = cv2.VideoCapture(path_screen)
        else:
            self.cap_screen = None
        if path_surface is not None:
            self.cap_surface = cv2.VideoCapture(path_surface)
        else:
            self.cap_surface = None
        self.im_screen = None
        self.im_surface = None

    def update(self, t):
        # Note: The order matters in which global variables are updated !!
        df = pd.DataFrame(self.data.loc[self.data['ts_sync'] >= t].iloc[0:1])
        p_quad = df[['PositionX', 'PositionY', 'PositionZ']].values.flatten()
        r = df[['RotationX', 'RotationY', 'RotationZ']].values.flatten()
        r_quad = Rotation.from_euler('xyz', r, degrees=False).as_matrix()
        self.curr_pose_quad = (p_quad, r_quad)
        p_cam = (p_quad + r_quad @ self.t_cam).flatten()
        r_cam = r_quad @ Rotation.from_euler('y', [self.uptilt], degrees=True).as_matrix()
        self.curr_pose_cam = (p_cam, r_cam)
        self.get_vectors_3d(df)
        self.curr_gates = self.get_visible_gates(self.curr_pose_cam)
        self.curr_point_gate_center = self.get_curr_point_gate_center()
        self.curr_point_gaze = self.get_curr_point_gaze(t)
        self.curr_point_cam_center = self.get_curr_point_cam_center()
        self.curr_point_heading = self.get_curr_point_heading()
        self.get_curr_angles()
        if self.cap_screen is not None:
            im = self.get_image(t, 'screen')
            im = self.superimpose(im)
            im = self.add_ray_2d(im, self.curr_point_gate_center[0], 'r')
            im = self.add_ray_2d(im, self.curr_point_gaze[0], 'b')
            im = self.add_ray_2d(im, self.curr_point_cam_center[0], 'k')
            im = self.add_ray_2d(im, self.curr_point_heading[0], 'm')
            self.im_screen = im
        if self.cap_surface is not None:
            im = self.get_image(t, 'surface')
            im = self.superimpose(im)
            im = self.add_ray_2d(im, self.curr_point_gate_center[0], 'r')
            im = self.add_ray_2d(im, self.curr_point_gaze[0], 'b')
            im = self.add_ray_2d(im, self.curr_point_cam_center[0], 'k')
            im = self.add_ray_2d(im, self.curr_point_heading[0], 'm')
            self.im_surface = im

    def get_vectors_3d(self, df):
        v_heading = df[['VelocityX', 'VelocityY', 'VelocityZ']].values.flatten()
        v_cam = self.transform_c_w(self.curr_pose_cam, np.array([0.5, 0.5]), d=1.) - self.curr_pose_cam[0]
        v_gaze = self.transform_c_w(self.curr_pose_cam, np.array([df['x_su'].values[0] / df['width'].values[0],
                 df['y_su'].values[0] / df['height'].values[0]]), d=1.) - self.curr_pose_cam[0]
        v_gates = []
        for g in self.gates[1]:
            v_gates.append(g - self.curr_pose_cam[0])
        self.vectors_3d = {'heading': v_heading, 'cam': v_cam, 'gaze': v_gaze, 'gates': v_gates}

    def add_camera_3d(self, ax=None, dist=6.):
        if ax is None:
            fig=plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        pts = None
        for pt in [(0, 0), (0, 1), (1, 1), (1, 0)]:
            p_ = self.transform_c_w(self.curr_pose_cam, pt, d=dist).reshape(3, 1)
            if pts is None:
                pts = p_
            else:
                pts = np.hstack((pts, p_))
        pts = np.hstack((pts, pts[:, 0:1]))

        ax.plot(pts[0, :], pts[1, :], pts[2, :], 'k-')
        p = self.curr_pose_cam[0]
        for i in range(pts.shape[1]-1):
            ax.plot([p[0], pts[0, i]], [p[1], pts[1, i]], [p[2], pts[2, i]], 'k-')

    def add_body_axes_3d(self, ax=None, dist=3.):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        t = self.curr_pose_quad[0].flatten()
        r = self.curr_pose_quad[1]
        pts = None
        for pt, c in zip([(1, 0, 0), (0, 1, 0), (0, 0, 1)], ['r', 'g', 'b']):
            pt = r @ np.array(pt) * dist
            ax.plot([t[0], t[0] + pt[0]], [t[1], t[1] + pt[1]], [t[2], t[2] + pt[2]], c=c, lw=3)

    def add_ray_3d(self, ax=None, p=None, c='k'):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        t = self.curr_pose_quad[0].flatten()
        if p is None:
            p = t
        ax.plot([t[0], p[0]], [t[1], p[1]], [t[2], p[2]], c=c, lw=2)

    def add_ray_2d(self, im, p=None, c='k'):
        h, w, _ = im.shape
        if c == 'y':
            c = (255, 255, 0)
        elif c == 'm':
            c = (255, 0, 255)
        elif c == 'b':
            c = (0, 0, 255)
        elif c == 'r':
            c = (255, 0, 0)
        elif c == 'g':
            c = (0, 255, 0)
        else:
            c = (0, 0, 0)
        if p is not None:
            im = cv2.circle(im, (int(p[0] * w), int(p[1] * h)), 5, c, 6)
        return im

    def set_uptilt(self, uptilt):
        self.uptilt = uptilt

    def set_k(self, k):
        self.K = k

    def set_d(self, d):
        self.D = d

    def set_gate_width(self, gate_width):
        self.gate_width = gate_width
        self.gates = self.set_gates(self.gates_df)

    def set_t_cam(self, t_cam):
        self.t_cam = t_cam

    def set_gates(self, gates):
        corners = []
        centers = []
        for i in range(gates.shape[0]):
            g_pos = gates.iloc[i][['pos_x', 'pos_y', 'pos_z']].values
            g_rot = gates.iloc[i][['rot_x_quat', 'rot_y_quat', 'rot_z_quat', 'rot_w_quat']].values
            ln = self.gate_width / 2.  # todo use actual width of gates from data frame
            g_proto = np.array([[-ln, -ln, ln, ln, -ln],
                                [0., 0., 0., 0., 0.],
                                [-ln, ln, ln, -ln, -ln]])
            g_curr = ((Rotation.from_euler('z', [90], degrees=True).apply(
                Rotation.from_quat(g_rot).apply(g_proto.T)).T +
                       g_pos.reshape(3, 1)).astype(float))
            corners.append(g_curr)
            centers.append(g_pos)
        return corners, centers

    def angle_between(self, u, v):
        u = u.flatten()
        if u.shape[0] == 2:
            u[2] = 1.
        u_norm = u / np.linalg.norm(u)
        v = v.flatten()
        if v.shape[0] == 2:
            v[2] = 1.
        v_norm = v / np.linalg.norm(v)
        return np.arccos(np.clip(np.dot(u_norm, v_norm), -1.0, 1.0))

    def signed_angle(self, u, v, angles=False):
        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)
        signed_angle = np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0])
        if np.abs(signed_angle) > np.pi:
            if signed_angle < 0:
                signed_angle += 2 * np.pi
            else:
                signed_angle -= 2 * np.pi
        if angles:
            signed_angle = np.degrees(signed_angle)
        return signed_angle

    def get_visible_gates(self, pose):
        t, r = pose
        t = t.flatten()
        corners = []
        centers = []
        for i in range(len(self.gates[0])):
            corner = self.gates[0][i]
            center = self.gates[1][i]
            ind = []
            for j in range(corner.shape[1]):
                p = np.linalg.pinv(r) @ (corner[:, j] - t)
                angle = np.degrees(self.angle_between(p, np.array([1, 0, 0])))
                if (p[0, 0] > 0.) and (angle < self.fov): # note: this is actually twice the fov
                    ind.append(True)
                else:
                    ind.append(False)
            corner = corner[:, ind]
            if corner.shape[1] > 0:
                corners.append(corner)
                centers.append(center)
        return corners, centers

    def get_curr_angles(self):
        p_quad = self.curr_pose_cam[0]
        p_cam = self.curr_point_cam_center[1]
        p_gate = self.curr_point_gate_center[1]
        p_gaze = self.curr_point_gaze[1]
        angles = {'gate_cam':  self.angle_between(p_gate - p_quad, p_cam - p_quad),
                  'gate_gaze': self.angle_between(p_gate - p_quad, p_gaze - p_quad)}
        self.angles = angles

    def get_curr_point_gaze(self, t, d=None):
        if d is None:
            d = np.linalg.norm(self.curr_point_gate_center[1] - self.curr_pose_cam[0])
        df = self.data.loc[self.data['ts_sync'] >= t].iloc[0:1]
        gaze_c = np.array([df['x_su'].values / df['width'].values, df['y_su'].values / df['height'].values]).flatten()
        gaze_w = self.transform_c_w(self.curr_pose_cam, gaze_c, d=d)
        return gaze_c, gaze_w

    def get_curr_point_cam_center(self, d=None):
        if d is None:
            d = np.linalg.norm(self.curr_point_gate_center[1] - self.curr_pose_cam[0])
        p_c = np.array([0.5, 0.5])
        p_w = self.transform_c_w(self.curr_pose_cam, p_c, d=d)
        return p_c, p_w

    def get_curr_point_gate_center(self):
        quad = self.curr_pose_cam[0]
        centers = self.curr_gates[1]
        dist = [np.linalg.norm(p - quad) for p in centers]
        i = np.argmin(dist)
        p_w = centers[i]
        p_c = self.transform_w_c(self.curr_pose_cam, p_w)
        return p_c, p_w

    def get_curr_point_heading(self):
        quad = self.curr_pose_cam[0]
        heading = self.vectors_3d['heading']
        p_w = quad + heading
        p_c = self.transform_w_c(self.curr_pose_cam, p_w)
        return p_c, p_w

    def get_frame(self, t, name='screen'):
        if name == 'screen':
            return self.data.loc[self.data['ts_sync'] >= t].iloc[0:1]['frame_sync_screen']
        elif name == 'surface':
            return self.data.loc[self.data['ts_sync'] >= t].iloc[0:1]['frame_sync_surface']
        else:
            return None

    def get_cap_info(self, name='screen'):
        if name == 'surface':
            cap = self.cap_surface
        elif name == 'screen':
            cap = self.cap_screen
        else:
            return None
        return [cap.get(i) for i in range(15)]

    def transform_c_w(self, pose, p, d=1.):
        tf_inv = np.linalg.pinv(Rotation.from_euler('zx', [-90, -90], degrees=True).as_matrix())
        p = np.array([p[0], p[1], 1.]).reshape(3, 1)
        x = np.linalg.pinv(self.K @ np.eye(3)) @ p
        x = (x / np.linalg.norm(x)) * d
        x = tf_inv @ x
        x = pose[1] @ x
        p_ = x.flatten() + pose[0].flatten()
        return p_

    def transform_w_c(self, pose, p):
        t, r = pose
        tf = Rotation.from_euler('zx', [-90, -90], degrees=True).as_matrix()
        x = p.flatten() - t.flatten()
        x = np.linalg.pinv(r[0]) @ x
        x = tf @ x
        rvec, _ = cv2.Rodrigues(np.identity(3))
        tvec = np.zeros((1, 3))
        p_ = cv2.projectPoints(np.float32([x]), rvec, tvec, self.K, self.D)[0]
        p_ = p_.squeeze().flatten()
        return p_

    def get_image(self, t, name='screen'):
        im = None
        cap = None
        if name == 'screen':
            cap = self.cap_screen
        elif name == 'surface':
            cap = self.cap_surface
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.get_frame(t, name))
            _, im = cap.read()
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    def superimpose(self, im):
        h, w, _ = im.shape
        corners = self.curr_gates[0]
        gates = []
        for g in corners:
            pts = np.zeros((2, g.shape[1]))
            for i in range(g.shape[1]):
                p = g[:,i:i+1]
                p_ = self.transform_w_c(self.curr_pose_cam, p)
                pts[:, i:i+1] = (p_ * np.array([w, h])).reshape(2, 1)
            gates.append(pts)
        for g in gates:
            if g.shape[1] > 1:
                for i in range(g.shape[1]-1):
                    im = cv2.line(im, (int(g[0, i]), int(g[1, i])), (int(g[0, i+1]), int(g[1, i+1])), (255, 0, 0), 2)
        return im

    def plot_image(self, ax=None, name='surface'):
        if name == 'surface':
            im = self.im_surface
        elif name == 'screen':
            im = self.im_screen
        else:
            im = None
        if im is not None:
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111)
            ax.imshow(im)

    def plot_2d(self, ax=None, scaling=(1., 1.)):
        corners = self.curr_gates[0]
        gates = []
        for g in corners:
            pts = np.zeros((2, g.shape[1]))
            for i in range(g.shape[1]):
                p = g[:, i:i+1]
                p_ = self.transform_w_c(self.curr_pose_cam, p)
                pts[:, i:i+1] = p_.reshape(2, 1)
            gates.append(pts)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        for g in gates:
            ax.plot(np.array(g[0, :]) * scaling[0], np.array(g[1, :]) * scaling[1], 'r-')

    def plot_3d(self, ax=None):
        # todo: add camera and gaze to plot
        if ax is None:
            fig=plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        p = self.curr_pose_cam[0]
        ax.plot([p[0]], [p[1]], [p[2]], 'bo')
        # for g in self.curr_gates[0]:
        for g in self.gates[0]:
                ax.plot(np.array(g[0, :]), np.array(g[1, :]), np.array(g[2, :]), 'r-')
        self.add_camera_3d(ax)
        self.add_body_axes_3d(ax)
        self.add_ray_3d(ax, self.curr_point_gate_center[1], 'r')
        self.add_ray_3d(ax, self.curr_point_gaze[1], 'b')
        self.add_ray_3d(ax, self.curr_point_cam_center[1], 'k')
        self.add_ray_3d(ax, self.curr_point_heading[1], 'm')
        plt.tight_layout()

    def figure_3d_and_2d(self, t, show=True):
        self.update(t)
        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(121, projection='3d')
        self.plot_3d(ax1)
        plt.gca().set_xlim((-30, 30))
        plt.gca().set_ylim((-30, 30))
        plt.gca().set_zlim((-30, 30))
        # plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.gca().invert_zaxis()
        plt.gca().view_init(90, -90)
        ax2 = fig.add_subplot(122)
        self.plot_image(ax2, 'surface')
        # self.plot_2d(ax2, scaling=(800., 600.)) # obsolete, could be useful for debug only
        plt.gca().set_xlim((0, 800))
        plt.gca().set_ylim((0, 600))
        plt.gca().invert_yaxis()
        if show:
            plt.show()
        return fig

    def figure_2d(self, t, show=True):
        self.update(t)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        self.plot_image(ax, 'surface')
        # self.plot_2d(ax, scaling=(800., 600.)) # obsolete, could be useful for debug only
        plt.gca().set_xlim((0, 800))
        plt.gca().set_ylim((0, 600))
        plt.gca().invert_yaxis()
        if show:
            plt.show()
        return fig
