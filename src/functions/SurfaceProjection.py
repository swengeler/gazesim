#https://pythonmatplotlibtips.blogspot.com/2018/01/combine-3d-two-2d-animations-in-one-figure-timedanimation.html
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3D
from scipy.spatial.transform import Rotation
import cv2

class SurfaceProjection(animation.TimedAnimation):

    def __init__(self, cap, W, R, gates, fps=25, fig=None, ax=0):

        self.camera_matrix = np.array([[0.41, 0., 0.5], [0., 0.56, 0.5], [0., 0., 1.]])
        self.camera_dist_coefs = np.array([[0., 0., 0., 0.]])

        self.cap = cap
        interval = 1000/fps

        #downsample to visualization framerate
        ts_cap = W.ts.values
        ts_ideal = np.arange(W.ts.min(), W.ts.max(), 1/fps)
        idx = [ np.argmin(np.abs(ts_cap - t)) for t in ts_ideal]
        self.ts = W.loc[:, ('ts')].iloc[idx].values
        self.frames = W.loc[:, ('frame')].iloc[idx].values
        ts_R = R.ts.values
        idx = [ np.argmin(np.abs(ts_R - t)) for t in self.ts]
        self.R = R.iloc[idx]
        self.gates = gates

        if fig is None:
            fig = plt.figure(figsize=(10, 10))
            self.ax = fig.add_subplot(1, 1, 1)
        else:
            self.ax = fig.axes[ax]

        self.ani = animation.FuncAnimation(fig, self.update, interval=interval, blit=False)

    def get_image(self, frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, im = self.cap.read()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    def update(self, i):
        if i<self.frames.shape[0]:
            self.ax.cla()
            im = self.get_image(int(self.frames[i]))
            name = self.R.close_intersect_name.iloc[i]
            if name.find('gate') != -1:
                j = int(name.split('gate')[-1])
                p_cam = self.R.loc[:, ('cam_pos_x', 'cam_pos_y', 'cam_pos_z')].iloc[i].values
                r_cam = self.R.loc[:, ('cam_rot_x_quat', 'cam_rot_y_quat', 'cam_rot_z_quat', 'cam_rot_w_quat')].iloc[i].values
                im = self.superimpose(im, p_cam, r_cam, self.gates[j])
            self.ax.imshow(im)
        # self.ax.plot([400], [400], 'ro')

    def superimpose(self, im, p_cam, r_cam, gate):
        h, w, _ = im.shape
        corners = gate.corners
        pts = np.zeros((2, corners.shape[1]))
        for i in range(corners.shape[1]):
            p = corners[:,i:i+1]
            p_ = self.trans_W_C(p_cam, r_cam, p)
            pts[:, i:i+1] = (p_ * np.array([w, h])).reshape(2, 1)

        for i in range(pts.shape[1]-1):
            im = cv2.line(im, (int(pts[0, i]), int(pts[1, i])),
                              (int(pts[0, i+1]), int(pts[1, i+1])),
                              (255, 0, 0), 2)
        return im

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