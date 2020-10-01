import numpy as np
import cv2
from scipy.spatial.transform import Rotation


class Camera(object):

    def __init__(self, fov=120., camera_matrix=None, dist_coefs=None, length=3.):
        self._corners = None
        self._origin = None
        self._frame = None
        self.set_defaults(fov, camera_matrix, dist_coefs, length)

    @property
    def position(self):
        return self._position

    @property
    def rotation(self):
        return self._rotation

    @property
    def corners(self):
        return self._corners

    @property
    def frame(self):
        return self._frame

    def trans_C_W(self, t_cam, r_cam, p_2d):
        # =====
        # Doesnt work, figure out why
        # Important: this transforms the opencv world frame (x=right, y=down, z=forward) to our world frame (x=forward, y=left, z=up)
        # tf = Rotation.from_euler('xz', [np.pi / 2, np.pi / 2])
        # tf = Rotation.from_euler('zx', [-np.pi / 2, -np.pi / 2])
        # =====
        p = np.array([p_2d[0], p_2d[1], 1.]).reshape(3, 1)
        x = np.linalg.pinv(self.camera_matrix @ np.eye(3)) @ p
        x = x.flatten()
        x = np.array([x[2], -x[0], -x[
            1]])  # convert from Opencv (x=right, y=down, z=forward) to World (x=forward, y=left, z=up) coordinate frame
        x = (x / np.linalg.norm(x)) * self.length
        # x = tf.apply(x.flatten())
        x = Rotation.from_quat(r_cam).apply(x.flatten())
        p_3d = x.flatten() + t_cam.flatten()
        return p_3d

    def set_defaults(self, fov, camera_matrix, dist_coefs, length):
        self.camera_matrix = camera_matrix
        self.dist_coefs = dist_coefs
        self.length = length
        self.fov = fov
        #by default camera is centered on the world origin
        p = np.array([0., 0., 0.]).reshape(3, 1)
        #by default in opencv the camera faces along the z axis
        #transforming opencv world frame to our world frame automatically puts the camera along the x axis
        #thus we use a unit quaternion for the default camera orientation
        r = np.array([0., 0., 0., 1.])
        pts = None
        for pt in [(0,0), (0, 1), (1, 1), (1, 0)]:
            p_ = self.trans_C_W(p, r, pt).reshape(3, 1)
            if pts is None:
                pts = p_
            else:
                pts = np.hstack((pts, p_))
        self.proto_corners = pts
        self.proto_position = p
        self.proto_rotation = r
        self.update(p, r)

    def update(self, p, r):
        p = p.flatten()
        r = Rotation.from_quat(r)
        self._position = self.proto_position + p.flatten()
        self._rotation = (r * Rotation.from_quat(self.proto_rotation)).as_quat()
        corners = []
        for i in range(self.proto_corners.shape[1]):
            pt = self.proto_corners[:,i]
            corners.append(r.apply(pt) + p)
        corners = np.array(corners).T
        self._corners = corners
        frame = np.hstack((corners, corners[:, :1]))
        for i in range(1, corners.shape[1]):
            frame = np.hstack((frame, p.reshape(3, 1)))
            frame = np.hstack((frame, corners[:, i:i+1]))
        self._frame = frame
