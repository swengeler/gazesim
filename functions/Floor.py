import numpy as np
from shapely.geometry import LineString
from scipy.spatial.transform import Rotation


class Floor(object):

    def __init__(self, x=(-30., 30.), y=(-15., 15.), z=0.):
        self._corners = np.array([[x[0], x[0], x[1], x[1], x[0]],
                                  [y[0], y[1], y[1], y[0], y[0]],
                                  [z,    z,    z,    z,    z]  ]) #assume surface, thus no thickness
        self._center = np.array([x[1]-x[0], y[1]-y[0], z])
        self._rotation = np.array([0.,0.,0.,1.])
        #gates are represented as 2d lines (as if seen from the top)
        self.xz = LineString([(self._corners[0, 0], self._corners[2, 0]), (self._corners[0, 2], self._corners[2, 2])])
        self.yz = LineString([(self._corners[0, 1], self._corners[2, 1]), (self._corners[0, 2], self._corners[2, 2])])

    @property
    def corners(self):
        return self._corners

    @property
    def center(self):
        return self._center

    @property
    def rotation(self):
        return self._rotation

    def intersect(self, p0, p1):
        point_2d = None
        point_3d = None
        line_xz = LineString([(p0[0], p0[2]), (p1[0], p1[2])])
        line_yz = LineString([(p0[1], p0[2]), (p1[1], p1[2])])
        if (self.xz.intersects(line_xz)) and (self.yz.intersects(line_yz)):
            xz = [val for val in self.xz.intersection(line_xz).coords]
            if len(xz) == 2:
                ind = np.argmin(np.array([np.linalg.norm(np.array(xz[0]) - p0[[0, 2]]),
                                          np.linalg.norm(np.array(xz[1]) - p0[[0, 2]])]))
            else:
                ind = 0
            p_xz = xz[ind]

            yz = [val for val in self.yz.intersection(line_yz).coords]
            if len(yz) == 2:
                ind = np.argmin(np.array([np.linalg.norm(np.array(yz[0]) - p0[[1, 2]]),
                                          np.linalg.norm(np.array(yz[1]) - p0[[1, 2]])]))
            else:
                ind = 0
            p_yz = yz[ind]

            point_3d = np.array([p_xz[0], p_yz[0], self._center[2]])
            point_2d = np.array([p_xz[0], p_yz[0]])

        return point_2d, point_3d
