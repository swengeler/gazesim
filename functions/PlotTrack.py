import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


class PlotTrack(object):
    '''
    # plotting the gates
    '''
    def __init__(self, df, col='r', lw=4):
        self.df = df
        self.col = col
        self.lw = lw

    def make_gate(self, r):
        g = np.array([
            (-r[0], 0., -r[2]),
            (-r[0], 0., r[2]),
            (r[0], 0., r[2]),
            (r[0], 0., -r[2]),
            (-r[0], 0., -r[2])])
        return g

    def rotate_gate(self, g, quat):
        r1 = Rotation.from_quat(quat)
        r2 = Rotation.from_euler('z', [90], degrees=True)  # WEIRD! needs extra 90 deg z-axis rotation (WHY?)
        return r2.apply(r1.apply(g))

    def translate_gate(self, g, trans):
        return np.array([x + np.array(trans) for x in g])

    def plot_gate(self, ax, g):
        ax.plot([x[0] for x in g],
                [x[1] for x in g],
                [x[2] for x in g], c=self.col, linewidth=self.lw)
        return ax

    def get_gates(self):
        gates = []
        for i in range(self.df.shape[0]):
            g = self.make_gate(self.df.iloc[i][['dim_x', 'dim_y', 'dim_z']].values)
            # g = self.rotate_gate(g, self.df.iloc[i][['rot_x_deg', 'rot_y_deg', 'rot_z_deg']].values)
            g = self.rotate_gate(g, self.df.iloc[i][['rot_x_quat', 'rot_y_quat', 'rot_z_quat', 'rot_w_quat']].values)
            g = self.translate_gate(g, self.df.iloc[i][['pos_x', 'pos_y', 'pos_z']].values)
            gates.append(g)
        return gates

    def plot_surface(self, ax, w=30):
        # https://stackoverflow.com/questions/36060933/matplotlib-plot-a-plane-and-points-in-3d-simultaneously
        # w = 30    # surface half width in x y, centered on (0,0)
        h = 0  # offset in height
        x, y = np.meshgrid(np.linspace(-w, w, 2 * w + 1), np.linspace(-w, w, 2 * w + 1))
        z = np.ones(x.shape) * h
        ax.plot_surface(x, y, z, color='grey', alpha=0.9, antialiased=True)
        return ax

    def run(self, view=(10, 45), figsize=(15, 15), t=[0, 0, 0], w=30, show_ground=True):

        # w = 30 # half width of view and plane
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        if show_ground:
            ax = self.plot_surface(ax, w)
        gates = self.get_gates()
        for g in gates:
            ax = self.plot_gate(ax, g)

        ax.set_xlim((-w + t[0], w + t[0]))
        ax.set_ylim((-w + t[1], w + t[1]))
        ax.set_zlim((-w + t[2], w + t[2]))
        ax.invert_zaxis()

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.view_init(view[0], view[1])

        return (fig, ax)

