import os
import cv2
import numpy as np

class GatePassingEventDetector(object):

    def __init__(self, PATH):
        self.cap = cv2.VideoCapture(PATH)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # region of interest for gate passing trigger on 800x600 image [y=520:580, x=20:780], but here express it in
        # proportion to the actual image size
        self.roi = [int(0.85 * h), int(0.95 * h), int(0.05 * w), int(0.95 * w)]
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frames = np.arange(0, self.num_frames, 1)
        self.times = self.frames / self.fps
        self.threshold = 2.
        self._costs = np.empty((self.num_frames,))
        self._costs[:] = np.nan
        self._event_frames = np.array([])
        self._event_times = np.array([])

    @property
    def costs(self):
        return self._costs

    @property
    def event_frames(self):
        return self._event_frames

    @property
    def event_times(self):
        return self._event_times

    def fetch_image(self, f):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        _, im = self.cap.read()
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    def cost_from_image(self, im):
        #this only works for images of the AlphaPilot Simulator, which display visual signal, gray trigger surface
        #on the bottom third of the screen, when the drone passes the gate
        #Here we assume a screen resolution of 800x600 pixels, and we remove some of the border area +-20 pixels
        #because the surface images are not always croppec perfectly
        #the const function consists of the mean RGB pixel intensity across all pixels
        return np.mean(np.median(im[self.roi[0] : self.roi[1], self.roi[2] : self.roi[3], :].reshape(-1, 3), axis=0).flatten())

    def compute_cost(self):
        for f in range(self.num_frames):
            print('..computing cost for frame {}/{} [{:.2f}%]'.format(f, self.num_frames, 100 * f / self.num_frames), end='\r')
            self._costs[f] = self.cost_from_image(self.fetch_image(f))

    def normalize(self, x):
        if np.nanstd(x) == 0:
            return x
        else:
            return (x - np.nanmedian(x)) / np.nanstd(x)

    def detect_above_threshold_onsets(self, x, t):
        return np.hstack((0, (np.diff((x > t).astype(int)) > 0).astype(int)))

    def detect_events(self):
        print('..detecting events.', end='\r')
        norm_cost = self.normalize(self._costs)
        onsets = self.detect_above_threshold_onsets(norm_cost, self.threshold)
        self._event_frames = self.frames[onsets==1]
        self._event_times = self.times[onsets==1]

    def run(self):
        self.compute_cost()
        self.detect_events()
        print('Done.                                                      ')
