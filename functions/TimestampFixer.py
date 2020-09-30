import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d

try:
    from functions.GatePassingEventDetector import *
except:
    from src.functions.GatePassingEventDetector import *


class TimestampFixer(object):

    def __init__(self, PATH, sr=1000, win=(-2,2), sd=20, plot_cost=False, plot_results=False):
        self.PATH = PATH
        E = pd.read_csv(PATH + 'events.csv')
        D = pd.read_csv(PATH + 'drone.csv')
        self.utc0 = D['utc_timestamp'].iloc[0] #utc timestamp of the start of drone.csv logfile
        self.events_ground_truth = E['ts'].values #gate pasing events from drone data, time relative to start of logging
        self.sr = sr
        self.win = win
        self.sd = sd
        self.plot_cost = plot_cost
        self.plot_results = plot_results

    def run(self, files=['screen', 'surface']):
        #only proceed if at least two ground truth events are available
        if self.events_ground_truth.shape[0] < 2:
            print('..skipping processing, not enough ground truth events')
        else:
            #loop over different filetypes
            for file in files:
                #check if video input file exists
                if os.path.isfile(self.PATH + file + '.mp4'):
                    #detect gate passing events from video frames
                    print('..processing {}'.format(self.PATH + file + '.mp4'))
                    obj = GatePassingEventDetector(self.PATH + file + '.mp4')
                    obj.run()
                    events_to_fix = obj.event_times
                    #continue only of there at least some events to fix
                    if events_to_fix.shape[0] < 2:
                        print('..skipping processing, too few events_to_fix')
                    else:
                        #perform correlation analysis to determine the time offset between timestamps and groundtruth
                        ts_offset, r = self.time_offset_from_correlation(self.events_ground_truth, events_to_fix, sr=self.sr,
                                                                         win=self.win, sd=self.sd)
                        #update the video timestamps
                        if file == 'surface':
                            currfile = 'world'
                        else:
                            currfile = file
                        T = pd.read_csv(self.PATH + currfile + '_timestamps.csv')
                        T['ts'] = obj.times + ts_offset
                        T['utc_timestamp'] = T['ts'] + self.utc0
                        T.to_csv(self.PATH + currfile + '_timestamps.csv', index=False)
                        #for surface video in addition correct the gaze data files
                        if file == 'surface':
                            for currfile in ['gaze_positions', 'gaze_on_surface', 'pupil_positions']:
                                G = pd.read_csv(self.PATH + currfile + '.csv')
                                if currfile == 'pupil_positions':
                                    tsname = 'pupil_timestamp'
                                else:
                                    tsname = 'gaze_timestamp'
                                G['utc_timestamp'] = G[tsname].values - T['world_timestamp'].iloc[0] + T['utc_timestamp'].iloc[0]
                                G['ts'] = G['utc_timestamp'].values - self.utc0
                                G.to_csv(self.PATH + currfile + '.csv', index=False)

    def convolve(self, e, t, sd=20):
        period = np.median(np.diff(t)) #period in sec, time between subsequent time bins
        #bin the event timestamps
        x = np.zeros(t.shape)
        for _e in e:
            x[(t >= _e) & (t < (_e + period))] = 1.
        #convolve the binned timestamps with gaussians
        x_filt = gaussian_filter1d(x, sd)
        #normalize to amplitude of 1
        x_filt = x_filt / np.max(x_filt)
        return (x, x_filt)

    def time_offset_from_correlation(self, events_gt, events_oi, sr=1000, win=(-2., 2.), sd=20):
        #make the time vector for binning the events (use higher resolution than in any of the original data)
        t_start = np.min([np.min(events_gt), np.min(events_oi)]) + win[0]
        t_end = np.max([np.max(events_gt), np.max(events_oi)]) + win[1]
        time = np.arange(t_start, t_end, 1 / sr)
        #convolve ground truth timestamp bins with gaussians
        _, gt_filt = self.convolve(events_gt, time, sd)
        #loop over a time offset window
        time_offsets = np.arange(win[0], win[1], 1 / sr)
        r = np.zeros(time_offsets.shape)
        r[:] = np.nan
        for offset in time_offsets:
            #shift the events by the current time offset
            curr_events = events_oi + offset
            #bin the events and convolve them with gaussian
            _, curr_filt = self.convolve(curr_events, time, sd)
            #compute the correlation coefficient
            corr = np.corrcoef(gt_filt, curr_filt)[0, 1]
            r[time_offsets == offset] = corr
            print('..computing correlation for offset = {:.3f} sec, r = {:.3f}'.format(offset, corr), end='\r')
        #determine peak correlation coefficient and time of peak correlation
        peak_offset = time_offsets[np.argmax(r)]
        peak_corr = np.max(r)
        #plot the cost across time offsets and the peak correlation
        if self.plot_cost:
            plt.figure(figsize=(15, 5))
            plt.plot(time_offsets, r)
            plt.plot(peak_offset, peak_corr, 'ro')
            plt.title('Events Peak Correlation, r = {:.3f} at time offset = {:.3f} sec'.format(peak_corr, peak_offset))
        #plot an overlay of the ground truth and the corrected timestamps
        if self.plot_results:
            final_events = events_oi + peak_offset
            final, final_filt = self.convolve(final_events, time, sd)
            plt.figure(figsize=(15, 5))
            plt.plot(time, gt_filt)
            plt.plot(time, final_filt)
            plt.ylim((0, np.max(gt_filt)))
            plt.legend(('ground_truth', 'corrected'))
            plt.title('Events after Correction')
        #return the determined time offset and peak correlation coefficient
        return (peak_offset, peak_corr)
