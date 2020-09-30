import os
import cv2
import msgpack
import pandas as pd
import numpy as np


class GazeImporter(object):

    def __init__(self, path, export_images=['surface_und_overlay'], method='timestamp',
                 overwrite=False, export_gaze=True, export_frames=True, export_video=True):
        self.export_gaze = export_gaze
        self.export_frames = export_frames
        self.export_video = export_video
        self.overwrite = overwrite
        self.rootpath = path
        self.set_valid_paths(path)
        self.set_method(method)
        self.set_export_images(export_images)

    def set_valid_paths(self, path):
        paths = []
        for maindir in os.walk(path):
            if maindir[0] == path:
                for subdir in sorted(maindir[1]):
                    inpath = ''.join([maindir[0], subdir, '/gaze/'])
                    outpath = ''.join([maindir[0], subdir, '/gaze/'])
                    if os.path.exists(inpath) and (self.overwrite or not os.path.isfile(outpath[:-1] + '.csv')):
                        paths.append((inpath, outpath))
        count = 0
        if len(paths) > 0:
            print('Processing options:')
            for p in paths:
                print('key [{}]: {}'.format(count, '/'.join(p[1].split('/')[:-1])))
                count += 1
            print()
        else:
            print('Processing done.')
        self.valid_paths = paths

    def set_export_images(self, lst):
        self.export_images = lst

    def set_method(self, s):
        self.method = s

    # simple video to image
    def video_to_image(self, vidcap, frame):
        # important: every vidcap.read() call raises the frame counter
        # frame counter starts from 1 (not zero)
        # frame is relative to the .mp4 file, not the same as in gaze_positions.csv
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        success, image = vidcap.read()
        return image

    # get a dictionairy of properties for video and current frame
    def video_properties(self, vidcap, frame=None):
        if frame is not None:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        prop_names = [
            'POS_MSEC', 'POS_FRAMES', 'POS_AVI_RATIO', 'FRAME_WIDTH',
            'FRAME_HEIGHT', 'FPS', 'FOURCC', 'FRAME_COUNT', 'FORMAT',
            'MODE', 'BRIGHTNESS', 'CONTRAST', 'SATURATION', 'HUE',
            'GAIN', 'EXPOSURE', 'CONVERT_RGB', 'WHITE_BALANCE_U',
            'WHITE_BALANCE_V', 'RECTIFICATION', 'ISO_SPEED', 'BUFFERSIZE']
        prop = {}
        for i in range(len(prop_names)):
            prop[prop_names[i].lower()] = vidcap.get(i)
        return prop

    # undistort world image
    def undistort_world_image(self, image, k, d):
        # image: distorted world image
        # k: camera_matrix
        # d: distortion coefficients
        # image_und: undistorted world image
        image_und = cv2.undistort(
            image,
            k,
            d,
            None,
            k)
        return image_und

    # undistort points
    def undistort_points(self, p, k, d):
        # p: distorted points, ndarray of 2d points, n x 2
        # k: camera_matrix
        # d: distortion coefficients
        # p_und: undistorted points, ndarray of 2d points, n x 2
        if len(p) != 0:
            p_und = cv2.undistortPoints(
                p.reshape(-1, 1, 2).astype(np.float32),
                k,
                d,
                P=k).reshape(-1, 2)
        else:
            p_und = p
        return p_und

    # make the transform from world to surface
    def make_transform_W_S(self, t, width=800, height=600):
        # t: transform surface to world
        # surface borders in surface coordinates (S, 0-1 range)
        surfborders_S = np.array(
            ((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32)
        # transform surface borders to world coordinates (W, 800x600 pixels)
        surfborders_W = cv2.perspectiveTransform(
            surfborders_S.reshape(-1, 1, 2),
            t).reshape(-1, 2)
        # world borders in world coordinates
        worldborders_W = np.array(((0, height - 1), (width - 1, height - 1), (width - 1, 0), (0, 0)), dtype=np.float32)
        # transform from world to surface frame in World coordinates
        tf_W_S = cv2.getPerspectiveTransform(surfborders_W, worldborders_W)
        return tf_W_S

    # transform points
    def transform_points_world_to_surface(self, p, t, width=800, height=600):
        # p: points in world frame, ndarray of 2d points, n x 2
        # t: transform world to surface
        # surface borders in surface coordinates (S, 0-1 range)
        surfborders_S = np.array(
            ((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32)
        # transform surface borders to world coordinates (W, 800x600 pixels)
        surfborders_W = cv2.perspectiveTransform(
            surfborders_S.reshape(-1, 1, 2),
            t).reshape(-1, 2)
        # world borders in world coordinates
        worldborders_W = np.array(((0, height - 1), (width - 1, height - 1), (width - 1, 0), (0, 0)), dtype=np.float32)
        # transform from world to surface frame in World coordinates
        tf_W_S = cv2.getPerspectiveTransform(surfborders_W, worldborders_W)
        # transform surface borders to world coordinates (W, e.g. 800 x 600 pixels)
        if len(p) != 0:
            p_trans = cv2.perspectiveTransform(
                p.reshape(-1, 1, 2).astype(np.float32),
                tf_W_S).reshape(-1, 2)
        else:
            p_trans = p
        return p_trans

    # image and transform to surface image frame
    def world_image_to_surface_image(self, image, transform):
        # surface borders in surface coordinates (S, 0-1 range)
        surfborders_S = np.array(
            ((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32)
        # transform surface borders to world coordinates (W, 800x600 pixels)
        surfborders_W = cv2.perspectiveTransform(
            surfborders_S.reshape(-1, 1, 2),
            transform).reshape(-1, 2)
        # get the image width and height
        height, width, _ = image.shape
        # world borders in world coordinates
        worldborders_W = np.array(((0, height - 1), (width - 1, height - 1), (width - 1, 0), (0, 0)), dtype=np.float32)
        # transform from world to surface frame in World coordinates
        tf_W_S = cv2.getPerspectiveTransform(surfborders_W, worldborders_W)
        # apply transform
        image_surface = cv2.warpPerspective(image, tf_W_S, (width, height))
        return image_surface

    # video gaze overlay
    def image_points_overlay(self, image, points):
        image_modified = image.copy()
        if len(points) > 0:
            for x_, y_ in points:
                cv2.circle(image_modified, (int(x_), int(y_)), 5, (255, 0, 0), -1)
        return image_modified

    # video gaze overlay
    def video_gaze_overlay(self, vidcap, gaze, outfilepathname=None, distorted=False):
        # default filepath and name for output video
        if outfilepathname is None:
            outfilepath = os.getcwd() + '/'
            outfilepathname = outfilepath + '/noname.mp4'
        else:
            outfilepath = '/'.join(outfilepathname.split('/')[:-1]) + '/'
        # make the output path
        self.make_outpath(outfilepath)
        # read the first video frame
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, image = vidcap.read()
        # get some important information about the video
        width = vidcap.get(3)  # image width
        height = vidcap.get(4)  # image height
        fps = vidcap.get(5)  # frames per second
        fourcc = vidcap.get(6)  # fourcc codec integer identifier
        num_frames = vidcap.get(7)  # number of frames
        is_color = True  # whether to use color
        if distorted:
            df = gaze[['gaze_timestamp', 'world_index', 'norm_pos_x', 'norm_pos_y']]
            df.columns = ['ts', 'frame', 'x', 'y']
            df['x'] = df['x'] * width
            df['y'] = (1.0 - df['y']) * height
        else:
            df = gaze[['GazeTimeStamp', 'MediaFrameIndex', 'Gaze2dX', 'Gaze2dY']]
            df.columns = ['ts', 'frame', 'x', 'y']
        # video writer objecgt
        vidwriter = cv2.VideoWriter(
            outfilepathname,
            int(fourcc),
            fps,
            (int(width), int(height)),
            is_color)
        # loop over video frames
        for frame_vid in range(int(2000)):
            # for frame_vid in range(int(num_frames)):
            # read current frame
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_vid)
            success, image = vidcap.read()
            # find gaze positions for this frame
            x = df[df['frame'] == frame_vid]['x'].values
            y = df[df['frame'] == frame_vid]['y'].values
            image_modified = image.copy()
            print('frame: {}/{}, tf: {}'.format(frame_vid, num_frames, True))
            if len(x) > 0:
                # plot overlay
                # copy the raw image and draw an overlay
                for x_, y_ in zip(x, y):
                    cv2.circle(image_modified, (int(x_), int(y_)), 5, (255, 0, 0), -1)
            # write frame to output video
            vidwriter.write(image_modified)
        # close the video writer
        vidwriter.release()

    # modifies world camera video based on surface transforms, either adds overlay
    # of surface, or changes video to become surface image
    def world_video_to_surface_video(self, vidcap, transform, outfilepathname=None,
                                     distorted=False, to_surface=False, method='frame', frame_range=None):
        # default filepath and name for output video
        if outfilepathname is None:
            outfilepath = os.getcwd() + '/'
            outfilepathname = outfilepath + '/noname.mp4'
        else:
            outfilepath = '/'.join(outfilepathname.split('/')[:-1]) + '/'
        # make the output path
        self.make_outpath(outfilepath)
        # name of the transform to use
        if distorted:
            tf_name = 'surf_to_dist_img_trans'
        else:
            tf_name = 'surf_to_img_trans'
        # undistorted video
        df = transform[['world_index', 'world_timestamp', tf_name]]
        df.columns = ['frame', 'ts', 'trans']
        # read the first video frame
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, image = vidcap.read()
        # get some important information about the video
        t0_vid_ms = vidcap.get(0)  # time of video start
        width = vidcap.get(3)  # image width
        height = vidcap.get(4)  # image height
        fps = vidcap.get(5)  # frames per second
        fourcc = vidcap.get(6)  # fourcc codec integer identifier
        num_frames = vidcap.get(7)  # number of frames
        is_color = True  # whether to use color
        # define acceptable window in ms where to search for surface frames
        win_ms = (1000.0 / fps) / 2.0  # half a period
        # time of surface start
        t0_surf_sec = df[df['frame'] == 0]['ts'].values[0]
        # add exptime
        df['exptime_ms'] = (df['ts'] - t0_surf_sec) * 1000
        # video writer objecgt
        vidwriter = cv2.VideoWriter(
            outfilepathname,
            int(fourcc),
            fps,
            (int(width), int(height)),
            is_color)
        # if no frame range was specified
        if frame_range is None:
            frame_range = [0, int(num_frames)]
        # loop over video frames
        for frame_vid in range(frame_range[0], frame_range[1]):
            # read current frame
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_vid)
            success, image = vidcap.read()
            # time of current frame
            t_vid_ms = vidcap.get(0)
            # time delta from start of video
            exptime_vid_ms = t_vid_ms - t0_vid_ms
            # find the closest frames based on time
            ind = (
                    (df['exptime_ms'] > (exptime_vid_ms - win_ms)) &
                    (df['exptime_ms'] < (exptime_vid_ms + win_ms)))
            # if surface was found in the search window..
            if np.sum(ind) > 0:
                print('frame: {}/{}, tf: {}, method: {}'.format(frame_vid, frame_range[1], True, method))
                # get the frame of the surface
                # if time-based method..
                if method == 'time':
                    frame_surf = df[ind]['frame'].values[0]
                    # otherwise use frame-based method..
                else:
                    frame_surf = frame_vid
                # get transform
                tf_S_W = df[df['frame'] == frame_surf]['trans'].values[0]
                # modify frame:
                # either to surface frame..
                if to_surface:
                    image_modified = self.world_image_to_surface_image(image, tf_S_W)
                # or to world frame with surface overlay..
                else:
                    image_modified = self.world_image_to_world_image_with_surfborder(image, tf_S_W, frame_vid)
            # if no suface was found in search window..
            else:
                print('frame: {}/{}, tf: {}, method: {}'.format(frame_vid, frame_range[1], False, method))
                # modified image is the original image
                image_modified = image.copy()
            # write frame to output video
            vidwriter.write(image_modified)
        # close the video writer
        vidwriter.release()

    def make_outpath(self, path):
        outpath = '/'
        folders = path.split('/')
        for fold in folders:
            if len(fold) > 0:
                outpath += fold + '/'
                if os.path.isdir(outpath) == False:
                    os.mkdir(outpath)

    # surface transform from sting to (3,3) numpy array
    def transform_string_to_array(self, s):
        # remove unnecessary characters
        s = ('[' +
             s.replace('[', '')
             .replace(']', '')
             .replace('\n', '')
             .replace('  ', ' ')
             .strip()
             .replace(' ', ',')
             + ']')
        # convert to numpy array
        v = np.array(eval(s)).reshape((3, 3))
        return v

    # opencv video capture object and surface transform to overlay image frame
    def world_video_to_world_image_with_surfborder(self, vidcap, transform, frame, col=(256, 0, 0), lw=10):
        # load video frame
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        success, image = vidcap.read()
        # convert to RGB color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # surface borders in surface coordinates (S, 0-1 range)
        surfborders_S = np.array(
            ((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32)
        # transform surface borders to world coordinates (W, e.g. 800 x 600 pixels)
        surfborders_W = cv2.perspectiveTransform(
            surfborders_S.reshape(-1, 1, 2),
            transform).reshape(-1, 2)
        # copy the raw image and draw an overlay
        image_overlay = image.copy()
        cv2.line(image_overlay, tuple(surfborders_W[0]), tuple(surfborders_W[1]), col, lw)
        cv2.line(image_overlay, tuple(surfborders_W[1]), tuple(surfborders_W[2]), col, lw)
        cv2.line(image_overlay, tuple(surfborders_W[2]), tuple(surfborders_W[3]), col, lw)
        cv2.line(image_overlay, tuple(surfborders_W[3]), tuple(surfborders_W[0]), col, lw)
        # return image frame and image frame with border overlay
        return image, image_overlay

    # world image and surface transform to overlay image frame
    def world_image_to_world_image_with_surfborder(self, image, transform, frame, col=(256, 0, 0), lw=10):
        # surface borders in surface coordinates (S, 0-1 range)
        surfborders_S = np.array(
            ((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32)
        # transform surface borders to world coordinates (W, e.g. 800 x 600 pixels)
        surfborders_W = cv2.perspectiveTransform(
            surfborders_S.reshape(-1, 1, 2),
            transform).reshape(-1, 2)
        # copy the raw image and draw an overlay
        image_overlay = image.copy()
        cv2.line(image_overlay, tuple(surfborders_W[0]), tuple(surfborders_W[1]), col, lw)
        cv2.line(image_overlay, tuple(surfborders_W[1]), tuple(surfborders_W[2]), col, lw)
        cv2.line(image_overlay, tuple(surfborders_W[2]), tuple(surfborders_W[3]), col, lw)
        cv2.line(image_overlay, tuple(surfborders_W[3]), tuple(surfborders_W[0]), col, lw)
        # return image frame and image frame with border overlay
        return image_overlay

    # opencv video capture object and transform to surface image frame
    def world_video_to_surface_image(self, vidcap, transform, frame):
        # load video frame
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        success, image = vidcap.read()
        # convert to RGB color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # surface borders in surface coordinates (S, 0-1 range)
        surfborders_S = np.array(
            ((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32)
        # transform surface borders to world coordinates (W, 800x600 pixels)
        surfborders_W = cv2.perspectiveTransform(
            surfborders_S.reshape(-1, 1, 2),
            transform).reshape(-1, 2)
        # get the image width and height
        height, width, _ = image.shape
        # world borders in world coordinates
        worldborders_W = np.array(((0, height - 1), (width - 1, height - 1), (width - 1, 0), (0, 0)), dtype=np.float32)
        # transform from world to surface frame in World coordinates
        tf_W_S = cv2.getPerspectiveTransform(surfborders_W, worldborders_W)
        # apply transform
        image2 = cv2.warpPerspective(image, tf_W_S, (width, height))
        return image, image2

    # fix the world index
    def fix_world_index(self, vidcap, wt, gp, tf, method='timestamp'):
        if method == 'timestamp':
            ts_video_start = wt.iloc[0][0] - (0.5 * (1 / vidcap.get(5)))
            gp['world_index'] = np.floor((gp['gaze_timestamp'] - ts_video_start) / (1 / vidcap.get(5)))
            tf['world_index'] = np.floor((tf['world_timestamp'] - ts_video_start) / (1 / vidcap.get(5)))
        else:
            idx_video_start = gp['world_index'].min()
            gp['world_index'] -= idx_video_start
            tf['world_index'] -= idx_video_start
        return (gp, tf)

    # export image update
    def update_gazedata(self, frame, gaze_wd, tf, camera_matrix, dist_coefs, width, height):
        # prepare the output table
        df = gaze_wd.copy()
        df = df[df['world_index'] == frame]
        df['width'] = width
        df['height'] = height
        for val1 in ['x', 'y']:
            for val2 in ['wd', 'wu', 'sd', 'su']:
                df[val1 + '_' + val2] = np.nan
        # World Distorted + Surface
        trans_dist = tf[tf['world_index'] == frame]['surf_to_dist_img_trans'].values
        if len(trans_dist) > 0:
            trans_dist = trans_dist[0]
        # World Distorted + Surface Gaze
        points_dist = df.copy()
        points_dist = points_dist[['norm_pos_x', 'norm_pos_y']]
        points_dist.columns = ['x', 'y']
        points_dist['x'] = points_dist['x'] * width
        points_dist['y'] = (1.0 - points_dist['y']) * height
        points_dist = points_dist.values
        df['x_wd'] = points_dist[:, 0]
        df['y_wd'] = points_dist[:, 1]
        # Surface Distorted + Gaze
        if len(trans_dist) > 0:
            points_sd = self.transform_points_world_to_surface(points_dist, trans_dist, width, height)
            df['x_sd'] = points_sd[:, 0]
            df['y_sd'] = points_sd[:, 1]
        # World Undistorted + Surface
        trans_und = tf[tf['world_index'] == frame]['surf_to_img_trans'].values
        if len(trans_und) > 0:
            trans_und = trans_und[0]
        # World Undistorted + Surface Gaze
        points_und = self.undistort_points(points_dist, camera_matrix, dist_coefs)
        df['x_wu'] = points_und[:, 0]
        df['y_wu'] = points_und[:, 1]
        # undistorted surface image plus overlay
        if len(trans_und) > 0:
            points_su = self.transform_points_world_to_surface(points_und, trans_und, width, height)
            df['x_su'] = points_su[:, 0]
            df['y_su'] = points_su[:, 1]
        return df

    # export image update
    def update_images(self, frame, video_wd, gaze_data, tf, camera_matrix, dist_coefs, width, height):
        fn = 'frame%08d.jpg' % frame
        # prepare the output table
        df = gaze_data.copy()
        df = df[df['world_index'] == frame]
        # World Distorted
        video_wd.set(cv2.CAP_PROP_POS_FRAMES, frame)
        (_, image_dist) = video_wd.read()
        # proceed only if valid image is available
        if image_dist is not None:
            if 'world_dist' in self.export_images:
                fpn = self.outpath + 'world_dist/' + fn
                cv2.imwrite(fpn, image_dist)
            # World Distorted + Surface
            trans_dist = tf[tf['world_index'] == frame]['surf_to_dist_img_trans'].values
            if len(trans_dist) > 0:
                has_trans_dist = True
                trans_dist = trans_dist[0]
                image_dist_surf = self.world_image_to_world_image_with_surfborder(image_dist, trans_dist, frame)
            else:
                has_trans_dist = False
                image_dist_surf = image_dist.copy()
            if 'world_dist_surf' in self.export_images:
                fpn = self.outpath + 'world_dist_surf/' + fn
                cv2.imwrite(fpn, image_dist_surf)
            # World Distorted + Surface Gaze
            points_dist = df[['x_wd', 'y_wd']].values
            image_dist_surf_overlay = self.image_points_overlay(image_dist_surf, points_dist)
            if 'world_dist_surf_overlay' in self.export_images:
                fpn = self.outpath + 'world_dist_surf_overlay/' + fn
                cv2.imwrite(fpn, image_dist_surf_overlay)
            # Surface Distorted
            if has_trans_dist:
                image_sd = self.world_image_to_surface_image(image_dist, trans_dist)
            else:
                image_sd = np.zeros((width, height, 3), np.uint8)
                image_sd[:] = (0, 0, 0)  # BGR
            if 'surface_dist' in self.export_images:
                fpn = self.outpath + 'surface_dist/' + fn
                cv2.imwrite(fpn, image_sd)
            # Surface Distorted + Gaze
            if has_trans_dist:
                points_sd = df[['x_sd', 'y_sd']].values
                image_sd_overlay = self.image_points_overlay(image_sd, points_sd)
            else:
                image_sd_overlay = np.zeros((width, height, 3), np.uint8)
                image_sd_overlay[:] = (0, 0, 0)  # BGR
            if 'surface_dist_overlay' in self.export_images:
                fpn = self.outpath + 'surface_dist_overlay/' + fn
                cv2.imwrite(fpn, image_sd_overlay)
            # World Undistorted
            image_und = self.undistort_world_image(image_dist, camera_matrix, dist_coefs)
            if 'world_und' in self.export_images:
                fpn = self.outpath + 'world_und/' + fn
                cv2.imwrite(fpn, image_und)
            # World Undistorted + Surface
            trans_und = tf[tf['world_index'] == frame]['surf_to_img_trans'].values
            if len(trans_und) > 0:
                has_trans_und = True
                trans_und = trans_und[0]
                image_und_surf = self.world_image_to_world_image_with_surfborder(image_und, trans_und, frame)
            else:
                has_trans_und = False
                image_und_surf = image_und.copy()
            if 'world_und_surf' in self.export_images:
                fpn = self.outpath + 'world_und_surf/' + fn
                cv2.imwrite(fpn, image_und_surf)
            # World Undistorted + Surface Gaze
            points_und = df[['x_wu', 'y_wu']].values
            image_und_surf_overlay = self.image_points_overlay(image_und_surf, points_und)
            if 'world_und_surf_overlay' in self.export_images:
                fpn = self.outpath + 'world_und_surf_overlay/' + fn
                cv2.imwrite(fpn, image_und_surf_overlay)
            # Surface Undistorted
            if has_trans_und:
                image_su = self.world_image_to_surface_image(image_und, trans_und)
            else:
                image_su = np.zeros((width, height, 3), np.uint8)
                image_su[:] = (0, 0, 0)  # BGR
            if 'surface_und' in self.export_images:
                fpn = self.outpath + 'surface_und/' + fn
                cv2.imwrite(fpn, image_su)
            # undistorted surface image plus overlay
            if has_trans_und:
                points_su = df[['x_su', 'y_su']].values
                image_su_overlay = self.image_points_overlay(image_su, points_su)
            else:
                image_su_overlay = np.zeros((width, height, 3), np.uint8)
                image_su_overlay[:] = (0, 0, 0)  # BGR
            if 'surface_und_overlay' in self.export_images:
                fpn = self.outpath + 'surface_und_overlay/' + fn
                cv2.imwrite(fpn, image_su_overlay)

    def process_data(self, inpath, outpath):
        self.inpath = inpath
        self.outpath = outpath
        self.make_outpath(self.outpath)
        print('inpath:  {}\noutpath: {}'.format(self.inpath, self.outpath))
        # root directory for pupillabs exported data
        rootpath = self.inpath
        # load distorted video
        fpn = rootpath + 'world.mp4'
        video_wd = cv2.VideoCapture(fpn)
        width = int(video_wd.get(3))
        height = int(video_wd.get(4))
        fps = video_wd.get(5)
        fourcc = int(video_wd.get(6))
        frames = int(video_wd.get(7)) - 1
        # load gaze distorted
        fpn = rootpath + 'gaze_positions.csv'
        gaze_wd = pd.read_csv(fpn)
        # load video timestamps
        fpn = rootpath + 'world_timestamps.csv'
        world_ts = pd.read_csv(fpn)
        # load surface data
        fpn = rootpath + 'surfaces/' + 'surf_positions_Surface 1.csv'
        tf = pd.read_csv(fpn)
        # fix world_index (either by timestamps or by setting first frame to zero)
        gaze_wd, tf = self.fix_world_index(video_wd, world_ts, gaze_wd, tf, method=self.method)
        # convert transform strings to numpy arrarys
        for c in [c for c in tf.columns if 'trans' in c]:
            tf[c] = tf[c].apply(self.transform_string_to_array)
        # load calibration based world camera intrinsics
        fpn = rootpath + 'world.intrinsics'
        with open(fpn, "rb") as fh:
            world_intrinsics = msgpack.unpack(fh, raw=False)
        camera_matrix = np.array(world_intrinsics['({}, {})'.format(width, height)]['camera_matrix']).reshape(3, 3)
        dist_coefs = np.array(world_intrinsics['({}, {})'.format(width, height)]['dist_coefs'])
        # Three processing steps:
        # Step 1: Make and save gaze data output table
        if self.export_gaze:
            gaze_data = pd.DataFrame()
            for frame in range(int(gaze_wd['world_index'].min()), int(gaze_wd['world_index'].max())):
                print('..processing gazedata: {:.2f} %'.format(100 * frame / frames), end='\r')
                df = self.update_gazedata(frame, gaze_wd, tf, camera_matrix, dist_coefs, width, height)
                gaze_data = gaze_data.append(df, ignore_index=True)
            fpn = self.outpath[:-1] + '.csv'
            print('')
            print('..saving gaze.csv')
            gaze_data.to_csv(fpn, index=False)
        # loop over video frames
        if len(self.export_images) > 0:
            for name in self.export_images:
                self.make_outpath(self.outpath + name + '/')
            # Step 2: Make and save images for export
            if self.export_frames:
                for frame in range(0, frames):
                    print('..saving images: {:.2f} %'.format(100 * frame / frames), end='\r')
                    self.update_images(frame, video_wd, gaze_data, tf, camera_matrix, dist_coefs, width, height)
            # Step 3: Make and save video from exported images
            if self.export_video:
                inpath = self.outpath + name + '/frame%08d.jpg'
                self.images_to_video(inpath, fourcc, fps, width, height)

    def images_to_video(self, inpath, fourcc, fps, width, height):
        outpath = '/'.join(inpath.split('/')[:-1]) + '.mp4'

    def run(self, indices=None):
        if indices is None:
            paths = self.valid_paths
        else:
            if isinstance(indices, int):
                indices = [indices]
            paths = []
            for i in indices:
                if i < len(self.valid_paths):
                    paths.append(self.valid_paths[i])
        if len(paths) > 0:
            for inpath, outpath in paths:
                self.process_data(inpath, outpath)
                print('Done.')

def main(inpath, outpath=None):
    gazeImporter = GazeImporter(inpath, outpath)
    gazeImporter.run()

if __name__=="__main__":
    main()

