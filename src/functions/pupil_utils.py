import sys
import os
import cv2
import csv
import msgpack
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def transform_string_to_array(s):
    '''
    surface transform from sting to (3,3) numpy array
    '''
    #remove unnecessary characters
    s = ('[' +
        s.replace('[','')
        .replace(']','')
        .replace('\n','')
        .replace('  ',' ')
        .strip()
        .replace(' ',',')
        + ']')
    #convert to numpy array
    v = np.array(eval(s)).reshape((3,3))
    return v

def undistort_world_image(im, K, D):
    '''
    # im: distorted world image
    # K: camera_matrix
    # D: distortion coefficients
    # im_u: undistorted image
    '''
    im_u = cv2.undistort(im, K, D, None, K)
    return im_u

def undistort_points(p, k, d):
    '''
    # p: distorted points, ndarray of 2d points, n x 2
    # k: camera_matrix
    # d: distortion coefficients
    # p_und: undistorted points, ndarray of 2d points, n x 2
    '''
    if len(p) != 0:
        p_und = cv2.undistortPoints(p.reshape(-1, 1, 2).astype(np.float32), k, d, P=k).reshape(-1, 2)
    else:
        p_und = p
    return p_und

# def make_transform_W_S(t, width=800, height=600):
#     # t: transform surface to world
#     # surface borders in surface coordinates (S, 0-1 range)
#     surfborders_S = np.array(
#         ((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32)
#     # transform surface borders to world coordinates (W, 800x600 pixels)
#     surfborders_W = cv2.perspectiveTransform(
#         surfborders_S.reshape(-1, 1, 2),
#         t).reshape(-1, 2)
#     # world borders in world coordinates
#     worldborders_W = np.array(((0, height - 1), (width - 1, height - 1), (width - 1, 0), (0, 0)), dtype=np.float32)
#     # transform from world to surface frame in World coordinates
#     tf_W_S = cv2.getPerspectiveTransform(surfborders_W, worldborders_W)
#     return tf_W_S

def transform_points_world_to_surface(p, t, width=800, height=600):
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

def world_image_to_surface_image(image, transform):
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

def image_points_overlay(image, points):
    image_modified = image.copy()
    if len(points) > 0:
        for x_, y_ in points:
            cv2.circle(image_modified, (int(x_), int(y_)), 5, (255, 0, 0), -1)
    return image_modified

#opencv video capture object and surface transform to overlay image frame
def world_image_to_world_image_with_surfborder(im, t, col=(256,0,0), lw=10):
    #surface borders in surface coordinates (S, 0-1 range)
    surfborders_S = np.array(
                  ((0,0),(1,0),(1,1),(0,1)),dtype=np.float32)
    #transform surface borders to world coordinates (W, e.g. 800 x 600 pixels)
    surfborders_W = cv2.perspectiveTransform(
                         surfborders_S.reshape(-1,1,2),
                         t).reshape(-1,2)
    #copy the raw image and draw an overlay
    image_overlay=im.copy()
    cv2.line(image_overlay, tuple(surfborders_W[0]), tuple(surfborders_W[1]), col, lw)
    cv2.line(image_overlay, tuple(surfborders_W[1]), tuple(surfborders_W[2]), col, lw)
    cv2.line(image_overlay, tuple(surfborders_W[2]), tuple(surfborders_W[3]), col, lw)
    cv2.line(image_overlay, tuple(surfborders_W[3]), tuple(surfborders_W[0]), col, lw)
    #return image frame and image frame with border overlay
    return image_overlay

def make_video(cap, wts, tf, gaze, frames, K, D, outpath=None, method='index', output='wdo', width=800, height=600,
               colname_index='world_index', colname_timestamp='world_timestamp'):
    fps = cap.get(5)
    fourcc = int(cap.get(6))
    if outpath is None:
        outpath = os.getcwd()+'/test.mp4'
    writer = cv2.VideoWriter(outpath, fourcc, fps, (width, height))
    print('..saving ' + outpath)
    for f in frames:
        print('..exporting {} [{:.2f} %]'.format(outpath, 100 * f / frames[-1]), end='\r')
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        _, im = cap.read()
        im = apply_transform_to_image(im, wts, tf, gaze, f, K, D, method=method, output=output, width=width, height=height,
                             colname_index=colname_index, colname_timestamp=colname_timestamp)
        writer.write(im)
    print('')
    writer.release()

def apply_transform_to_image(im, wts, tf, gaze, f, K, D, method='index', output='wdo', width=800, height=600,
                    colname_index='world_index', colname_timestamp='world_timestamp'):
    # selection method [for transform from tf]
    if method == 'index':
        ## Method_1: find the closest frame in tf to the desired frame [f]
        # wi = tf[colname_index].values
        # f_ = wi[np.argmin(np.abs(wi - f))]
        # curr_tf = tf.iloc[np.argmin(np.abs(tf[colname_index] - f_))]
        ## Method_2: find the latest frame in tf to the desired frame [f]
        #exception handling: if desired frame is smaller than frames in tf:
        if f<tf[colname_index].min():
            curr_tf = tf.iloc[0]
        else:
            curr_tf = tf.loc[tf[colname_index] <= f].iloc[-1]
        curr_gaze = gaze.loc[gaze[colname_index] == f]
    elif method == 'timestamp':  # find the closest world_timestamp
        timestamp = wts.loc[wts[colname_index] == f][colname_timestamp].values
        curr_tf = tf.iloc[np.argmin(np.abs(tf[colname_timestamp] - timestamp))]
        curr_gaze = gaze.loc[gaze[colname_index] == f]
    # output image tpe
    if output == 'wdo':  # world distorted overlay
        trans = curr_tf.surf_to_dist_img_trans
        im = world_image_to_world_image_with_surfborder(im, trans, lw=5)
    elif output == 'sd':  # surface distorted
        trans = curr_tf.surf_to_dist_img_trans
        im = world_image_to_surface_image(im, trans)
    elif output == 'wuo':  # world undistorted overlay
        im = undistort_world_image(im, K, D)
        trans = curr_tf.surf_to_img_trans
        im = world_image_to_world_image_with_surfborder(im, trans, lw=5)
    elif output == 'su':  # surface_undistorted
        im = undistort_world_image(im, K, D)
        trans = curr_tf.surf_to_img_trans
        im = world_image_to_surface_image(im, trans)
    # add gaze overlay
    if curr_gaze.size:
        p = curr_gaze[['norm_pos_x', 'norm_pos_y']].values
        p[:, 0] *= width
        p[:, 1] = (1. - p[:, 1]) * height
        if (output == 'sd') or (output == 'su'):
            p = transform_points_world_to_surface(p, trans)
        im = image_points_overlay(im, p)
    return im


def gaze_to_surface_mapping(wts, tf, gaze, frames, K, D, method='index', output='su', width=800, height=600,
                           colname_index='frame', colname_timestamp='ts'):

    gaze['norm_x_su'] = np.nan
    gaze['norm_y_su'] = np.nan
    gaze['x_su'] = np.nan
    gaze['y_su'] = np.nan
    gaze['screen_width'] = np.nan
    gaze['screen_height'] = np.nan
    #only proceed if some transforms available
    if tf.shape[0] > 0:
        for f in frames:
            print('..mapping gaze to surface [{:.2f} %]'.format(100 * f / frames[-1]), end='\r')
            # selection method [for transform from tf]
            if method == 'index':  # find the closest world_index
                ## Method_1: find the closest frame in tf to the desired frame [f]
                # wi = tf[colname_index].values
                # f_ = wi[np.argmin(np.abs(wi - f))]
                # curr_tf = tf.iloc[np.argmin(np.abs(tf[colname_index] - f_))]
                ## Method_2: find the latest frame in tf to the desired frame [f]
                # exception handling: if desired frame is smaller than frames in tf:
                if f < tf[colname_index].min():
                    curr_tf = tf.iloc[0]
                else:
                    curr_tf = tf.loc[tf[colname_index] <= f].iloc[-1]
            elif method == 'timestamp':  # find the closest world_timestamp
                timestamp = wts.loc[wts[colname_index] == f][colname_timestamp].values
                curr_tf = tf.iloc[np.argmin(np.abs(tf[colname_timestamp] - timestamp))]
            curr_gaze = gaze.loc[gaze[colname_index] == f]
            # output image tpe
            if output == 'wdo':  # world distorted overlay
                trans = curr_tf.surf_to_dist_img_trans
            elif output == 'sd':  # surface distorted
                trans = curr_tf.surf_to_dist_img_trans
            elif output == 'wuo':  # world undistorted overlay
                trans = curr_tf.surf_to_img_trans
            elif output == 'su':  # surface_undistorted
                trans = curr_tf.surf_to_img_trans
            # add gaze overlay
            if curr_gaze.size:
                p_norm = curr_gaze[['norm_pos_x', 'norm_pos_y']].values
                p = np.hstack((  p_norm[:, 0:1] * width, (1. - p_norm[:, 1:]) * height ))
                if (output == 'sd') or (output == 'su'):
                    p2 = transform_points_world_to_surface(p, trans)
                else:
                    p2 = p
                p_norm2 = np.hstack((p2[:,0:1] / width, 1. - (p2[:,1:] / height) ))
                gaze['norm_x_su'].loc[gaze[colname_index] == f] = p_norm2[:, 0]
                gaze['norm_y_su'].loc[gaze[colname_index] == f] = p_norm2[:, 1]
                gaze['x_su'].loc[gaze[colname_index] == f] = p2[:, 0]
                gaze['y_su'].loc[gaze[colname_index] == f] = p2[:, 1]
                gaze['screen_width'].loc[gaze[colname_index] == f] = width
                gaze['screen_height'].loc[gaze[colname_index] == f] = height
        print()


    return gaze