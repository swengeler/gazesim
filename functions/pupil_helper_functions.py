import sys
import os
import cv2
import csv
import numpy as np
import pandas as pd


#simple video to image
def video_to_image(vidcap, frame):

    #important: every vidcap.read() call raises the frame counter
    #frame counter starts from 1 (not zero) 
    #frame is relative to the .mp4 file, not the same as in gaze_positions.csv
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame) 
    success, image = vidcap.read()

    return image

#get a dictionairy of properties for video and current frame
def video_properties(vidcap, frame=None):

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
        prop[ prop_names[i].lower() ] = vidcap.get(i)

    return prop

#undistort world image
def undistort_world_image(image, k, d):
#image: distorted world image
#k: camera_matrix
#d: distortion coefficients
#image_und: undistorted world image
    image_und = cv2.undistort(
    image,
    k,
    d,
    None,
    k)

    return image_und

#undistort points
def undistort_points(p, k, d):
#p: distorted points, ndarray of 2d points, n x 2
#k: camera_matrix
#d: distortion coefficients
#p_und: undistorted points, ndarray of 2d points, n x 2
    p_und = cv2.undistortPoints(
        p.reshape(-1,1,2).astype(np.float32), 
        k, 
        d, 
        P=k).reshape(-1,2)

    return p_und

#make the transform from world to surface
def make_transform_W_S(t, width=800, height=600):
#t: transform surface to world

    #surface borders in surface coordinates (S, 0-1 range)
    surfborders_S = np.array(
                  ((0,0),(1,0),(1,1),(0,1)),dtype=np.float32)

    #transform surface borders to world coordinates (W, 800x600 pixels)
    surfborders_W = cv2.perspectiveTransform(
                         surfborders_S.reshape(-1,1,2),
                         t).reshape(-1,2)

    #world borders in world coordinates
    worldborders_W = np.array(((0,height-1),(width-1,height-1),(width-1,0),(0,0)),dtype=np.float32)

    #transform from world to surface frame in World coordinates
    tf_W_S = cv2.getPerspectiveTransform(surfborders_W, worldborders_W)

    return tf_W_S

#transform points
def transform_points_world_to_surface(p, t, width=800, height=600):
#p: points in world frame, ndarray of 2d points, n x 2
#t: transform world to surface

    #surface borders in surface coordinates (S, 0-1 range)
    surfborders_S = np.array(
                  ((0,0),(1,0),(1,1),(0,1)),dtype=np.float32)

    #transform surface borders to world coordinates (W, 800x600 pixels)
    surfborders_W = cv2.perspectiveTransform(
                         surfborders_S.reshape(-1,1,2),
                         t).reshape(-1,2)

    #world borders in world coordinates
    worldborders_W = np.array(((0,height-1),(width-1,height-1),(width-1,0),(0,0)),dtype=np.float32)

    #transform from world to surface frame in World coordinates
    tf_W_S = cv2.getPerspectiveTransform(surfborders_W, worldborders_W)

    #transform surface borders to world coordinates (W, e.g. 800 x 600 pixels)
    p_trans = cv2.perspectiveTransform(
        p.reshape(-1,1,2).astype(np.float32),
        tf_W_S).reshape(-1,2)

    return p_trans

#image and transform to surface image frame
def world_image_to_surface_image(image, transform):

    #surface borders in surface coordinates (S, 0-1 range)
    surfborders_S = np.array(
                  ((0,0),(1,0),(1,1),(0,1)),dtype=np.float32)

    #transform surface borders to world coordinates (W, 800x600 pixels)
    surfborders_W = cv2.perspectiveTransform(
                         surfborders_S.reshape(-1,1,2),
                         transform).reshape(-1,2)
    
    #get the image width and height
    height, width, _ = image.shape
    
    #world borders in world coordinates
    worldborders_W = np.array(((0,height-1),(width-1,height-1),(width-1,0),(0,0)),dtype=np.float32)

    #transform from world to surface frame in World coordinates
    tf_W_S = cv2.getPerspectiveTransform(surfborders_W, worldborders_W)

    #apply transform
    image_surface = cv2.warpPerspective(image, tf_W_S, (width, height))

    return image_surface

#video gaze overlay
def image_points_overlay(image, points):

    image_modified=image.copy()

    if len(points)>0:
      
        for x_, y_ in points:

            cv2.circle(image_modified, (int(x_), int(y_)), 5, (255,0,0), -1)

    return image_modified

#video gaze overlay
def video_gaze_overlay(vidcap, gaze, outfilepathname=None, distorted=False):

    #default filepath and name for output video
    if outfilepathname is None:
        outfilepath = os.getcwd()+'/'
        outfilepathname = outfilepath + '/noname.mp4'
    else:
        outfilepath = '/'.join(outfilepathname.split('/')[:-1])+'/'

    #make the output path
    make_outpath(outfilepath)

    #read the first video frame
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, image = vidcap.read()

    #get some important information about the video
    width = vidcap.get(3)      #image width
    height = vidcap.get(4)     #image height
    fps = vidcap.get(5)        #frames per second
    fourcc = vidcap.get(6)     #fourcc codec integer identifier
    num_frames = vidcap.get(7) #number of frames
    is_color = True            #whether to use color

    if distorted:
        df = gaze[['gaze_timestamp','world_index','norm_pos_x','norm_pos_y']]
        df.columns = ['ts','frame','x','y']
        df['x'] = df['x'] * width
        df['y'] = (1.0-df['y']) * height
    else:
        df = gaze[['GazeTimeStamp','MediaFrameIndex','Gaze2dX','Gaze2dY']]
        df.columns = ['ts','frame','x','y']

    #video writer objecgt
    vidwriter = cv2.VideoWriter(
        outfilepathname,
        int(fourcc), 
        fps, 
        (int(width),int(height)),
        is_color)

    #loop over video frames
    for frame_vid in range(int(2000)):
    # for frame_vid in range(int(num_frames)):

        #read current frame
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_vid)
        success, image = vidcap.read()

        #find gaze positions for this frame
        x = df[df['frame']==frame_vid]['x'].values
        y = df[df['frame']==frame_vid]['y'].values

        image_modified=image.copy()

        print('frame: {}/{}, tf: {}'.format(frame_vid,num_frames,True))


        if len(x)>0:
            #plot overlay
            #copy the raw image and draw an overlay
            
            for x_,y_ in zip(x,y):
                # print(x_)
                # print(y_)

                cv2.circle(image_modified, (int(x_), int(y_)), 5, (255,0,0), -1)
            


        #write frame to output video
        vidwriter.write(image_modified)

    #close the video writer
    vidwriter.release()

#modifies world camera video based on surface transforms, either adds overlay of surface, or changes video to become surface image
def world_video_to_surface_video(vidcap, transform, outfilepathname=None, distorted=False, to_surface=False, method='frame', frame_range = None):

    #default filepath and name for output video
    if outfilepathname is None:
        outfilepath = os.getcwd()+'/'
        outfilepathname = outfilepath + '/noname.mp4'
    else:
        outfilepath = '/'.join(outfilepathname.split('/')[:-1])+'/'

    #make the output path
    make_outpath(outfilepath)

    #name of the transform to use
    if distorted:
        tf_name = 'surf_to_dist_img_trans'
    else:
        tf_name = 'surf_to_img_trans'

    #undistorted video
    df = transform[['world_index','world_timestamp',tf_name]]
    df.columns = ['frame','ts','trans']

    #read the first video frame
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, image = vidcap.read()

    #get some important information about the video
    t0_vid_ms = vidcap.get(0)  #time of video start
    width = vidcap.get(3)      #image width
    height = vidcap.get(4)     #image height
    fps = vidcap.get(5)        #frames per second
    fourcc = vidcap.get(6)     #fourcc codec integer identifier
    num_frames = vidcap.get(7) #number of frames
    is_color = True            #whether to use color
    
    #define acceptable window in ms where to search for surface frames
    win_ms = (1000.0 / fps) / 2.0  #half a period
    
    #time of surface start
    t0_surf_sec = df[df['frame']==0]['ts'].values[0]

    #add exptime
    df['exptime_ms'] = ( df['ts'] - t0_surf_sec ) * 1000

    #video writer objecgt
    vidwriter = cv2.VideoWriter(
        outfilepathname,
        int(fourcc), 
        fps, 
        (int(width),int(height)),
        is_color)

    #if no frame range was specified
    if frame_range is None:
        frame_range = [0,int(num_frames)]

    #loop over video frames
    for frame_vid in range(frame_range[0],frame_range[1]):

        #read current frame
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_vid)
        success, image = vidcap.read()

        #time of current frame
        t_vid_ms = vidcap.get(0)
        
        #time delta from start of video
        exptime_vid_ms = t_vid_ms - t0_vid_ms
        
        #find the closest frames based on time
        ind = (
            (df['exptime_ms'] > (exptime_vid_ms - win_ms)) &  
            (df['exptime_ms'] < (exptime_vid_ms + win_ms)))

        #if surface was found in the search window..
        if np.sum(ind)>0:

            print('frame: {}/{}, tf: {}, method: {}'.format(frame_vid,frame_range[1],True, method))
            
            #get the frame of the surface
            #if time-based method..
            if method == 'time':
                frame_surf = df[ind]['frame'].values[0] 
            #otherwise use frame-based method..
            else:
                frame_surf = frame_vid

            #get transform
            tf_S_W = df[ df['frame'] == frame_surf ]['trans'].values[0]

            #modify frame:
            #either to surface frame..
            if to_surface:
                image_modified = world_image_to_surface_image(image, tf_S_W)
            #or to world frame with surface overlay..
            else:
                image_modified = world_image_to_world_image_with_surfborder(image, tf_S_W, frame_vid)

        #if no suface was found in search window..
        else:
            print('frame: {}/{}, tf: {}, method: {}'.format(frame_vid,frame_range[1],False,method))

            #modified image is the original image
            image_modified = image.copy()

        #write frame to output video
        vidwriter.write(image_modified)

    #close the video writer
    vidwriter.release()

#make filepath
def make_outpath(path):

    outpath = '/'
    folders = path.split('/')

    for fold in folders:
        if len(fold)>0:
            outpath += fold +'/'
            if os.path.isdir(outpath) == False:
                os.mkdir(outpath)

#surface transform from sting to (3,3) numpy array
def transform_string_to_array(s):
    
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

#opencv video capture object and surface transform to overlay image frame
def world_video_to_world_image_with_surfborder(vidcap, transform, frame, col = (256,0,0), lw = 10):

    #load video frame
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    success, image = vidcap.read()

    #convert to RGB color
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #surface borders in surface coordinates (S, 0-1 range)
    surfborders_S = np.array(
                  ((0,0),(1,0),(1,1),(0,1)),dtype=np.float32)

    #transform surface borders to world coordinates (W, e.g. 800 x 600 pixels)
    surfborders_W = cv2.perspectiveTransform(
                         surfborders_S.reshape(-1,1,2),
                         transform).reshape(-1,2)

    #copy the raw image and draw an overlay
    image_overlay=image.copy()
    cv2.line(image_overlay, tuple(surfborders_W[0]), tuple(surfborders_W[1]), col, lw)
    cv2.line(image_overlay, tuple(surfborders_W[1]), tuple(surfborders_W[2]), col, lw)
    cv2.line(image_overlay, tuple(surfborders_W[2]), tuple(surfborders_W[3]), col, lw)
    cv2.line(image_overlay, tuple(surfborders_W[3]), tuple(surfborders_W[0]), col, lw)
    
    #return image frame and image frame with border overlay
    return image, image_overlay

#world image and surface transform to overlay image frame
def world_image_to_world_image_with_surfborder(image, transform, frame, col = (256,0,0), lw = 10):

    #surface borders in surface coordinates (S, 0-1 range)
    surfborders_S = np.array(
                  ((0,0),(1,0),(1,1),(0,1)),dtype=np.float32)

    #transform surface borders to world coordinates (W, e.g. 800 x 600 pixels)
    surfborders_W = cv2.perspectiveTransform(
                         surfborders_S.reshape(-1,1,2),
                         transform).reshape(-1,2)

    #copy the raw image and draw an overlay
    image_overlay=image.copy()
    cv2.line(image_overlay, tuple(surfborders_W[0]), tuple(surfborders_W[1]), col, lw)
    cv2.line(image_overlay, tuple(surfborders_W[1]), tuple(surfborders_W[2]), col, lw)
    cv2.line(image_overlay, tuple(surfborders_W[2]), tuple(surfborders_W[3]), col, lw)
    cv2.line(image_overlay, tuple(surfborders_W[3]), tuple(surfborders_W[0]), col, lw)
    
    #return image frame and image frame with border overlay
    return image_overlay

#opencv video capture object and transform to surface image frame
def world_video_to_surface_image(vidcap, transform, frame):

    #load video frame
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    success, image = vidcap.read()

    #convert to RGB color
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #surface borders in surface coordinates (S, 0-1 range)
    surfborders_S = np.array(
                  ((0,0),(1,0),(1,1),(0,1)),dtype=np.float32)

    #transform surface borders to world coordinates (W, 800x600 pixels)
    surfborders_W = cv2.perspectiveTransform(
                         surfborders_S.reshape(-1,1,2),
                         transform).reshape(-1,2)
    
    #get the image width and height
    height, width, _ = image.shape
    
    #world borders in world coordinates
    worldborders_W = np.array(((0,height-1),(width-1,height-1),(width-1,0),(0,0)),dtype=np.float32)

    #transform from world to surface frame in World coordinates
    tf_W_S = cv2.getPerspectiveTransform(surfborders_W, worldborders_W)

    #apply transform
    image2 = cv2.warpPerspective(image, tf_W_S, (width, height))

    return image, image2



