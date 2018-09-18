import cv2
import numpy as np
import os
import pickle

# from matplotlib import pyplot as plt


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 200,
                       qualityLevel = 0.1,
                       minDistance = 30)#,
                       # blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def compute_trajectory(filename):
    if 'mp4' in filename:
        pkl = filename.replace('mp4', 'pkl')
    else:
        pkl = filename.replace('avi', 'pkl')

    if os.path.exists(pkl):
        transform = pickle.load(open(pkl, 'rb'))
        return transform

    cap = cv2.VideoCapture(filename)
    ret, frame_prev = cap.read()
    frame_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    point_prev = cv2.goodFeaturesToTrack(frame_prev, mask = None, **feature_params)
    transform = []

    while True:
        ret,frame = cap.read()
        if not ret:
            break
        frame_cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        point_cur, status, err = cv2.calcOpticalFlowPyrLK(frame_prev, frame_cur, point_prev, None, **lk_params)
        good_cur = point_cur[status==1]
        good_prev = point_prev[status==1]
        trans = cv2.estimateRigidTransform(good_prev, good_cur, False)
        if trans is None:
            transform.append(transform[-1])
        else:
            transform.append(trans)
        frame_prev = frame_cur
        point_prev = cv2.goodFeaturesToTrack(frame_prev, mask = None, **feature_params)
    cap.release()
    transform = np.array(transform)
    pickle.dump(transform, open(pkl, 'wb'))
    return transform

# def plot(y, name):
#     x = np.arange(y.shape[0])
#     fig = plt.figure(figsize=(64,32))
#     plt.plot(x, y)
#     plt.savefig(name+'.jpg')

def smooth(transform, r=30):
    x = transform[:,0,2]
    y = transform[:,1,2]
    a = np.arctan2(transform[:,1,0], transform[:,0,0])

    da = smooth_single(a, r)
    dx = smooth_single(x, r)
    dy = smooth_single(y, r)

    smoothed = np.zeros_like(transform)
    smoothed[:,0,0] = np.cos(da)
    smoothed[:,0,1] = -np.sin(da)
    smoothed[:,1,0] = np.sin(da)
    smoothed[:,1,1] = np.cos(da)
    smoothed[:,0,2] = dx
    smoothed[:,1,2] = dy

    return smoothed

def smooth_single(dx, r=30):
    '''
    Args:
        dx: diff between consecutive frames
        r: moving average size
    Return:
        d: smoothed dx

    x: original trajectory
    avg: smoothed trajectory
    '''
    x = np.cumsum(dx)
    x_pad = np.pad(x, (r, r), mode='edge')
    avg = np.convolve(x_pad, np.ones((2*r+1,))/(2*r+1), mode='valid')
    d = dx + avg - x
    return d

def show_and_write(filename, T, outname='out'):
    outname += '.mp4'
    cap = cv2.VideoCapture(filename)
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    out = cv2.VideoWriter(outname, 0x00000021, 30.0, (w,h))
    for i in range(T.shape[0]-1):
        if not ret:
            break
        frame_cur = cv2.warpAffine(frame, T[i], (w, h))
        out.write(frame_cur)
        canvas = np.zeros((h, 2*w+10, 3))
        canvas[:, :w,...] = frame
        canvas[:, w+10:, ...] = frame_cur
        canvas = cv2.resize(canvas, ((2*w+10)//2, h//2))
        cv2.imshow('compare', canvas.astype(np.uint8))
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        ret, frame = cap.read()
    cap.release()
    out.release()

def main(filename):    
    out_name = filename.split('.')[0]+'_stabled'
    transform = compute_trajectory(filename)
    smoothed = smooth(transform, 30)
    show_and_write(filename, smoothed, out_name)


if __name__ == '__main__':
    main('hippo.mp4')