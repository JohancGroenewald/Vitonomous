import cv2
import numpy as np
import math
import datetime

ip = '192.168.0.140'
port = 8000
user = 'admin'
password = '_Password1'
stream_type = 'videostream.cgi'
auth = '?user={}&password={}'.format(user, password)
url = 'http://{0}:{1}/{2}{3}'.format(ip, port, stream_type, auth)
wait_delay = 2

record = False
fps = 16
four_cc = cv2.VideoWriter.fourcc(*'DIVX')
in_color = True
# ###################################################################
verbose = 1
# ###################################################################
capture = cv2.VideoCapture(url)
stream_out = None
# ###################################################################
tx, ty, tw, th = None, None, None, None
GX, GY, GW, GH = None, None, None, None
# ###################################################################
horizontal = 0
horizontal_tracking = [0]
# ###################################################################
tracking_rectangle = None
gesture_rectangle = None
# ###################################################################
g_scale = 0.15
# ###################################################################
tracking_record = [(0, 0)]
signal_x_record = []
signal_y_record = []
# ###################################################################
# ###################################################################
# ###################################################################
grabbed, re_frame = capture.read()
if grabbed:
    h, w, c = re_frame.shape
    tx, ty, tw, th = 0, 0, w, h
    tracking_rectangle = (tx, ty, tw, th)
    GX, GY, GW, GH = int(w//2-w*g_scale//2), \
                     int(h//2-h*g_scale//2), \
                     int(w*g_scale), \
                     int(h*g_scale)
    gesture_rectangle = (GX, GY, GW, GH)
    record_shape = w, h

while grabbed:
    grabbed, frame_in = capture.read()
    if not grabbed:
        print('Error grabbing frame')
        break
    # ###############################################################
    gray_frame_in = cv2.cvtColor(frame_in, cv2.COLOR_BGR2GRAY)      #
    gray_re_frame = cv2.cvtColor(re_frame, cv2.COLOR_BGR2GRAY)      #
    # ###############################################################
    diff_frame = cv2.absdiff(gray_frame_in, gray_re_frame)
    # ###############################################################
    # frame_out = diff_frame
    frame_out = frame_in.copy()
    # ###############################################################
    kernel = np.ones((5, 5), np.float32) / 25
    diff_frame = cv2.filter2D(diff_frame, -1, kernel)
    raw_indices = np.nonzero(diff_frame > 32)
    indices = [(x,y) for x,y in zip(raw_indices[1], raw_indices[0])]
    indices = [] if indices is None else indices
    process = 200 < len(indices) < 2000
    # ###############################################################
    if process:
        boundary = cv2.boundingRect(np.array(indices))
        tx, ty, tw, th = boundary
    # ###############################################################
    color_white = (255, 255, 255)
    thickness = 1
    rtx, rty, rtw, rth = tracking_rectangle
    c_rtx, c_rty = rtx+rtw//2, rty+rth//2
    gx, gy, gw, gh = gesture_rectangle
    c_gx, c_gy = gx+gw//2, gy+gh//2
    # moving ########################################################
    velocity = 3
    delta = rtx+rtw - c_gx
    if delta < 0 and horizontal <= 0:
        delta = abs(delta)
        adjust = delta if delta < velocity else velocity
        gx -= adjust
        horizontal = -1
        if horizontal_tracking[-1] != horizontal:
            horizontal_tracking.append(horizontal)
            print(horizontal_tracking)
    elif delta > 0 and horizontal >= 0:
        delta = abs(delta)
        adjust = delta if delta < velocity else velocity
        gx += adjust
        horizontal = +1
        if horizontal_tracking[-1] != horizontal:
            horizontal_tracking.append(horizontal)
            print(horizontal_tracking)
    else:
        horizontal = 0
        if horizontal_tracking[-1] != horizontal:
            horizontal_tracking.append(horizontal)
            print(horizontal_tracking)
        if horizontal_tracking.count(horizontal) > 3:
            if horizontal_tracking == [0, 1, 0, -1, 0, 1, 0]:
                print('SELECT >>')
            elif horizontal_tracking == [0, -1, 0, 1, 0, -1, 0]:
                print('SELECT <<')
            horizontal_tracking = [horizontal]
            print(horizontal_tracking)

    # ###############################################################
    gesture_rectangle = (gx, GY, GW, GH)
    # ###############################################################
    tracking_rectangle = (tx, ty, tw, th)
    if verbose:
        color = color_white
        cv2.rectangle(
            frame_out, (tx, ty), (tx + tw, ty + th), color, thickness
        )
    # ###############################################################
    color_blue = (255, 0, 0)
    gx, gy, gw, gh = gesture_rectangle
    if verbose:
        color = color_blue
        thickness = 2
        cv2.rectangle(
            frame_out, (gx, gy), (gx + gw, gy + gh), color, thickness
        )
    # ###############################################################
    cv2.imshow('IP Camera', frame_out)
    # ###############################################################
    if record:
        stream_out.write(frame_out)
    # ###############################################################
    key = cv2.waitKey(delay=wait_delay) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        print(tracking_record)
        tracking_record = [(0, 0)]
    elif key == ord('r') and record is False:
        uri = 'recording_{}.avi'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        stream_out = cv2.VideoWriter(uri, four_cc, fps, record_shape, in_color)
        record = True
    elif key == ord('r') and record is True:
        record = False
    # ###############################################################
    re_frame = frame_in
    # ###############################################################

cv2.destroyAllWindows()
