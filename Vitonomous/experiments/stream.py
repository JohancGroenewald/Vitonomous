import cv2
import numpy as np
import math
import datetime

verbose = 1
# ###################################################################
ip = '192.168.0.140'
port = 8000
user = 'admin'
password = '_Password1'
stream_type = 'videostream.cgi'
auth = '?user={}&password={}'.format(user, password)
url = 'http://{0}:{1}/{2}{3}'.format(ip, port, stream_type, auth)
wait_delay = 2
# ###################################################################
record = False
fps = 16
four_cc = cv2.VideoWriter.fourcc(*'DIVX')
in_color = True
# ###################################################################
color_white = (255, 255, 255)
color_blue = (255, 0, 0)
# ###################################################################
capture = cv2.VideoCapture(url)
stream_out = None
# ###################################################################
h, w, c = None, None, None
tx, ty, tw, th = None, None, None, None
c_tx, c_ty = None, None
GW, GH = None, None
L_GX, R_GX, C_GX = None, None, None
T_GY, B_GY, C_GY = None, None, None
# ###################################################################
horizontal = 0
horizontal_tracking = [0]
# ###################################################################
tracking_rectangle = None
gesture_rectangle = None
# ###################################################################
gesture = 0
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
    record_shape = (w, h)

    c_tx, c_ty = w//2, h//2
    tx, ty, tw, th = c_tx-1, c_ty-1, c_tx+1, c_ty+1
    tracking_rectangle = (c_tx, c_ty, tw, th)

    GW, GH = int(w*g_scale//2), int(h*g_scale//2)
    C_GX, C_GY = w//2, h//2
    L_GX, R_GX = C_GX - C_GX//2, C_GX + C_GX//2
    gesture_rectangle = (C_GX, C_GY)

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
    c_tx, c_ty = tx+tw//2, ty+th//2
    c_trx, c_try, trw, trh = tracking_rectangle
    c_gx, c_gy = gesture_rectangle
    # moving ########################################################
    velocity = 3
    # delta = tx - rtx
    # if delta < 0 and horizontal <= 0:
    #     delta = abs(delta)
    #     adjust = delta if delta < velocity else velocity
    #     gx -= adjust
    #     horizontal = -1
    #     if horizontal_tracking[-1] != horizontal:
    #         horizontal_tracking.append(horizontal)
    #         print(horizontal_tracking)
    # elif delta > 0 and horizontal >= 0:
    #     delta = abs(delta)
    #     adjust = delta if delta < velocity else velocity
    #     gx += adjust
    #     horizontal = +1
    #     if horizontal_tracking[-1] != horizontal:
    #         horizontal_tracking.append(horizontal)
    #         print(horizontal_tracking)
    # else:
    #     horizontal = 0
    #     if horizontal_tracking[-1] != horizontal:
    #         horizontal_tracking.append(horizontal)
    #         print(horizontal_tracking)
    #     if horizontal_tracking.count(horizontal) == 4:
    #         gesture = 0
    #         if horizontal_tracking == [0, 1, 0, -1, 0, 1, 0]:
    #             gesture = 1
    #         elif horizontal_tracking == [0, -1, 0, 1, 0, -1, 0]:
    #             gesture = 2
    #         horizontal_tracking = [horizontal]
    #         gx = GX
    #         # print(horizontal_tracking)
    #     elif horizontal_tracking.count(horizontal) > 4:
    #         gesture = 0
    #         horizontal_tracking = [horizontal]
    #         gx = GX
    # ###############################################################
    pass
    # ###############################################################
    tracking_rectangle = (c_tx, c_ty, tw, th)
    gesture_rectangle = (c_gx, c_gy)
    # ###############################################################
    if verbose:
        tx, ty = c_tx-tw//2, c_ty-th//2
        color = color_white
        thickness = 1
        cv2.rectangle(
            frame_out, (tx, ty), (tx + tw, ty + th), color, thickness
        )
    # ###############################################################
    if verbose:
        gx, gy, gw, gh = c_gx-GW, c_gy-GH, GW, GH
        color = color_blue
        thickness = 2
        cv2.rectangle(
            frame_out, (gx, gy), (gx + gw, gy + gh), color, thickness
        )
    # ###############################################################
    if verbose and gesture > 0:
        text = None
        if gesture == 1:
            text = 'SELECT >>'
        elif gesture == 2:
            text = 'SELECT <<'
        if text:
            text_size_wh, _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
            )
            tw, th = text_size_wh
            origin = (w//2-tw//2, h-th)
            cv2.putText(
                frame_out, text, origin,
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA
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
