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
# ###################################################################
re_boundary = None
gesture_rectangle = None
bx, by, bw, bh = None, None, None, None
# ###################################################################
gesture_record = [(0, 0)]
signal_x_record = []
signal_y_record = []
# ###################################################################
stable_counter = 0
wider_counter = 0
# ###################################################################
stable_triggered = False
gesture_triggered = False
gesture_processed = False
gesture_detected = 0
# ###################################################################
grabbed, re_frame = capture.read()
if grabbed:
    h, w, c = re_frame.shape
    bx, by, bw, bh = 0, 0, w, h
    gesture_rectangle = (bx, by, bw, bh)
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
        bx, by, bw, bh = boundary
    # ###############################################################
    color = (255, 255, 255)
    thickness = 1
    grx, gry, grw, grh = gesture_rectangle
    # stable ########################################################
    stable = grx-bx == gry-by == grw-bw == grh-bh == 0
    if stable and not stable_triggered:
        stable_counter += 1
        if stable_counter > 10:
            stable_triggered = True
    elif stable and stable_triggered:
        color = (255, 0, 0)
        thickness = 2
    elif not stable:
        stable_counter = 0
        stable_triggered = False
        gesture_triggered = False
        gesture_processed = False
    # translating ###################################################
    if not stable_triggered:
        if len(gesture_record) < 100:
            adj_x = 0
            adj_y = 0
            cen_x, cen_y = grx+grw//2, gry+grh//2
            if len(gesture_record) > 1:
                re_cx, re_cy = gesture_record[-1]
                # print(re_cx, re_cy, cen_x, cen_y)
                adj_x = (cen_x - re_cx) // 3
                adj_y = (cen_y - re_cy) // 3
                center = (adj_x+cen_x, adj_y+cen_y)
                if gesture_record[-1] != center:
                    gesture_record.append(center)
            center = (cen_x, cen_y)
            if gesture_record[-1] != center:
                gesture_record.append(center)
        offset = 2
        tx_max = h - 65
        tx = h - 30
        ty_max = h - 65
        ty = h - 30
        tx_high_count = 0
        tx_low_count = 0
        ty_high_count = 0
        ty_low_count = 0
        t_pivot = 5
        for i, center in enumerate(gesture_record[1:]):
            # cv2.circle(frame_out, center, 5, (255, 0, 0))
            cx, cy = center
            if verbose:
                cv2.line(frame_out, (offset*i, cx), (offset*i, h), (255, 255, 255), 1)
                cv2.line(frame_out, (w//2 + offset*i, cy), (w//2 + offset*i, h), (255, 255, 255), 1)
            if i > 0:
                rcx, rcy = re_center
                # if cx < rcx and rcx - cx > 7:
                #     tx_high_count += 1
                # elif cx > rcx and cx - rcx > 7:
                #     tx_low_count += 1
                # if cy < rcy and rcy - cy > 7:
                #     ty_high_count += 1
                # elif cy > rcy and cy - rcy > 7:
                #     ty_low_count += 1
                if cx < rcx:
                    tx_high_count += 1
                elif cx > rcx:
                    tx_low_count += 1
                if cy < rcy:
                    ty_high_count += 1
                elif cy > rcy:
                    ty_low_count += 1
                if tx_high_count > t_pivot:
                    tx = tx_max
                    tx_high_count = 0
                    tx_low_count = 0
                elif tx_low_count > t_pivot:
                    tx = h - 5
                    tx_high_count = 0
                    tx_low_count = 0
                if ty_high_count > t_pivot:
                    ty = ty_max
                    ty_high_count = 0
                    ty_low_count = 0
                elif ty_low_count > t_pivot:
                    ty = h - 5
                    ty_high_count = 0
                    ty_low_count = 0
                if len(signal_x_record) == 0 or tx != signal_x_record[-1]:
                    signal_x_record.append(tx)
                if len(signal_y_record) == 0 or ty != signal_y_record[-1]:
                    signal_y_record.append(ty)
            if verbose:
                cv2.line(frame_out, (offset*i, tx), (offset*i, h), (255, 0, 0), 1)
                cv2.line(frame_out, (w//2 + offset*i, ty), (w//2 + offset*i, h), (255, 0, 0), 1)
            re_center = center

    elif stable_triggered and not gesture_triggered and len(gesture_record) > 1:
        start, stop = gesture_record[1], gesture_record[-1]
        gesture_record = [(0, 0)]
        gesture_triggered = True
    elif stable_triggered and gesture_triggered and not gesture_processed:
        print('x', signal_x_record[:4])
        print('y', signal_y_record[:4])
        gesture_detected = 0
        if len(signal_x_record) > 1 and len(signal_y_record) > 1:
            if signal_x_record[:4] == [450, 415, 450, 415] and signal_y_record[:4] == [450, 415, 450, 415]:
                gesture_detected = 1
            elif signal_x_record[:4] == [450, 475, 450, 475] and signal_y_record[:4] == [450, 415, 450, 415]:
                gesture_detected = 2
            elif signal_x_record[:4] == [450, 475, 450, 475] and signal_y_record[:4] == [450, 475, 450, 475]:
                gesture_detected = 3
            elif signal_x_record[:4] == [450, 415, 450, 415] and signal_y_record[:4] == [450, 475, 450, 475]:
                gesture_detected = 4
        signal_x_record = []
        signal_y_record = []
        gesture_processed = True
    elif stable_triggered and gesture_triggered and gesture_processed:
        gesture_text = None
        if gesture_detected == 1:
            gesture_text = 'Volume UP'
        elif gesture_detected == 2:
            gesture_text = 'Volume DOWN'
        elif gesture_detected == 3:
            gesture_text = 'Move NEXT'
        elif gesture_detected == 4:
            gesture_text = 'Move PREVIOUS'
        if gesture_text:
            cv2.putText(
                frame_out, gesture_text, (0, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA
            )
    # shrinking #####################################################
    # moving ########################################################
    gesture_rectangle = (bx, by, bw, bh)
    if verbose:
        cv2.rectangle(
            frame_out, (bx, by), (bx+bw, by+bh), color, thickness
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
        print(gesture_record)
        gesture_record = [(0, 0)]
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
