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
record_shape = None
# ###################################################################
grabbed, re_frame = capture.read()
if grabbed:
    h, w, c = re_frame.shape
    record_shape = (w, h)
# ###################################################################
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
    blank_frame = np.zeros((h, w, 3), np.uint8)
    # ###############################################################
    # frame_out = diff_frame
    # frame_out = frame_in.copy()
    # ###############################################################

    # ###############################################################
    ret, thresh = cv2.threshold(diff_frame, 32, 255, 0)
    _, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(blank_frame, contours, -1, (0, 255, 0), 1)
    # ###############################################################
    c_img = cv2.cvtColor(blank_frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        c_img, cv2.HOUGH_GRADIENT, 1.2, 100,
        # param1=50, param2=30, minRadius=0, maxRadius=0
    )
    if circles is not None and len(circles) > 0:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(blank_frame,(i[0],i[1]),i[2],(0,255,0),2)
            cv2.circle(blank_frame,(i[0],i[1]),2,(0,0,255),3)
    # ###############################################################
    frame_out = blank_frame
    # ###############################################################

    # ###############################################################

    # ###############################################################

    # ###############################################################

    # ###############################################################
    cv2.imshow('IP Camera', frame_out)
    # ###############################################################
    if record:
        stream_out.write(frame_out)
    # ###############################################################
    key = cv2.waitKey(delay=wait_delay) & 0xFF
    if key == ord('q'):
        break
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
