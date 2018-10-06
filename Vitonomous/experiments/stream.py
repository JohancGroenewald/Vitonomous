import cv2
import numpy as np

ip = '192.168.0.140'
port = 8000
user = 'admin'
password = '_Password1'
stream_type = 'videostream.cgi'
auth = '?user={}&password={}'.format(user, password)
url = 'http://{0}:{1}/{2}{3}'.format(ip, port, stream_type, auth)
wait_delay = 2

capture = cv2.VideoCapture(url)
# ###################################################################
re_boundary = None
gesture_rectangle = None
bx, by, bw, bh = None, None, None, None
# ###################################################################
stable_counter = 0
wider_counter = 0
# ###################################################################
grabbed, re_frame = capture.read()
if grabbed:
    h, w, c = re_frame.shape
    bx, by, bw, bh = 0, 0, w, h
    gesture_rectangle = (bx, by, bw, bh)
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
        # cv2.rectangle(
        #     frame_out, (bx, by), (bx+bw, by+bh),
        #     (255, 255, 255), 1
        # )
        # ###########################################################
    color = (255, 255, 255)
    grx, gry, grw, grh = gesture_rectangle
    # stable ########################################################
    if grx-bx == gry-by == grw-bw == grh-bh == 0:
        stable_counter += 1
        if stable_counter > 10:
            color = (255, 0, 0)
            wider_counter = 0
    else:
        stable_counter = 0
    # growing #######################################################
    if stable_counter == 0:
        if bw - grw > 5:
            wider_counter += 1
        elif bw - grw < -5:
            wider_counter -= 1
        if wider_counter > 2:
            color = (0, 255, 0)
        elif wider_counter < -2:
            color = (0, 0, 255)

    # shrinking #####################################################
    # moving ########################################################
    gesture_rectangle = (bx, by, bw, bh)

    cv2.rectangle(
        frame_out, (bx, by), (bx+bw, by+bh), color, 1
    )
    # ###############################################################
    cv2.imshow("IP Camera", frame_out)
    if cv2.waitKey(delay=wait_delay) & 0xFF == ord('q'):
        break
    # ###############################################################
    re_frame = frame_in
    # ###############################################################

cv2.destroyAllWindows()
