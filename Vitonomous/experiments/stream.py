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
attached = False
bw = 100
# ###################################################################
grabbed, re_frame = capture.read()
if grabbed:
    h, w, c = re_frame.shape
    gesture_rectangle = [w//2-bw//2, h//2-bw//2, bw, bw]
    bx, by, bw, bh = gesture_rectangle
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
        # ###########################################################
        # if not attached:
        #     bxc = bx + bw//2
        #     byc = by + bh//2
        #     gx, gy, gw, gh = gesture_rectangle
        #     gxc = gx + gw//2
        #     gyc = gy + gh//2
        #     gxc += (bxc - gxc)
        #     gyc += (byc - gyc)
        #     gesture_rectangle[0] = gxc
        #     gesture_rectangle[1] = gyc
        # ###########################################################
        cv2.rectangle(
            frame_out, (bx, by), (bx+bw, by+bh),
            (255, 255, 255), 1
        )
        # ###########################################################
    # gx, gy, gw, gh = gesture_rectangle
    cv2.rectangle(
        frame_out, (bx, by), (bx+bw, by+bh),
        (255, 255, 255), 1
    )
    gesture_rectangle = (bx, by, bw, bh)
    # ###############################################################
    cv2.imshow("IP Camera", frame_out)
    if cv2.waitKey(delay=wait_delay) & 0xFF == ord('q'):
        break
    # ###############################################################
    re_frame = frame_in
    # ###############################################################

cv2.destroyAllWindows()
