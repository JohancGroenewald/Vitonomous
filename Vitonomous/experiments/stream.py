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

re_boundary = None
grabbed, re_frame = capture.read()
while grabbed:
    grabbed, frame_in = capture.read()
    if not grabbed:
        print('Error grabbing frame')
        break

    frame_out = frame_in.copy()
    gray_frame_in = cv2.cvtColor(frame_in, cv2.COLOR_BGR2GRAY)
    gray_re_frame = cv2.cvtColor(re_frame, cv2.COLOR_BGR2GRAY)

    diff_frame = cv2.absdiff(gray_frame_in, gray_re_frame)

    frame_out = diff_frame

    diff_frame[diff_frame < 32] = 0
    diff_frame[diff_frame > 64] = 0
    raw_indices = np.nonzero(diff_frame > 0)
    indices = [(x,y) for x,y in zip(raw_indices[1], raw_indices[0])]
    indices = [] if indices is None else indices
    process = 200 < len(indices) < 2000
    if process:
        z = np.hstack((raw_indices[1], raw_indices[0]))
        # width, height = cv2.GetSize(gray_frame_in)
        z = z.reshape((len(z), 1))
        z = np.float32(z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_PP_CENTERS
        compactness, labels, centers = cv2.kmeans(z, 2, None, criteria, 10, flags)
        if np.sum(labels) > 0:
            # print(centers, labels, np.sum(labels))
            indices = [(x,y) for x,y in zip(centers[0], centers[1])]
            boundary = cv2.boundingRect(np.array(indices)) if indices else None
            if boundary:
                bx, by, bw, bh = boundary
                draw = True #15 < bw < 180 and 15 < bh < 180
                if draw:
                    cv2.rectangle(
                        frame_out, (bx, by), (bx+bw, by+bh),
                        (255, 255, 255), 1
                    )

    cv2.imshow("IP Camera", frame_out)
    if cv2.waitKey(delay=wait_delay) & 0xFF == ord('q'):
        break

    re_frame = frame_in

cv2.destroyAllWindows()
