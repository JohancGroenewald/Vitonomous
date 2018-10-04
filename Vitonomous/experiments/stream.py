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

hist_image = np.zeros((256, 256), dtype=np.uint8)
grabbed, re_frame = capture.read()
while grabbed:
    grabbed, frame_in = capture.read()
    if not grabbed:
        print('Error grabbing frame')
        break

    gray_frame_in = cv2.cvtColor(frame_in, cv2.COLOR_BGR2GRAY)
    gray_re_frame = cv2.cvtColor(re_frame, cv2.COLOR_BGR2GRAY)

    diff_frame = cv2.absdiff(gray_frame_in, gray_re_frame)

    # hist = cv2.calcHist([diff_frame], [0], None, [256], [0, 256])
    hist, bins = np.histogram(diff_frame.ravel(), 256, [0, 256])

    hist_image.fill(255)
    for i, h in enumerate(hist):
        # hist_image[i:i+1, 0:h] = 255
        hist_image[0:h, i:i+1] = 0
    hist_image = hist_image.transpose((0, 1))

    frame_out = diff_frame
    cv2.imshow("IP Camera", frame_out)
    cv2.imshow("Histogram", hist_image)
    if cv2.waitKey(delay=wait_delay) & 0xFF == ord('q'):
        break

    re_frame = frame_in

cv2.destroyAllWindows()
