import cv2

ip = '192.168.0.140'
port = 8000
user = 'admin'
password = '_Password1'
stream_type = 'videostream.cgi'
auth = '?user={}&password={}'.format(user, password)
url = 'http://{0}:{1}/{2}{3}'.format(ip, port, stream_type, auth)
wait_delay = 2

while True:
    capture = cv2.VideoCapture(url)
    grabbed, frame_in = capture.read()

    if grabbed:
        gray_image = cv2.cvtColor(frame_in, cv2.COLOR_BGR2GRAY)
        frame_out = gray_image



        cv2.imshow("IP Camera", frame_out)
        if cv2.waitKey(delay=wait_delay) & 0xFF == ord('q'):
            break

    else:
        print('Error grabbing frame')
        break

cv2.destroyAllWindows()
