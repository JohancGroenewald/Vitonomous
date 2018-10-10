import cv2
import imutils
import numpy as np
import math
import datetime

verbose = 1
# ###################################################################
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)
confidence = 0.35
faces = []
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
fps = 20
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
skip_frames = 1
skip_frames_counter = 0
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
    frame_out = frame_in.copy()
    face_frame = imutils.resize(frame_out, width=400)
    fh, fw = face_frame.shape[:2]
    # ###############################################################
    if skip_frames_counter >= skip_frames:
        skip_frames_counter = 0

        blob = cv2.dnn.blobFromImage(
            cv2.resize(face_frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()
        # ###############################################################
        new_faces = []
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            detection_confidence = detections[0, 0, i, 2]
            # print(confidence)
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if detection_confidence > confidence:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([fw, fh, fw, fh])
                (startX, startY, endX, endY) = box.astype("int")
                new_faces.append((startX, startY, endX, endY, detection_confidence))
                # (startX, startY, endX, endY) = box.astype("int")
                # draw the bounding box of the face along with the associated
                # probability
                # text = "{:.2f}%".format(confidence * 100)
                # y = startY - 10 if startY - 10 > 10 else startY + 10
                # cv2.rectangle(face_frame, (startX, startY), (endX, endY),
                #     (255, 0, 0), 1)
                # cv2.putText(face_frame, text, (startX, y),
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
        if len(new_faces) > 0:
            faces = new_faces
        # ###############################################################
    else:
        skip_frames_counter += 1
    # ###############################################################
    # ###############################################################
    for (startX, startY, endX, endY, _confidence) in faces:
        text = "{:.2f}%, {:.2f}%".format(
            _confidence * 100,
            ((endX-startX)*(endY-startY))/(fw*fh)*100
        )
        _y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(
            face_frame, (startX, startY), (endX, endY), (255, 0, 0), 1
        )
        cv2.putText(
            face_frame, text, (startX, _y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1
        )
        c_fx, c_fy = startX+((endX-startX)//2), endY
        h_w, h_h = 2, int((endY - startY)*0.1)
        h_x, h_y, h_xx, h_yy = c_fx-h_w, c_fy, c_fx+h_w, c_fy+h_h
        cv2.rectangle(
            face_frame, (h_x, h_y), (h_xx, h_yy), (255, 0, 0), 1
        )
        c_fx, c_fy = c_fx, h_yy
        h_w, h_h = int((endX - startX)*3), int((endY - startY)*1.5)
        h_x, h_y, h_xx, h_yy = c_fx-h_w, c_fy, c_fx+h_w, c_fy+h_h
        cv2.rectangle(
            face_frame, (h_x, h_y), (h_xx, h_yy), (255, 255, 0), 2
        )
        # ###############################################################
        diff_frame = imutils.resize(diff_frame, width=400)
        kernel = np.ones((5, 5), np.float32) / 25
        diff_frame = diff_frame[h_y:h_yy, h_x:h_xx]
        if diff_frame is not None and len(diff_frame) > 0:
            diff_frame = cv2.filter2D(diff_frame, -1, kernel)
            if diff_frame is not None and len(diff_frame) > 0:
                raw_indices = np.nonzero(diff_frame > 32)
                indices = [(x,y) for x,y in zip(raw_indices[1], raw_indices[0])]
                indices = [] if indices is None else indices
                if indices:
                    boundary = cv2.boundingRect(np.array(indices))
                    tx, ty, tw, th = boundary
                    tx, ty = tx+h_x, ty+h_y
                    thickness = 1
                    cv2.rectangle(
                        face_frame, (tx, ty), (tx + tw, ty + th), color_white, thickness
                    )
        # ###############################################################
    # ###############################################################
    frame_out = imutils.resize(face_frame, width=w, height=h)
    # ###############################################################
    # ###############################################################
    if record:
        stream_out.write(frame_out)
        cv2.circle(frame_out, (20, 20), 10, (0, 0, 255), 8)
    # ###############################################################
    cv2.imshow('IP Camera', frame_out)
    # ###############################################################
    # ###############################################################
    key = cv2.waitKey(delay=wait_delay) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') and record is False:
        uri = 'recording_{}.avi'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        stream_out = cv2.VideoWriter(uri, four_cc, fps, record_shape, in_color)
        record = True
    elif key == ord('r') and record is True:
        if stream_out.isOpened():
            stream_out.release()
        record = False
    # ###############################################################
    re_frame = frame_in
    # ###############################################################

cv2.destroyAllWindows()
