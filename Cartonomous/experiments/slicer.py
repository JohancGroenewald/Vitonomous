import os
import cv2
import PIL
import imutils
from datetime import datetime

from sources import Sources
import image_slicer as image_slicer

video_source = os.path.join(Sources.ROOT, Sources.SOURCE, Sources.VIDEO)
files = os.listdir(video_source)
print(files)
url = os.path.join(Sources.ROOT, Sources.SOURCE, Sources.VIDEO, files[3])

wait_delay = 2
capture = cv2.VideoCapture(url)

frame_index = -1
session_directory = datetime.now().strftime('%Y%m%d_%H%M%S')
session_url = None

grabbed, re_frame = capture.read()
while grabbed:
    frame_index += 1
    grabbed, frame_in = capture.read()
    if not grabbed:
        print('End of frame stream')
        break
    # ###############################################################
    # ###############################################################
    gray_frame_in = cv2.cvtColor(frame_in, cv2.COLOR_BGR2GRAY)      #
    gray_re_frame = cv2.cvtColor(re_frame, cv2.COLOR_BGR2GRAY)      #
    # ###############################################################
    diff_frame = cv2.absdiff(gray_frame_in, gray_re_frame)
    # ###############################################################
    # frame_out = diff_frame
    gray_re_frame = imutils.resize(gray_re_frame, 320, 240)
    frame_out = gray_re_frame.copy()
    # ###############################################################
    # frame_out <-- blur
    # ###############################################################
    cv2.imshow('Camera Stream', frame_out)
    # ###############################################################
    # ###############################################################
    key = cv2.waitKey(delay=wait_delay) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):

        if session_url is None:
             session_url = os.path.join(Sources.ROOT, Sources.SOURCE, Sources.TILES, session_directory)
             os.mkdir(session_url)
        # # image = PIL.Image.fromarray(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB))
        # image = PIL.Image.fromarray(frame_out)
        save_url = os.path.join(
             session_url, '{}.png'.format(frame_index)
        )
        cv2.imwrite(save_url, frame_in)
        #
        # image_slicer.slice_image(image, 40, save_url)
    # ###############################################################
    re_frame = frame_in
    # ###############################################################
