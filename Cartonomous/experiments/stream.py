import os
import cv2
import PIL

from sources import Sources
import image_slicer.image_slicer as image_slicer

video_source = os.path.join(Sources.ROOT, Sources.SOURCE, Sources.VIDEO)
files = os.listdir(video_source)
print(files)

url = os.path.join(Sources.ROOT, Sources.SOURCE, Sources.VIDEO, files[3])

wait_delay = 2
capture = cv2.VideoCapture(url)

grabbed, re_frame = capture.read()
while grabbed:
    grabbed, frame_in = capture.read()
    if not grabbed:
        print('End of frame stream')
        break
    # ###############################################################
    gray_frame_in = cv2.cvtColor(frame_in, cv2.COLOR_BGR2GRAY)      #
    gray_re_frame = cv2.cvtColor(re_frame, cv2.COLOR_BGR2GRAY)      #
    # ###############################################################
    # diff_frame = cv2.absdiff(gray_frame_in, gray_re_frame)
    # ###############################################################
    # frame_out = diff_frame
    frame_out = frame_in.copy()
    # ###############################################################

    # ###############################################################
    cv2.imshow('Camera Stream', frame_out)
    # ###############################################################
    # ###############################################################
    key = cv2.waitKey(delay=wait_delay) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        image = PIL.Image.fromarray(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB))
        save_url = os.path.join(Sources.ROOT, Sources.SOURCE, Sources.TILES, files[3])
        image_slicer.slice_image(image, 50, save_url)
