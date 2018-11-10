import glob
import os

from sources import video_source
from engines import WindowStream, VideoStream, RectangleStream
from states import StateManager

available_videos = [
    (100, False,     '20181006_175023.mp4'),  # 0  <= Short but usable
    (100, False,     '20181006_175105.mp4'),  # 1  <= Goeie data (Golfbaan) (Kort)
    (100, False,     '20181006_175204.mp4'),  # 2  <= Goeie data (Golfbaan) (Lank)
    (100, False,     '20181006_175546.mp4'),  # 3  <= Goeie data (Golfbaan) (Kort) (Hill parking)
    (100, False,     '20181006_175653.mp4'),  # 4  <= Goeie data (Golfbaan) (Lank)
    (100, False,     '20181006_175934.mp4'),  # 5  <= Goeie data (Golfbaan) (Medium) (Grond pad)
    (100, False,     '20181006_180356.mp4'),  # 6  <= Goeie data (Golfbaan)
    (100, False,     '20181006_180654.mp4'),  # 7  <= Goeie data (Golfbaan) (Kort)
    (100, False,     '20181006_180834.mp4'),  # 8  <= Goeie data (Golfbaan)
    (100, False,     '20181018_151747.mp4'),  # 9  <= Goeie data (Orchard)
    (100, False,     '20181018_152417.mp4'),  # 10 <= Unusable
    (100, False,     '20181018_153449.mp4'),  # 11 <= Goeie data (Orchard)
    (100, False,     '20181018_153944.mp4'),  # 12 <= Goeie data (Orchard)
    (100, False,     '20181018_155457.mp4'),  # 13 <= Unusable
    (100, False,     '20181018_155537.mp4'),  # 14 <= Unusable
    (100, False,     '20181018_161541.mp4'),  # 15 <= Unusable
    (100, False,     '20181018_161852.mp4'),  # 16 <= Goeie data
    (100, False, 'StrandGolfbaan_a_01.mp4'),  # 17
    (100, False, 'StrandGolfbaan_a_02.mp4'),  # 18
    (100, False, 'StrandGolfbaan_a_03.mp4'),  # 19
    (100, False, 'StrandGolfbaan_b_01.mp4'),  # 20
]
video_index = 4
frame_rate, flip_frame, video_name = available_videos[video_index]
videos = glob.glob(video_source)
file_names = [os.path.basename(video) for video in videos]
# max_length = max([len(name) for name in file_names])
# for i, name in enumerate(file_names):
#     print(("    ({}, {}, {: >"+str(max_length+2)+"}),  # {}").format(100, False, "'{}'".format(name), i))
video_url = videos[video_index]
video_file_name = os.path.basename(video_url)
window_title = '{} ({})'.format(video_file_name, video_name)

print('Project opened')
video_stream = VideoStream(video_url, grab=False, resize=(800, 500))
if video_stream.load():
    rectangle_stream = RectangleStream(video_stream.wh(), (16, 16), rows=15, margin=20)
    window_stream = WindowStream(window_title, frame_rate)
    state = StateManager(video_stream, rectangle_stream, window_stream)
while True:
    # noinspection PyUnboundLocalVariable
    if not video_stream.next():
        break
    # noinspection PyUnboundLocalVariable
    state.show()
    # noinspection PyUnboundLocalVariable
    key = window_stream.show(video_stream.color_frame())
    # noinspection PyUnboundLocalVariable
    if state.accept(key) is False:
        break
print('Project closed')
