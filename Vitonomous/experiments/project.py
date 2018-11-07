import glob

from sources import video_source
from engines import WindowStream, VideoStream, RectangleStream
from states import StateManager
from datasets import DataSet
from classifiers import Classifications, NearestNeighbor

available_videos = [
    (100, '20181018_151747.mp4'),          # 0
    (100, '20181018_152417.mp4'),          # 1
    (100, '20181018_153449.mp4'),          # 2
    (100, '20181018_153944.mp4'),          # 3
    (100, '20181018_155457.mp4'),          # 4
    (100, '20181018_155537.mp4'),          # 5
    (100, '20181018_161541.mp4'),          # 6
    (100, '20181018_161852.mp4'),          # 7
    (40, 'StrandGolfbaan_a_01.mp4'),      # 8
    (40, 'StrandGolfbaan_a_02.mp4'),      # 9
    (40, 'StrandGolfbaan_a_03.mp4'),      # 10
    (20, 'StrandGolfbaan_b_01.mp4'),      # 11    <= Good start
]
video_index = 10
frame_rate, video_name = available_videos[video_index]
videos = glob.glob(video_source)
video_url = videos[video_index]

print('Project opened')
video_stream = VideoStream(video_url, grab=False, resize=(800, 500))
if video_stream.load():
    rectangle_stream = RectangleStream(video_stream.wh(), 8, 8, 10)
    window_stream = WindowStream('Camera Stream', frame_rate)
    training_set = DataSet()
    state = StateManager(video_stream, rectangle_stream, window_stream, training_set)
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
