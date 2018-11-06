import glob

from sources import video_source
from engines import WindowStream, VideoStream, RectangleStream
from states import StateManager
from datasets import DataSet
from classifiers import Classifications, NearestNeighbor

videos = glob.glob(video_source)
video_url = videos[3]

print('Project opened')
video_stream = VideoStream(video_url, grab=False)
if video_stream.load():
    rectangle_stream = RectangleStream(video_stream.wh(), 16, 16, 10)
    window_stream = WindowStream('Camera Stream', 30)
    training_set = DataSet()
    state = StateManager(video_stream, rectangle_stream, window_stream, training_set)
while True:
    # noinspection PyUnboundLocalVariable
    if not video_stream.next():
        break
    # noinspection PyUnboundLocalVariable
    state.show_predictions()
    # noinspection PyUnboundLocalVariable
    key = window_stream.show(video_stream.color_frame())
    # noinspection PyUnboundLocalVariable
    if state.accept(key) is False:
        break
print('Project closed')
