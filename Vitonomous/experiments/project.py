import glob
import os

from sources import video_source
from engines import FileSelection, WindowStream, VideoStream, RectangleStream
from states import StateManager

video_list = [
    (-1, False,     '20181006_175023.mp4', (1920, 1080), 0.8),  # 0  <= Short but usable
    (-1, False,     '20181006_175105.mp4', (1920, 1080), 0.8),  # 1  <= Goeie data (Golfbaan) (Kort)
    (-1, False,     '20181006_175204.mp4', (1920, 1080), 0.8),  # 2  <= Goeie data (Golfbaan) (Lank)
    (-1, False,     '20181006_175546.mp4', (1920, 1080), 0.8),  # 3  <= Goeie data (Golfbaan) (Kort) (Hill parking)
    (-1, False,     '20181006_175653.mp4', (1920, 1080), 0.8),  # 4  <= Goeie data (Golfbaan) (Lank)
    (-1, False,     '20181006_175934.mp4', (1920, 1080), 0.8),  # 5  <= Goeie data (Golfbaan) (Medium) (Grond pad)
    (-1, False,     '20181006_180356.mp4', (1920, 1080), 0.8),  # 6  <= Goeie data (Golfbaan) (Lank) (Bakstene en Brug)
    (-1, False,     '20181006_180654.mp4', (1920, 1080), 0.8),  # 7  <= Goeie data (Golfbaan) (Kort)
    (-1, False,     '20181006_180834.mp4', (1920, 1080), 0.8),  # 8  <= Goeie data (Golfbaan) (Medium) (Pad na Cart garage)
    (-1, False,     '20181018_151747.mp4', (3840, 2160), 0.3),  # 9  <= Goeie data (Orchard) (Jong bome)
    (-1, False,     '20181018_152417.mp4', (3840, 2160), 0.3),  # 10 <= Unusable
    (-1, False,     '20181018_153449.mp4', (3840, 2160), 0.3),  # 11 <= Goeie data (Orchard)
    (-1, False,     '20181018_153944.mp4', (3840, 2160), 0.3),  # 12 <= Goeie data (Orchard)
    (-1, False,     '20181018_155457.mp4', (3840, 2160), 0.3),  # 13 <= Unusable
    (-1, False,     '20181018_155537.mp4', (3840, 2160), 0.3),  # 14 <= Unusable
    (-1, False,     '20181018_161541.mp4', (3840, 2160), 0.3),  # 15 <= Unusable
    (-1, False,     '20181018_161852.mp4', (3840, 2160), 0.3),  # 16 <= Goeie data
    (-1, False, 'StrandGolfbaan_a_01.mp4', (848, 480), 1.3),    # 17 <= Kort en bruikbaar
    (-1, False, 'StrandGolfbaan_a_02.mp4', (848, 480), 1.3),    # 18
    (-1, False, 'StrandGolfbaan_a_03.mp4', (848, 480), 1.3),    # 19
    (-1, False, 'StrandGolfbaan_b_01.mp4', (848, 480), 1.3),    # 20 <= Kort en bruikbaar
]
frame_rate, flip_frame, wh, video_list_name, video_url, video_file_name = \
    FileSelection(video_source, video_list).file_with_index(
        4
    )

print('Project opened')
window_title = '{} ({})'.format(video_file_name, video_list_name)
video_stream = VideoStream(video_url, grab=False, resize=wh)
if video_stream.load():
    rectangle_stream = RectangleStream(video_stream.wh(), (8, 16), rows=15, margin=1)
    window_stream = WindowStream(window_title, frame_rate)
    state = StateManager(video_stream, rectangle_stream, window_stream)
while True:
    # noinspection PyUnboundLocalVariable
    if not video_stream.next():
        video_stream.disable_auto_grab()
    # noinspection PyUnboundLocalVariable
    state.show()
    # noinspection PyUnboundLocalVariable
    key = window_stream.show(video_stream.view_frame())
    # noinspection PyUnboundLocalVariable
    if state.accept(key) is False:
        break
print('Project closed')
