from itertools import chain

import cv2
from support import display
import color_constants as cc


class WindowStream(object):
    def __init__(self, window_name, frames_per_second):
        self.window_name = window_name
        self.wait_delay = int(1000 / frames_per_second)
        self.state_manager_callback = None
        cv2.namedWindow(self.window_name)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.state_manager_callback(event, x, y)

    def attach_state_manager_callback(self, state_manager_callback):
        self.state_manager_callback = state_manager_callback
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def show(self, frame, wait_delay=None):
        cv2.imshow(self.window_name, frame)
        delay = self.wait_delay if wait_delay is None else wait_delay
        key = cv2.waitKey(delay) & 0xFF
        return key


class VideoStream(object):
    def __init__(self, url, grab=True, auto_grab=False, resize=None):
        self.capture = cv2.VideoCapture(url)
        self.grab = grab
        self.auto_grab = auto_grab
        self.resize = resize
        self.shadow_frame = None
        self._color_frame = None
        self._gray_frame = None
        self.h = None
        self.w = None
        self.d = None

    def load(self):
        grabbed, self.shadow_frame = self.capture.read()
        if grabbed:
            if self.resize is not None:
                self.shadow_frame = cv2.resize(self.shadow_frame, self.resize)
            self._color_frame = self.shadow_frame.copy()
            self._gray_frame = cv2.cvtColor(self._color_frame, cv2.COLOR_BGR2GRAY)
            self.h, self.w, self.d = self._color_frame.shape
        return grabbed

    def next(self):
        if self.grab or self.auto_grab:
            grabbed, self.shadow_frame = self.capture.read()
        else:
            grabbed = True
        if grabbed:
            if self.resize is not None:
                self.shadow_frame = cv2.resize(self.shadow_frame, self.resize)
            self._color_frame = self.shadow_frame.copy()
            self._gray_frame = cv2.cvtColor(self._color_frame, cv2.COLOR_BGR2GRAY)
            self.grab = self.auto_grab
        return grabbed

    def color_frame(self):
        return self._color_frame

    def gray_frame(self):
        return self._gray_frame

    def wh(self):
        return self.w, self.h

    def toggle_auto_grab(self):
        self.auto_grab = not self.auto_grab

    def toggle_grab(self):
        self.grab = not self.grab


class RectangleStream(object):
    def __init__(self, wh, r_w, r_h, rows=-1, margin=-1):
        self.w, self.h = wh
        self.r_w = r_w
        self.r_h = r_h
        self.rows = rows
        self.margin = margin
        self.area = self.r_w * self.r_h
        self.rectangles = None
        self.rectangles_flattened = None
        self.x_range = self.y_range = None
        self.balance_w = self.balance_h = None
        self.m_tl = self.m_br = None
        self.current = None
        self.high = None
        self.build()
        self.select()

    def build(self):
        self.x_range, self.y_range = self.w//self.r_w, self.h//self.r_h
        self.balance_w, self.balance_h = (self.w-self.x_range*self.r_w)//2, (self.h-self.y_range*self.r_h)//2
        self.m_tl, self.m_br = (self.balance_w, self.balance_h), (self.w-self.balance_w, self.h-self.balance_h)
        self.rectangles = [
            [
                (
                    int(self.balance_w+self.r_w*x), int(self.balance_h+self.r_h*y),
                    int((self.balance_w+self.r_w*x)+self.r_w), int((self.balance_h+self.r_h*y)+self.r_h)
                )
                for x in range(self.x_range)
            ] for y in range(self.y_range)
        ]
        self.rectangles.reverse()

    def select(self, rows: int=0, margin: int=0):
        self.rows += rows
        self.margin += margin
        if self.margin == 0:
            self.margin += margin
        if self.rows == -1:
            s1, s2 = 0, -1
        else:
            s1, s2 = 0, self.rows
        if self.margin == -1:
            s3, s4 = 0, len(self.rectangles[0])
        else:
            s3, s4 = self.margin, len(self.rectangles[0])-self.margin
        self.rectangles_flattened = [row[s3:s4] for row in self.rectangles[s1:s2]]
        self.rectangles_flattened = list(chain.from_iterable(self.rectangles_flattened))
        self.current = 0
        self.high = len(self.rectangles_flattened)

    def render_on(self, frame):
        for i, rectangle in enumerate(self.rectangles_flattened):
            tl, br = (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3])
            cv2.rectangle(frame, tl, br, cc.YELLOW1.BGR(), 1)

    def sub_frame_from_xy(self, frame, x, y):
        l_x, _, _, l_yy = self.rectangles_flattened[0]
        _, l_y, l_xx, _ = self.rectangles_flattened[-1]
        if l_x <= x <= l_xx and l_y <= y <= l_yy:
            r = self.y_range-1-y//self.r_h
            c = (x-l_x)//self.r_w
            margin = 0 if self.margin == -1 else self.margin
            offset = r*(self.x_range-margin*2)+c
            # #########################################################################
            rectangle = self.rectangles_flattened[offset]
            x_l, x_r, y_t, y_b = rectangle[0], rectangle[2], rectangle[1], rectangle[3]
            # #########################################################################
            return frame[y_t:y_b, x_l:x_r]
        return None

    def center(self, rectangle):
        return rectangle[0]+self.r_w//2, rectangle[1]+self.r_h//2

    def __len__(self):
        return len(self.rectangles_flattened)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.high:
            self.current = 0
            raise StopIteration
        else:
            self.current += 1
            return self.rectangles_flattened[self.current - 1]
