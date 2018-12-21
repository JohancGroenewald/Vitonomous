import glob
import os
from itertools import chain

import numpy as np
import cv2
import color_constants as cc
from support import display
from scipy.ndimage import median_filter
from scipy.signal import convolve2d
from sklearn.cluster import KMeans

cv2.setUseOptimized(True)


class FileSelection(object):
    def __init__(self, source_url, source_list):
        self.source_url = source_url
        self.source_list = source_list

    def file_with_index(self, index, verbose=0):
        frame_rate, flip_frame, video_list_name, shape, scale = self.source_list[index]
        w, h = shape
        shape = (int(w*scale), int(h*scale))
        videos = glob.glob(self.source_url)
        if verbose > 0:
            file_names = [os.path.basename(video) for video in videos]
            max_length = max([len(name) for name in file_names])
            for i, name in enumerate(file_names):
                print(("    ({}, {}, {: >"+str(max_length+2)+"}),  # {}").format(100, False, "'{}'".format(name), i))
        video_url = videos[index]
        video_file_name = os.path.basename(video_url)
        return frame_rate, flip_frame, shape, video_list_name, video_url, video_file_name


class WindowStream(object):
    def __init__(self, window_name, frames_per_second):
        self.window_name = window_name
        if frames_per_second == -1:
            self.wait_delay = 1
        else:
            self.wait_delay = int(1000 / frames_per_second)
        self.state_manager_callback = None
        cv2.namedWindow(self.window_name)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_MOUSEMOVE:
            self.state_manager_callback(event, x, y)

    def attach_state_manager_callback(self, state_manager_callback):
        self.state_manager_callback = state_manager_callback
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def show(self, frame, wait_delay=None):
        cv2.imshow(self.window_name, frame)
        delay = self.wait_delay if wait_delay is None else wait_delay
        key = cv2.waitKeyEx(delay)
        return key


class VideoStream(object):
    def __init__(self, url, grab=True, auto_grab=False, resize=None, flip=None):
        self.capture = cv2.VideoCapture(url)
        self.grab = grab
        self.auto_grab = auto_grab
        self.resize = resize
        self.flip = flip
        self.re_shadow_frame = None
        self.shadow_frame = None
        self._color_frame = None
        self._gray_frame = None
        self._view_frame = None
        self.frame_store = []
        self.h = None
        self.w = None
        self.d = None
        self.frame_counter = 0
        self.color_view = True
        self.kernel = None
        self.cache = False
        self.tracking = 0

    def restart(self):
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def load(self):
        if self.read_frame():
            self.re_shadow_frame = self.shadow_frame.copy()
            self.h, self.w, self.d = self.shadow_frame.shape
            self.post_processing()
            self.assign_view_frame()
            return True
        return False

    def next(self):
        if self.grab or self.auto_grab:
            if self.shadow_frame is not None:
                self.re_shadow_frame = self.shadow_frame.copy()
            grabbed = self.read_frame()
        else:
            grabbed = True
        if grabbed:
            self.grab = self.auto_grab
            self.post_processing()
            self.assign_view_frame()
        return grabbed

    def read_frame(self):
        grabbed, self.shadow_frame = self.capture.read()
        if grabbed is False:
            self.shadow_frame = self.re_shadow_frame.copy()
        self.frame_counter = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
        if self.resize is not None:
            self.shadow_frame = cv2.resize(self.shadow_frame, self.resize)
        if self.flip is not None:
            self.shadow_frame = cv2.flip(self.shadow_frame, self.flip)
        return grabbed

    def post_processing(self):
        temp_image = self.shadow_frame
        # temp_image = cv2.cvtColor(self.shadow_frame, cv2.COLOR_BGR2GRAY)
        # if self.kernel is not None:
        #     temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
        # else:
        #     temp_image = cv2.cvtColor(self.shadow_frame, cv2.COLOR_BGR2GRAY)
        # self._color_frame = (-np.log(1/((1 + temp_image)/257) - 1)).astype("uint8")
        # self._color_frame = np.einsum("ijk,lk->ijl", self._color_frame, self.rotation_matrix(np.pi))
        # n=100
        # sobel_x = np.c_[
        #     [-1,0,1],
        #     [-2,0,2],q
        #     [-1,0,1]
        # ]
        #
        # sobel_y = np.c_[
        #     [1,2,1],
        #     [0,0,0],
        #     [-1,-2,-1]
        # ]
        # ims = []
        # for d in range(3):
        #     sx = convolve2d(temp_image[:,:,d], sobel_x, mode="same", boundary="symm")
        #     sy = convolve2d(temp_image[:,:,d], sobel_y, mode="same", boundary="symm")
        #     ims.append(np.sqrt(sx*sx + sy*sy))
        #
        # self._color_frame = np.stack(ims, axis=2).astype("uint8")
        # self._color_frame = self.median_filter_all_colours(temp_image, 71)
        # def simple_threshold(im, threshold=128):
        #     return ((im > threshold) * 255).astype("uint8")
        # temp_image = simple_threshold(temp_image)
        # thresholds = [100,120,128,138,150]

        # temp_image = cv2.cvtColor(self.shadow_frame, cv2.COLOR_BGR2GRAY)

        # h, w = temp_image.shape[:2]
        # im_small_long = temp_image.reshape((h * w, 3))
        # im_small_wide = im_small_long.reshape((h, w, 3))

        # km = KMeans(n_clusters=3)
        #
        # km.fit(im_small_long)

        # cc = km.cluster_centers_.astype(np.uint8)
        # self._color_frame = np.asarray([cc[i] for i in km.labels_]).reshape((h, w, 3))
        temp_image = cv2.cvtColor(self.shadow_frame, cv2.COLOR_BGR2GRAY)
        temp_image = cv2.medianBlur(temp_image, 7)
        # Frame store manager ################################################
        if len(self.frame_store) == 0:
            self.frame_store = [temp_image]
        else:
            del self.frame_store[0]
        self.frame_store.append(temp_image)

        # Final and mandatory gray and color frame assignment ################
        self._gray_frame = temp_image
        self._color_frame = self.shadow_frame.copy()

    def assign_view_frame(self):
        if self.color_view:
            self._view_frame = self._color_frame.copy()
        else:
            self._view_frame = cv2.cvtColor(self._gray_frame, cv2.COLOR_GRAY2BGR)

    def select_from_mask(self, temp_image):
        self.mask = np.zeros(temp_image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        xy, frame_data = self.kernel
        d, h, w = frame_data.shape
        h, w = h//2, w//2
        x, y = xy
        # rect = (start_x, start_y, width, height)
        rect = (x-w, y-h, x+w, y+h)
        cv2.grabCut(temp_image, self.mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        self.mask2 = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype('uint8')
        temp_image = temp_image*self.mask2[:, :, np.newaxis]
        return temp_image

    @staticmethod
    def remove_shadow(frame):
        rgb_planes = cv2.split(frame)
        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(
                diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
            )
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)
        frame = cv2.merge(result_planes)
        frame_norm = cv2.merge(result_norm_planes)
        return frame, frame_norm

    def assign_kernel(self, kernel):
        self.kernel = kernel
        self.cache = False

    def pick_color(self, x, y):
        return self._color_frame[y, x]

    def color_frame(self):
        return self._color_frame

    def gray_frame(self):
        return self._gray_frame

    def view_frame(self):
        return self._view_frame

    def wh(self):
        return self.w, self.h

    def disable_auto_grab(self):
        self.auto_grab = False

    def toggle_auto_grab(self):
        self.auto_grab = not self.auto_grab

    def toggle_grab(self, direction=1):
        position = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
        final_frame = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if direction == -1:
            # Goto the frame before the last frame
            self.frame_counter = position - 2
            if self.frame_counter < 0:
                self.frame_counter = final_frame - 1
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_counter)
        else:
            if position == final_frame:
                self.frame_counter = 0
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_counter)
        print(position, final_frame, self.frame_counter, direction)
        self.grab = not self.grab

    def toggle_view(self):
        self.color_view = not self.color_view


class AreaOfInterest(object):
    def __init__(self, r_w1, r_w2, r_h, shape, color, thickness):
        s_w, s_h = shape
        self.color = color
        self.thickness = thickness
        s_h -= 3
        w1, w2, h = s_w * r_w1, s_w * r_w2, s_h * r_h
        x1, y1 = (s_w - w1) // 2, s_h - h
        x3 = s_w - ((s_w - w2) // 2)
        self.xy1, self.xy2 = (int(x1), int(y1)), (int(x1 + w1), int(y1))
        self.xy3, self.xy4 = (int(x3), int(s_h)), (int(x3 - w2), int(s_h))
        points = np.array([self.xy1, self.xy2, self.xy3, self.xy4], np.int32)
        self.points = [points.reshape((-1, 1, 2))]

    def reshape(self, aoi_l, aoi_r):
        aoi_l.reverse()
        aoi_l.extend(aoi_r)
        self.prepare(aoi_l)

    def prepare(self, aoi):
        points = np.array(aoi, np.int32)
        self.points = [points.reshape((-1, 1, 2))]

    def render(self, frame, fill=False):
        if fill:
            cv2.fillPoly(frame, self.points, self.color.BGR(), lineType=cv2.LINE_AA)
        else:
            cv2.polylines(frame, self.points, True, self.color.BGR(), self.thickness, lineType=cv2.LINE_AA)


class RectangleStream(object):
    def __init__(self, wh, shape, bottom=-1, rows=-1, margin=-1):
        self.w, self.h = wh
        self.shape = shape
        self.r_w, self.r_h = shape
        self.bottom = bottom
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
        self.locked = False
        self.grid_shape = None
        self.build()
        self.select()

    def build(self):
        self.x_range, self.y_range = self.w // self.r_w, self.h // self.r_h
        self.balance_w, self.balance_h = (self.w - self.x_range * self.r_w) // 2, (
                self.h - self.y_range * self.r_h) // 2
        self.m_tl, self.m_br = (self.balance_w, self.balance_h), (self.w - self.balance_w, self.h - self.balance_h)
        self.rectangles = [[(int(self.balance_w + self.r_w * x), int(self.balance_h + self.r_h * y),
                             int((self.balance_w + self.r_w * x) + self.r_w),
                             int((self.balance_h + self.r_h * y) + self.r_h)) for x in range(self.x_range)] for y in
                           range(self.y_range)]
        self.rectangles.reverse()
        if self.bottom == -1:
            self.bottom = 0
        if self.rows == -1:
            self.rows = self.y_range
        if self.margin == -1:
            self.margin = 0
        self.grid_shape = (self.bottom, self.rows, self.margin, self.x_range - self.margin)

    def select(self, rows: int = 0, margin: int = 0, locked: bool = False):
        s1, s2, s3, s4 = self.grid_shape
        if locked:
            s1 += rows
            s2 += rows
            s3 -= margin
            s4 -= margin
        else:
            s2 += rows
            s3 += margin
            s4 -= margin
        # ##############################################################################################################
        self.grid_shape = (s1, s2, s3, s4)
        # ##############################################################################################################
        self.rectangles_flattened = [row[s3:s4] for row in self.rectangles[s1:s2]]
        self.rectangles_flattened = list(chain.from_iterable(self.rectangles_flattened))
        self.current = 0
        self.high = len(self.rectangles_flattened)

    def render_on(self, frame):
        for i, rectangle in enumerate(self.rectangles_flattened):
            tl, br = (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3])
            cv2.rectangle(frame, tl, br, cc.LIGHTYELLOW1.BGR(), 1)

    def sub_frame_from_xy(self, frame, x, y, in_color=False):
        l, _, _, b = self.rectangles_flattened[0]
        _, t, r, _ = self.rectangles_flattened[-1]
        rectangles = len(self.rectangles_flattened)
        if l <= x <= r and t <= y <= b and rectangles > 0:
            row = (b - y) // self.r_h
            column = (x - l) // self.r_w
            s1, s2, s3, s4 = self.grid_shape
            offset = row * (s4 - s3) + column
            # #########################################################################
            # display(l, x, r, t, y, b, row, column, offset, s2, s1)
            # #########################################################################
            if 0 <= offset < rectangles:
                rectangle = self.rectangles_flattened[offset]
                x_l, x_r, y_t, y_b = rectangle[0], rectangle[2], rectangle[1], rectangle[3]
                # #########################################################################
                x, y = self.center(rectangle)
                if in_color:
                    return offset, (x, y), frame[y_t:y_b, x_l:x_r, :]
                else:
                    return offset, (x, y), frame[y_t:y_b, x_l:x_r]
        return None, None, None

    def center(self, rectangle):
        return rectangle[0] + self.r_w // 2, rectangle[1] + self.r_h // 2

    def toggle_locked(self):
        self.locked = not self.locked

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
