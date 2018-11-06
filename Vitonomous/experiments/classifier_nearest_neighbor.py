import glob
import itertools
from datetime import datetime

import cv2
import numpy as np


def display(*args):
    import inspect, re
    calling_frame = inspect.currentframe().f_back
    calling_context = inspect.getframeinfo(calling_frame, 1).code_context[0]
    arguments = calling_context.split('(', 1)[1].split(')')[0]
    s = ['-> ']
    for i, argument in enumerate(arguments.split(',')):
        s.append('{:.<4}{:.>4}'.format(argument, str(args[i])))
    print(' |'.join(s))

class CoordinateStore:
    def __init__(self):
        self.points = []
        self.classification = []
        self.state = 1

    def select_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.points.append((x, y))
            self.classification.append(self.state)

    @staticmethod
    def point2offset(point):
        x, y = point
        x, y = x-balance_w, y-balance_h
        offset = ((y//s_h)*x_range)+(x//s_w)
        return offset

class Classifier:
    IS_NAC = 0
    IS_PATH = 1
    IS_LIMIT = 2
    IS_ENVIRONMENT = 3

class NearestNeighbor:
    def __init__(self):
        self.Xtr = None
        self.ytr = None
    def train(self, X, y):
        self.Xtr = X
        self.ytr = y
    def predict(self, X):
        num_test = X.shape[0]
        # display(num_test)
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)
        for i in range(num_test):
            distance = np.sum(np.abs(self.Xtr - X[i,:]), axis=1)
            # display(distance)
            min_index = np.argmin(distance)
            Ypred[i] = self.ytr[min_index]
        return Ypred

coordinate_store = CoordinateStore()
window_name = 'Camera Stream'
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, coordinate_store.select_point)

video_source = r'S:\Development\CartonomousSources\sources\golfbaan_strand\videos\*.mp4'

videos = glob.glob(video_source)
video_url = videos[3]

print('Opening frame stream')
processing_window = 25
wait_delay = int(1000 / 30) - processing_window
capture = cv2.VideoCapture(video_url)

frame_index = -1
session_directory = datetime.now().strftime('%Y%m%d_%H%M%S')
session_url = None

s_w, s_h = None, None
rectangle_set = []
hide_set = []
# ###################################################################
nearest_neighbor = NearestNeighbor()
# ###################################################################
grab_next = True
auto_grab_next = False
trained = False
extended_training = False
# ###################################################################
grabbed, re_frame = capture.read()
if grabbed:
    h, w, d = re_frame.shape
    s_w, s_h = 16, 16
    x_range, y_range = w//s_w, h//s_h
    balance_w, balance_h = (w-x_range*s_w)//2, (h-y_range*s_h)//2
    m_tl, m_br = (balance_w, balance_h), (w-balance_w, h-balance_h)
    rectangle_set = [
        [
            (
                int(balance_w+s_w*x), int(balance_h+s_h*y),
                int((balance_w+s_w*x)+s_w), int((balance_h+s_h*y)+s_h)
            )
            for x in range(x_range)
        ] for y in range(y_range)
    ]
    rectangle_set = list(itertools.chain.from_iterable(rectangle_set))
while grabbed:
    if grab_next or auto_grab_next:
        grab_next = False
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
        # diff_frame = cv2.absdiff(gray_frame_in, gray_re_frame)
        # kernel_sharpening = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        # frame_out = cv2.filter2D(gray_frame_in, -1, kernel_sharpening)
        frame_out = frame_in.copy()
        # ###############################################################
        # frame_out = gray_frame_in.copy()
        # gray_re_frame = imutils.resize(gray_re_frame, 320, 240)
        # ###############################################################
        if trained:
            X = np.zeros(shape=(len(rectangle_set), s_h*s_w))
        for i_rectangle, rectangle in enumerate(rectangle_set):
            tl, br = (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3])
            cv2.rectangle(frame_out, tl, br, (128, 128, 128), 1)
            if trained:
                x_l, x_r, y_t, y_b = rectangle[0], rectangle[2], rectangle[1], rectangle[3]
                prediction_rectangle = gray_frame_in[y_t:y_b, x_l:x_r]
                X[i_rectangle, :] = prediction_rectangle.ravel()
        if trained:
            predictions = nearest_neighbor.predict(X)
            for i_rectangle, rectangle in enumerate(rectangle_set):
                c_x, c_y = rectangle[0]+s_w//2, rectangle[1]+s_h//2
                point = (c_x, c_y)
                if predictions[i_rectangle] == Classifier.IS_PATH and Classifier.IS_PATH not in hide_set:
                    radius = 3
                    cv2.circle(frame_out, point, radius, (255, 0, 0), thickness=-1)
                if predictions[i_rectangle] == Classifier.IS_LIMIT and Classifier.IS_LIMIT not in hide_set:
                    padding = 3
                    pts = np.array([
                        [c_x, c_y-padding], [c_x+padding, c_y+padding], [c_x-padding, c_y+padding], [c_x, c_y-padding]
                    ], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(frame_out, [pts], True, (0, 0, 255), thickness=2)
                if predictions[i_rectangle] == Classifier.IS_ENVIRONMENT and Classifier.IS_ENVIRONMENT not in hide_set:
                    padding = 3
                    tl, br = (c_x-padding, c_y-padding), (c_x+padding, c_y+padding)
                    cv2.rectangle(frame_out, tl, br, (0, 255, 0), thickness=-1)

        # ###############################################################
        # cv2.rectangle(frame_out, m_tl, m_br, (255, 255, 255), 1)
        # ###############################################################
        # frame_out = gray_re_frame.copy()
        # ###############################################################
    if not trained:
        for i_point, point in enumerate(coordinate_store.points):
            offset = coordinate_store.point2offset(point)
            rectangle = rectangle_set[offset]
            c_x, c_y = rectangle[0]+s_w//2, rectangle[1]+s_h//2
            point = (c_x, c_y)
            if coordinate_store.classification[i_point] == Classifier.IS_PATH:
                radius = 3
                cv2.circle(frame_out, point, radius, (255, 255, 255), -1)
            if coordinate_store.classification[i_point] == Classifier.IS_LIMIT:
                pts = np.array([
                    [c_x, c_y-3], [c_x+3, c_y+3], [c_x-3, c_y+3], [c_x, c_y-3]
                ], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame_out, [pts], True, (255, 255, 255), thickness=2)
            if coordinate_store.classification[i_point] == Classifier.IS_ENVIRONMENT:
                tl, br = (c_x-3, c_y-3), (c_x+3, c_y+3)
                cv2.rectangle(frame_out, tl, br, (255, 255, 255), thickness=-1)

    # ###############################################################
    cv2.imshow(window_name, frame_out)
    # ###############################################################
    # ###############################################################
    key = cv2.waitKey(delay=wait_delay) & 0xFF
    if key == ord('1'):
        if not trained:
            coordinate_store.state = Classifier.IS_PATH
        elif Classifier.IS_PATH not in hide_set:
            hide_set.append(Classifier.IS_PATH)
        else:
            hide_set.remove(Classifier.IS_PATH)
    elif key == ord('2'):
        if not trained:
            coordinate_store.state = Classifier.IS_LIMIT
        elif Classifier.IS_LIMIT not in hide_set:
            hide_set.append(Classifier.IS_LIMIT)
        else:
            hide_set.remove(Classifier.IS_LIMIT)
    elif key == ord('3'):
        if not trained:
            coordinate_store.state = Classifier.IS_ENVIRONMENT
        elif Classifier.IS_ENVIRONMENT not in hide_set:
            hide_set.append(Classifier.IS_ENVIRONMENT)
        else:
            hide_set.remove(Classifier.IS_ENVIRONMENT)
    elif key == ord('q'):
        break
    elif key == ord('n'):
        grab_next = True
    elif key == ord('r'):
        auto_grab_next = not auto_grab_next
    elif key == ord('t') and trained is True:
        grab_next = True
        trained = False
        extended_training = True
    elif key == ord('t') and trained is False:
        train_X = np.zeros(shape=(len(coordinate_store.points), s_h*s_w))
        train_y = np.ones(shape=(len(coordinate_store.points)))
        for i, point in enumerate(coordinate_store.points):
            offset = coordinate_store.point2offset(point)
            rectangle = rectangle_set[offset]
            x_l, x_r, y_t, y_b = rectangle[0], rectangle[2], rectangle[1], rectangle[3]
            training_rectangle = gray_frame_in[y_t:y_b, x_l:x_r]
            train_X[i, :] = training_rectangle.ravel()
            train_y[i] = coordinate_store.classification[i]
            # #################################################################
            # tl, br = (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3])
            # cv2.rectangle(frame_out, tl, br, (255, 255, 255), 3)
            # #################################################################
        nearest_neighbor.train(train_X, train_y)
        grab_next = True
        trained = True
    elif key == ord('s'):
        pass
        # if session_url is None:
        #      session_url = os.path.join(Sources.ROOT, Sources.SOURCE, Sources.TILES, session_directory)
        #      os.mkdir(session_url)
        # # # image = PIL.Image.fromarray(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB))
        # # image = PIL.Image.fromarray(frame_out)
        # save_url = os.path.join(
        #      session_url, '{}.png'.format(frame_index)
        # )
        # cv2.imwrite(save_url, frame_in)
        #
        # image_slicer.slice_image(image, 40, save_url)
    # ###############################################################
    if grab_next:
        re_frame = frame_in
    # ###############################################################
