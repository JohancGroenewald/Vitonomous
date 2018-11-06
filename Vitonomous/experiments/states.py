import cv2
import numpy as np

from engines import VideoStream, RectangleStream, WindowStream
from datasets import DataSet
from classifiers import Classifications, NearestNeighbor

import color_constants as cc

# noinspection PyAttributeOutsideInit,PyArgumentList
class StateManager(object):
    def __init__(self, video_stream: VideoStream, rectangle_stream: RectangleStream, window_stream: WindowStream,
                 training_set: DataSet):
        self.video_stream = video_stream
        self.rectangle_stream = rectangle_stream
        self.window_stream = window_stream
        self.training_set = training_set
        # ##############################################################################################################
        self.window_stream.attach_state_manager_callback(self.state_manager_callback)
        # ##############################################################################################################
        self.key_quit = ord('q')
        self.key_next = ord('n')
        self.key_run = ord('r')
        self.key_plus_rectangle_row = ord('=')
        self.key_minus_rectangle_row = ord('-')
        self.key_plus_rectangle_column = ord('+')
        self.key_minus_rectangle_column = ord('_')
        # ##############################################################################################################
        self.key_grid = ord('g')
        # ##############################################################################################################
        self.key_classify = ord('c')
        self.key_train = ord('t')
        self.key_predict = ord('p')
        self.key_reset_classify = ord('r')
        self.key_1 = ord('1')
        self.key_2 = ord('2')
        self.key_3 = ord('3')
        self.key_4 = ord('4')
        # ##############################################################################################################
        self.grid_enabled = True
        # ##############################################################################################################
        self.classify = False
        self.classification = Classifications.IS_PATH
        self.classifier = NearestNeighbor()
        self.hidden_classes_set = []
        # ##############################################################################################################

    def accept(self, key):
        # ##############################################################################################################
        if key == self.key_quit:
            return False
        # ##############################################################################################################
        elif key == self.key_next:
            self.video_stream.toggle_grab()
        elif key == self.key_run:
            self.video_stream.toggle_auto_grab()
        elif key == self.key_plus_rectangle_row:
            self.rectangle_stream.select(1)
        elif key == self.key_minus_rectangle_row:
            self.rectangle_stream.select(-1)
        elif key == self.key_plus_rectangle_column:
            self.rectangle_stream.select(margin=-1)
        elif key == self.key_minus_rectangle_column:
            self.rectangle_stream.select(margin=1)
        # ##############################################################################################################
        elif key == self.key_grid:
            self.grid_enabled = not self.grid_enabled
        # ##############################################################################################################
        elif key == self.key_classify:
            self.classify = not self.classify
            print('MODE: classify {}'.format(self.classify))
            if self.classify:
                print('\nCLASSIFYING: {} '.format(Classifications.name(self.classification)), end='', flush=True)
        elif key == self.key_reset_classify:
            self.training_set.clear()
            print('Training data has been cleared')
        elif key == self.key_train:
            print()
            print('MODE: Training...', end='')
            self.train()
            print('done')
            self.classify = False
        elif key in [self.key_1, self.key_2, self.key_3, self.key_4]:
            classification = Classifications.IS_NAC if key == self.key_1 else \
                             Classifications.IS_PATH if key == self.key_2 else \
                             Classifications.IS_LIMIT if key == self.key_3 else \
                             Classifications.IS_ENVIRONMENT
            if self.classify:
                self.classification = classification
                print('\nCLASSIFYING: {} '.format(Classifications.name(self.classification)), end='', flush=True)
            elif classification not in self.hidden_classes_set:
                self.hidden_classes_set.append(classification)
            else:
                self.hidden_classes_set.remove(classification)
        return True

    def state_manager_callback(self, event, x, y):
        if self.classify:
            sub_frame = self.rectangle_stream.sub_frame_from_xy(self.video_stream.gray_frame(), x, y)
            if sub_frame is not None:
                self.training_set.push(self.classification, sub_frame.ravel())
                print('+', end='', flush=True)

    def train(self):
        train_X = np.zeros(shape=(len(self.training_set), self.rectangle_stream.area))
        train_y = np.ones(shape=(len(self.training_set)))
        keys, values = self.training_set.flatten()
        for i, value in enumerate(values):
            train_X[i, :] = value
            train_y[i] = keys[i]
        self.classifier.train(train_X, train_y)

    def show_predictions(self):
        if self.grid_enabled:
            self.rectangle_stream.render_on(self.video_stream.color_frame())
        if self.classify or len(self.classifier) == 0:
            return
        X = np.zeros(shape=(len(self.rectangle_stream), self.rectangle_stream.area))
        for i, rectangle in enumerate(self.rectangle_stream):
            x_l, x_r, y_t, y_b = rectangle[0], rectangle[2], rectangle[1], rectangle[3]
            frame = self.video_stream.gray_frame()[y_t:y_b, x_l:x_r]
            X[i, :] = frame.ravel()
        predictions = self.classifier.predict(X)
        for i, rectangle in enumerate(self.rectangle_stream):
            if predictions[i] in self.hidden_classes_set:
                continue
            x, y = self.rectangle_stream.center(rectangle)
            point = (x, y)
            if predictions[i] == Classifications.IS_NAC:
                radius = 3
                cv2.circle(self.video_stream.color_frame(), point, radius, cc.BLACK.BGR(), thickness=-1)
            elif predictions[i] == Classifications.IS_PATH:
                radius = 3
                cv2.circle(self.video_stream.color_frame(), point, radius, cc.BLUE.BGR(), thickness=-1)
            elif predictions[i] == Classifications.IS_LIMIT:
                padding = 3
                pts = np.array([
                    [x, y-padding], [x+padding, y+padding], [x-padding, y+padding], [x, y-padding]
                ], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(self.video_stream.color_frame(), [pts], True, cc.RED1.BGR(), thickness=2)
            elif predictions[i] == Classifications.IS_ENVIRONMENT:
                padding = 3
                tl, br = (x-padding, y-padding), (x+padding, y+padding)
                cv2.rectangle(self.video_stream.color_frame(), tl, br, cc.GREEN.BGR(), thickness=-1)
