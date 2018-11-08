import cv2
import numpy as np

from engines import VideoStream, RectangleStream, WindowStream
from datasets import TrainingSet
from classifiers import Classifications, NearestNeighbor

import color_constants as cc


# noinspection PyAttributeOutsideInit,PyArgumentList
class StateManager(object):
    def __init__(self, video_stream: VideoStream, rectangle_stream: RectangleStream, window_stream: WindowStream):
        self.video_stream = video_stream
        self.rectangle_stream = rectangle_stream
        self.window_stream = window_stream
        self.training_set = TrainingSet()
        self.classification_set = TrainingSet()
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
        self.key_lock_rectangle = ord('\\')
        # ##############################################################################################################
        self.key_grid = ord('g')
        # ##############################################################################################################
        self.key_classify = ord('c')
        self.key_train = ord('t')
        self.key_predict = ord('p')
        self.key_reset_classifications = ord('R')
        self.key_save_classifications = ord('s')
        self.key_load_classifications = ord('l')
        self.key_0 = ord('0')
        self.key_1 = ord('1')
        self.key_2 = ord('2')
        self.key_3 = ord('3')
        self.key_4 = ord('4')
        self.key_5 = ord('5')
        self.key_6 = ord('6')
        self.key_7 = ord('7')
        self.key_8 = ord('8')
        self.key_9 = ord('9')
        self.key_list = [
            self.key_0,
            self.key_1,
            self.key_2,
            self.key_3,
            self.key_4,
            self.key_5,
            self.key_6,
            self.key_7,
            self.key_8,
            self.key_9,
        ]
        # ##############################################################################################################
        self.key_enable_blanking = ord('B')
        self.key_cycle_blanking = ord('b')
        self.key_block_mode = 2
        self.key_remember_class = 18
        # ##############################################################################################################
        self.key_generate = ord('j')
        # ##############################################################################################################
        self.grid_enabled = True
        # ##############################################################################################################
        self.classify = False
        self.classification = Classifications.IS_CLASS_1
        self.classifier = NearestNeighbor()
        self.hidden_classes_set = []
        self.remembered_classes_set = []
        self.blank = False
        self.block_mode = False
        # ##############################################################################################################
        self.classify_session = []
        # ##############################################################################################################
        self.class_colors = [
            cc.MAROON,      # 0
            cc.BROWN,       # 1
            cc.OLIVE,       # 2
            cc.TEAL,        # 3
            cc.NAVY,        # 4
            cc.BLACK,       # 5
            cc.RED1,        # 6
            cc.ORANGE,      # 7
            cc.YELLOW1,     # 8
            cc.LIMEGREEN,   # 9
            cc.GREEN,       # 10
            cc.CYAN2,       # 11
            cc.BLUE,        # 12
            cc.PURPLE,      # 13
            cc.MAGENTA,     # 14
            cc.GRAY,        # 15
            cc.PINK,        # 16
            cc.APRICOT,     # 17
            cc.BEIGE,       # 18
            cc.MINT,        # 19
            cc.LAVENDER,    # 20
            cc.WHITE,       # 21
        ]
        # ##############################################################################################################
        self.supported_classes = Classifications.IS_CLASS_19
        self.re_key = 0
        # ##############################################################################################################

    def accept(self, key):
        if key == self.re_key:
            return
        self.re_key = key
        # (flags&0xFFFF==cv2.EVENT_FLAG_CTRLKEY)
        # ctrl_key = True if key == 17 else False
        # print('------------------------')
        # print('key    : {}'.format(key))
        # print('key 16  : {}'.format(key & 0xFF))
        # print('key 32 : {}'.format(key & 0xFFFF))
        # print('key 64 : {}'.format(key & 0xFFFFFFFF))

        key = key & 0xFF
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
        elif key == self.key_lock_rectangle:
            self.rectangle_stream.toggle_locked()
        # ##############################################################################################################
        elif key == self.key_grid:
            self.grid_enabled = not self.grid_enabled
        # ##############################################################################################################
        elif key == self.key_classify:
            self.state_toggle_classification()
        elif key == self.key_train:
            self.state_run_training()
        elif key == self.key_predict:
            pass
        elif key in self.key_list:
            self.state_select_classifications(key)
        # ##############################################################################################################
        elif key == self.key_reset_classifications:
            self.state_reset_classifications()
        elif key == self.key_save_classifications:
            self.state_save_classifications()
        elif key == self.key_load_classifications:
            self.state_load_classifications()
        # ##############################################################################################################
        elif key == self.key_generate:
            self.classifications_generate()
        # ##############################################################################################################
        elif key == self.key_enable_blanking:
            self.state_toggle_blanking()
        elif key == self.key_cycle_blanking:
            self.cycle_blanking()
        elif key == self.key_block_mode:
            self.cycle_block_mode()
        elif key == self.key_remember_class:
            self.remember_class()
        return True

    # ##################################################################################################################
    def state_manager_callback(self, event, x, y):
        if self.classify:
            self.validate_class_selection(x, y)
    # ##################################################################################################################

    def validate_class_selection(self, x, y):
        xy, sub_frame = self.rectangle_stream.sub_frame_from_xy(self.video_stream.gray_frame(), x, y)
        if sub_frame is not None:
            self.push_classification(self.video_stream.frame_counter, xy)
            self.push_sub_frame(sub_frame, self.video_stream.frame_counter)

    def push_classification(self, frame_counter, xy):
        self.classification_set.push(frame_counter, xy, self.classification)

    def push_sub_frame(self, sub_frame, frame_counter):
        self.training_set.push(frame_counter, self.classification, sub_frame.ravel())

    def classifications_generate(self):
        classes = self.supported_classes + 1
        class_sets = 2
        s, t = 1, 227
        slice = t/classes
        for c in range(classes):
            for r in range(class_sets):
                _s, _t = int(s+(slice*c)), int(s+(slice*(c+1)))
                sub_frame = np.random.random_integers(_s, _t, self.rectangle_stream.shape)
                self.training_set.push(0, c, sub_frame.ravel())

    def state_toggle_classification(self):
        self.classify = not self.classify
        if not self.classify:
            print()
        print('MODE: Classification {}'.format('Enabled' if self.classify else 'Disabled'))
        if self.classify:
            self.print_classifying(self.classification)

    def state_reset_classifications(self):
        self.training_set.clear()
        print('Training data has been cleared')

    def state_save_classifications(self):
        print('MODE: Saving classifications...', end='')
        self.training_set.save()
        self.classifier.save()
        print('done')

    def state_load_classifications(self):
        print('MODE: Loading classifications...', end='')
        self.training_set.load()
        self.classifier.load()
        print('done')

    # ##################################################################################################################
    def state_run_training(self):
        print('MODE: Training...', end='')
        self.train_classifier_nearest_neighbor()
        print('done')
        self.classify = False

    def state_select_classifications(self, key):
        classification = self.key_list.index(key)
        if self.classify:
            self.classification = classification
            self.print_classifying(classification)
        elif classification not in self.hidden_classes_set:
            self.hidden_classes_set.append(classification)
            self.print_excluding(classification, exclude=True)
        else:
            self.hidden_classes_set.remove(classification)
            self.print_excluding(classification, exclude=False)

    def state_toggle_blanking(self):
        self.blank = not self.blank
        print('MODE: {} hidden classes'.format('Blanking' if self.blank else 'UnBlanking'))
        if not self.blank:
            self.hidden_classes_set = []
            self.remembered_classes_set = []

    def cycle_blanking(self):
        classification = self.classification
        classification += 1
        if classification > self.supported_classes:
            classification = Classifications.IS_NAC
        self.hidden_classes_set = [
            c for c in range(Classifications.IS_CLASS_20)
            if c != classification and c not in self.remembered_classes_set
        ]
        self.print_excluding(classification, exclude=True)
        self.classification = classification

    def cycle_block_mode(self):
        self.block_mode = not self.block_mode
        if self.block_mode:
            self.classification = -1
            self.cycle_blanking()
        else:
            self.hidden_classes_set = []

    def remember_class(self):
        if self.classification not in self.remembered_classes_set:
            self.remembered_classes_set.append(self.classification)

    def print_classifying(self, classification):
        color_name = list(cc.colors.keys())[list(cc.colors.values()).index(self.class_colors[classification])]
        print('CLASSIFYING: {} <{}>'.format(Classifications.name(classification), color_name))

    def print_excluding(self, classification, exclude):
        color_name = list(cc.colors.keys())[list(cc.colors.values()).index(self.class_colors[classification])]
        state_string = 'EXCLUDING' if exclude else 'INCLUDING'
        print('{}: {} <{}>'.format(state_string, Classifications.name(classification), color_name))

    def train_classifier_nearest_neighbor(self):
        train_X = np.zeros(shape=(len(self.training_set), self.rectangle_stream.area))
        train_y = np.ones(shape=(len(self.training_set)))
        keys, values = self.training_set.flatten()
        for i, value in enumerate(values):
            train_X[i, :] = value
            train_y[i] = keys[i]
        self.classifier.train(train_X, train_y)

    def query_classifier(self):
        X = np.zeros(shape=(len(self.rectangle_stream), self.rectangle_stream.area))
        for i, rectangle in enumerate(self.rectangle_stream):
            x_l, x_r, y_t, y_b = rectangle[0], rectangle[2], rectangle[1], rectangle[3]
            frame = self.video_stream.gray_frame()[y_t:y_b, x_l:x_r]
            X[i, :] = frame.ravel()
        return self.classifier.predict(X)

    def show_predictions(self, predictions):
        for i, rectangle in enumerate(self.rectangle_stream):
            prediction = int(predictions[i])
            if prediction in self.hidden_classes_set:
                if self.blank:
                    tl, br = (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3])
                    cv2.rectangle(self.video_stream.color_frame(), tl, br, cc.YELLOW1.BGR(), -1)
            else:
                x, y = self.rectangle_stream.center(rectangle)
                class_color = self.class_colors[prediction]
                if self.block_mode:
                    tl, br = (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3])
                    cv2.rectangle(self.video_stream.color_frame(), tl, br, class_color.BGR(), -1)
                else:
                    point = (x, y)
                    radius = 3
                    cv2.circle(self.video_stream.color_frame(), point, radius, class_color.BGR(), thickness=-1)

    def show_grid(self):
        if self.grid_enabled:
            self.rectangle_stream.render_on(self.video_stream.color_frame())

    def show_classification_selection(self):
        keys, values = self.classification_set.flatten(self.video_stream.frame_counter)
        for i, point in enumerate(keys):
            classification = values[i]
            radius = 3
            class_color = self.class_colors[classification]
            cv2.circle(self.video_stream.color_frame(), point, radius, class_color.BGR(), thickness=-1)
        pass

    def show(self):
        self.show_grid()
        if self.classify:
            self.show_classification_selection()
        elif len(self.classifier) > 0:
            predictions = self.query_classifier()
            self.show_predictions(predictions)

    def save(self):
        pass

    def load(self):
        pass
