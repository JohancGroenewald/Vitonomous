import cv2
import numpy as np

from engines import VideoStream, RectangleStream, WindowStream, AreaOfInterest
from datasets import TrainingSet
from classifiers import Classifications, NearestNeighbor

import color_constants as cc


# noinspection PyAttributeOutsideInit,PyArgumentList
class StateManager(object):
    def __init__(self, video_stream: VideoStream, rectangle_stream: RectangleStream, window_stream: WindowStream):
        self.video_stream = video_stream
        self.rectangle_stream = rectangle_stream
        self.window_stream = window_stream
        # ##############################################################################################################
        self.training_set = TrainingSet()
        self.classification_set = TrainingSet()
        self.driving_aoi = AreaOfInterest(0.5, 0.7, 0.5, self.video_stream.wh(), cc.WHITE)
        self.tracking_aoi = AreaOfInterest(0.1, 0.1, 0.5, self.video_stream.wh(), cc.YELLOW1)
        # ##############################################################################################################
        self.window_stream.attach_state_manager_callback(self.state_manager_callback)
        # ##############################################################################################################
        self.key_bindings = {
            ord('q'): (self.quit_application, {}),
            ord('n'): (self.video_stream.toggle_grab, {}),
            ord('r'): (self.video_stream.toggle_auto_grab, {}),
            ord('='): (self.rectangle_stream.select, {'rows': 1}),
            ord('-'): (self.rectangle_stream.select, {'rows': -1}),
            ord('+'): (self.rectangle_stream.select, {'margin': -1}),
            ord('_'): (self.rectangle_stream.select, {'margin': 1}),
            2490368: (self.rectangle_stream.select, {'rows': 1, 'locked': True}),
            2621440: (self.rectangle_stream.select, {'rows': -1, 'locked': True}),
            2424832: (self.rectangle_stream.select, {'margin': 1, 'locked': True}),
            2555904: (self.rectangle_stream.select, {'margin': -1, 'locked': True}),
            # ##########################################################################################################
            ord('g'): (self.toggle_show_grid, {}),
            ord('G'): (self.toggle_render_grid_content, {}),
            # ##########################################################################################################
            ord('c'): (self.state_toggle_classification, {}),
            ord('t'): (self.state_run_training, {}),
            # ord('p'): (pass, {}),
            ord('R'): (self.state_reset_classifications, {}),
            ord('s'): (self.state_save_classifications, {}),
            ord('l'): (self.state_load_classifications, {}),
            ord('B'): (self.state_toggle_blanking, {}),
            ord('b'): (self.cycle_blanking, {}),
            2: (self.cycle_block_mode, {}),     # ^b
            18: (self.remember_class, {}),      # ^r
            ord('j'): (self.classifications_generate, {}),
            ord('0'): (self.state_select_classifications, {'key': 0}),
            ord('1'): (self.state_select_classifications, {'key': 1}),
            ord('2'): (self.state_select_classifications, {'key': 2}),
            ord('3'): (self.state_select_classifications, {'key': 3}),
            ord('4'): (self.state_select_classifications, {'key': 4}),
            ord('5'): (self.state_select_classifications, {'key': 5}),
            ord('6'): (self.state_select_classifications, {'key': 6}),
            ord('7'): (self.state_select_classifications, {'key': 7}),
            ord('8'): (self.state_select_classifications, {'key': 8}),
            ord('9'): (self.state_select_classifications, {'key': 9}),
        }
        # ##############################################################################################################
        self.grid_enabled = True
        self.grid_rendered = True
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
        self.supported_classes = Classifications.IS_CLASS_3
        self.re_key = 0
        # ##############################################################################################################

    def accept(self, key):
        if key == -1:
            pass
        elif key in self.key_bindings:
            method, kwargs = self.key_bindings[key]
            return method(**kwargs)
        else:
            print('key: {} {} {}'.format(key, bin(key), hex(key)))
        return True

    # ##################################################################################################################
    def state_manager_callback(self, event, x, y):
        if self.classify:
            self.validate_class_selection(x, y)
    # ##################################################################################################################

    @staticmethod
    def quit_application():
        return False

    def toggle_show_grid(self):
        self.grid_enabled = not self.grid_enabled

    def toggle_render_grid_content(self):
        self.grid_rendered = not self.grid_rendered

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
        # s, t = 1, 227
        s, t = 0, 255
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
        print(self.remembered_classes_set)

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

    def map_aoi(self, predictions, classes):
        aoi_l = []
        aoi_r = []
        if len(self.rectangle_stream) == 0:
            return aoi_l, aoi_r
        re_xy = self.rectangle_stream.center(self.rectangle_stream.rectangles_flattened[0])
        next_row = True
        for i, rectangle in enumerate(self.rectangle_stream):
            if predictions[i] not in classes:
                continue
            xy = self.rectangle_stream.center(rectangle)
            if next_row:
                aoi_l.append(re_xy if len(aoi_l) > 0 else xy)
            next_row = re_xy[1] != xy[1]
            if next_row:
                if len(aoi_l) == len(aoi_r):
                    aoi_l.append(re_xy)
                aoi_r.append(re_xy)
            re_xy = xy
        samples = len(aoi_l)+len(aoi_r)
        if samples > 0:
            aoi_r.append(re_xy)
        return samples, aoi_l, aoi_r

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

    def show(self):
        self.show_grid()
        if self.classify:
            self.show_classification_selection()
        elif len(self.classifier) > 0:
            predictions = self.query_classifier()
            if self.grid_rendered:
                self.show_predictions(predictions)
            # self.driving_aoi.render(self.video_stream.color_frame())
            samples, aoi_l, aoi_r = self.map_aoi(predictions, self.remembered_classes_set)
            if samples > 0:
                self.tracking_aoi.reshape(aoi_l, aoi_r)
                self.tracking_aoi.render(self.video_stream.color_frame())

    def save(self):
        pass

    def load(self):
        pass
