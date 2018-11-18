import cv2
import numpy as np

from engines import VideoStream, RectangleStream, WindowStream, AreaOfInterest
from datasets import TrainingSet
from classifiers import Classifications, NearestNeighbor, NeuralNetwork
from support import display
from classifiers_mxnet import MxnetClassifier

import color_constants as cc


# noinspection PyAttributeOutsideInit,PyArgumentList
class StateManager(object):
    def __init__(self, video_stream: VideoStream, rectangle_stream: RectangleStream, window_stream: WindowStream):
        self.video_stream = video_stream
        self.rectangle_stream = rectangle_stream
        self.window_stream = window_stream
        # ##############################################################################################################
        self.supported_classes = Classifications.IS_CLASS_8
        # CLASS 1: Tarmac
        # CLASS 2: Side Paving
        # CLASS 3: Brick Paving
        # CLASS 4: Earth
        # CLASS 5: Stones
        # CLASS 6: Wood
        # CLASS 7: Cables
        # CLASS 8: Environment
        # ##############################################################################################################
        self.classifier = MxnetClassifier(inputs=self.rectangle_stream.area, classes=self.supported_classes)
        self.train_classifier = self.train_classifier_mxnet
        self.query_classifier = self.query_classifier_mxnet
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
            ord('m')-ord('`'): (self.toggle_view_frame, {}),
            # ##########################################################################################################
            ord('c'): (self.state_toggle_classification, {}),
            ord('p'): (self.state_paint_mode, {}),
            # ##########################################################################################################
            ord('t'): (self.state_run_training, {}),
            ord('k'): (self.toggle_kernel_selection, {}),
            ord('R'): (self.state_reset_classifications, {}),
            ord('s'): (self.state_save_classifications, {}),
            ord('l'): (self.state_load_classifications, {}),
            ord('B'): (self.state_toggle_blanking, {}),
            ord('b'): (self.cycle_blanking, {}),
            ord('b')-ord('`'): (self.cycle_block_mode, {}),     # ^b
            ord('r')-ord('`'): (self.remember_class, {}),       # ^r
            ord('j'): (self.classifications_generate, {}),
            ord('0'): (self.state_select_classifications, {'classification': 0}),
            ord('1'): (self.state_select_classifications, {'classification': 1}),
            ord('2'): (self.state_select_classifications, {'classification': 2}),
            ord('3'): (self.state_select_classifications, {'classification': 3}),
            ord('4'): (self.state_select_classifications, {'classification': 4}),
            ord('5'): (self.state_select_classifications, {'classification': 5}),
            ord('6'): (self.state_select_classifications, {'classification': 6}),
            ord('7'): (self.state_select_classifications, {'classification': 7}),
            ord('8'): (self.state_select_classifications, {'classification': 8}),
            ord('9'): (self.state_select_classifications, {'classification': 9}),
        }
        # ##############################################################################################################
        self.grid_enabled = True
        self.grid_rendered = True
        # ##############################################################################################################
        self.classify = False
        self.paint_mode = False
        self.re_offset = []
        self.classification = Classifications.IS_CLASS_1
        self.hidden_classes_set = []
        self.remembered_classes_set = []
        self.blank = False
        self.block_mode = False
        # ##############################################################################################################
        self.selecting_kernel = False
        # ##############################################################################################################
        self.classify_session = []
        # ##############################################################################################################
        self.class_colors = [
            cc.BLACK,       # 0
            cc.BROWN,       # 1
            cc.OLIVE,       # 2
            cc.TEAL,        # 3
            cc.NAVY,        # 4
            cc.MAROON,      # 5
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
            if event == cv2.EVENT_LBUTTONUP:
                self.validate_class_selection(x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.paint_mode is True:
                self.validate_continues_class_selection(x, y)
    # ##################################################################################################################

    @staticmethod
    def quit_application():
        return False

    def toggle_show_grid(self):
        self.grid_enabled = not self.grid_enabled

    def toggle_render_grid_content(self):
        self.grid_rendered = not self.grid_rendered

    def toggle_kernel_selection(self):
        self.selecting_kernel = not self.selecting_kernel
        print('MODE: Kernel selection {}'.format('Enabled' if self.selecting_kernel else 'Disabled'))

    def toggle_view_frame(self):
        self.video_stream.toggle_view()

    def state_toggle_classification(self):
        self.classify = not self.classify
        if not self.classify:
            print()
        print('MODE: Classification {}'.format('Enabled' if self.classify else 'Disabled'))
        if self.classify:
            self.print_classifying(self.classification)
        self.re_offset.clear()

    def state_paint_mode(self):
        self.paint_mode = not self.paint_mode
        print('MODE: Paint Mode {}'.format('Enabled' if self.paint_mode else 'Disabled'))

    def state_reset_classifications(self):
        self.training_set.clear()
        print('Training data has been cleared')

    def state_save_classifications(self):
        print('MODE: Saving classifications...', end='')
        self.training_set.save()
        print('done')

    def state_load_classifications(self):
        print('MODE: Loading classifications...', end='')
        self.training_set.load()
        print('done')

    # ##################################################################################################################
    def validate_class_selection(self, x, y):
        offset, xy, sub_frame = self.rectangle_stream.sub_frame_from_xy(self.video_stream.color_frame(), x, y)
        print('color at x={}, y={}: {}'.format(x, y, self.video_stream.pick_color(x, y)))
        if sub_frame is not None and offset not in self.re_offset:
            self.push_class_selection(sub_frame, self.video_stream.frame_counter)
            self.push_classification(self.video_stream.frame_counter, xy)
            self.re_offset.append(offset)

    def validate_continues_class_selection(self, x, y):
        offset, xy, sub_frame = self.rectangle_stream.sub_frame_from_xy(self.video_stream.color_frame(), x, y)
        if sub_frame is not None and offset not in self.re_offset:
            self.push_class_selection(sub_frame, self.video_stream.frame_counter)
            self.push_classification(self.video_stream.frame_counter, xy)
            self.re_offset.append(offset)

    def push_class_selection(self, sub_frame, frame_counter):
        self.training_set.push(
            frame_counter, *self.classifier.prepare(self.classification, sub_frame, dropout=False)
        )
        if self.selecting_kernel:
            self.video_stream.kernel = sub_frame

    def push_classification(self, frame_counter, xy):
        self.classification_set.push(frame_counter, xy, self.classification)

    def classifications_generate(self):
        classes = self.supported_classes
        class_sets = 2
        # s, t = 1, 227
        s, t = 0, 255
        slice = t/classes
        for c in range(1, classes+1):
            for r in range(class_sets):
                _s, _t = int(s+(slice*c)), int(s+(slice*(c+1)))
                sub_frame = np.random.random_integers(_s, _t, self.rectangle_stream.shape)
                self.training_set.push(0, c, self.classifier.prepare(sub_frame))

    # ##################################################################################################################
    def state_select_classifications(self, classification):
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
        else:
            self.remembered_classes_set.remove(self.classification)
        print(self.remembered_classes_set)

    def print_classifying(self, classification):
        color_name = list(cc.colors.keys())[list(cc.colors.values()).index(self.class_colors[classification])]
        print('CLASSIFYING: {} <{}>'.format(Classifications.name(classification), color_name))

    def print_excluding(self, classification, exclude):
        color_name = list(cc.colors.keys())[list(cc.colors.values()).index(self.class_colors[classification])]
        state_string = 'EXCLUDING' if exclude else 'INCLUDING'
        print('{}: {} <{}>'.format(state_string, Classifications.name(classification), color_name))

    # ##################################################################################################################
    def state_run_training(self):
        print('MODE: Training')
        self.train_classifier()
        print('Training done')
        self.classify = False

    def train_classifier_nearest_neighbor(self):
        train_X = np.zeros(shape=(len(self.training_set), self.rectangle_stream.area))
        train_y = np.ones(shape=(len(self.training_set)))
        keys, values, labels = self.training_set.flatten()
        for i, value in enumerate(values):
            train_X[i, :] = value
            train_y[i] = keys[i]
        self.classifier.train(train_X, train_y)

    def train_classifier_neural_network(self):
        display(self.supported_classes)
        train_y, train_X, labels = self.training_set.flatten(encoded_classes=self.supported_classes)
        self.classifier.train(train_X, train_y, labels, epochs=self.epochs, break_out=self.accruracy)

    def train_classifier_mxnet(self):
        display(self.supported_classes)
        train_y, train_X, labels = self.training_set.flatten()
        dataset = [
            data for data in zip(train_X, labels)
        ]
        self.classifier.train(dataset)

    def query_classifier_nearest_neighbor(self):
        X = np.zeros(shape=(len(self.rectangle_stream), self.rectangle_stream.area))
        for i, rectangle in enumerate(self.rectangle_stream):
            x_l, x_r, y_t, y_b = rectangle[0], rectangle[2], rectangle[1], rectangle[3]
            frame = self.video_stream.gray_frame()[y_t:y_b, x_l:x_r]
            X[i, :] = self.classifier.prepare(frame)
        return self.classifier.predict(X)

    def query_classifier_neural_network(self):
        predictions = []
        for i, rectangle in enumerate(self.rectangle_stream):
            x_l, x_r, y_t, y_b = rectangle[0], rectangle[2], rectangle[1], rectangle[3]
            frame = self.video_stream.gray_frame()[y_t:y_b, x_l:x_r]
            Y = self.classifier.predict(self.classifier.prepare(frame))
            print(Y)
            l = np.argmax(Y)
            if Y[l] > self.accruracy:
                predictions.append(l+1)
            else:
                predictions.append(Classifications.IS_NAC)
        return predictions

    def query_classifier_mxnet(self):
        predictions = []
        batch = []
        for i, rectangle in enumerate(self.rectangle_stream):
            x_l, x_r, y_t, y_b = rectangle[0], rectangle[2], rectangle[1], rectangle[3]
            frame = self.video_stream.color_frame()[y_t:y_b, x_l:x_r, :]
            label, data = self.classifier.prepare(0, frame)
            batch.append((data, label))
        Y = self.classifier.predict(batch)
        for y in Y:
            i = np.argmax(y)
            predictions.append(i)
        return predictions

    # ##################################################################################################################
    def show_predictions(self, predictions):
        for i, rectangle in enumerate(self.rectangle_stream):
            prediction = int(predictions[i])
            if prediction in self.hidden_classes_set:
                if self.blank:
                    tl, br = (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3])
                    cv2.rectangle(self.video_stream.view_frame(), tl, br, cc.YELLOW1.BGR(), -1)
            else:
                x, y = self.rectangle_stream.center(rectangle)
                class_color = self.class_colors[prediction]
                if self.block_mode:
                    tl, br = (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3])
                    cv2.rectangle(self.video_stream.view_frame(), tl, br, class_color.BGR(), -1)
                elif prediction > Classifications.IS_NAC:
                    point = (x, y)
                    radius = 3
                    cv2.circle(self.video_stream.view_frame(), point, radius, class_color.BGR(), thickness=-1)

    def class_aoi(self, predictions, classes):
        samples = 0
        aoi_l = []
        aoi_r = []
        lxy, rxy = None, None
        if len(self.rectangle_stream) > 0:
            for i, rectangle in enumerate(self.rectangle_stream):
                xy = self.rectangle_stream.center(rectangle)
                allowed_class = predictions[i] in classes
                if lxy is None:
                    if allowed_class:
                        lxy = xy
                        rxy = lxy
                else:
                    if lxy[1] != xy[1]:
                        aoi_l.append(lxy)
                        aoi_r.append(rxy)
                        lxy, rxy = None, None
                        if allowed_class:
                            lxy = xy
                            rxy = lxy
                        continue
                    if allowed_class:
                        rxy = self.rectangle_stream.center(rectangle)
            if lxy is not None:
                aoi_l.append(lxy)
                aoi_r.append(rxy)
            samples = len(aoi_l)+len(aoi_r)
        return samples, aoi_l, aoi_r

    def show_grid(self):
        if self.grid_enabled:
            self.rectangle_stream.render_on(self.video_stream.view_frame())

    def show_classification_selection(self):
        keys, values, labels = self.classification_set.flatten(self.video_stream.frame_counter)
        for i, point in enumerate(labels):
            classification = values[i]
            radius = 5
            class_color = self.class_colors[classification]
            cv2.circle(self.video_stream.view_frame(), point, radius, class_color.BGR(), thickness=-1)

    # ##################################################################################################################
    def show(self):
        self.show_grid()
        if self.classify:
            self.show_classification_selection()
        elif len(self.classifier) > 0:
            predictions = self.query_classifier()
            if self.grid_rendered:
                self.show_predictions(predictions)
            # self.driving_aoi.render(self.video_stream.view_frame())
            samples, aoi_l, aoi_r = self.class_aoi(predictions, self.remembered_classes_set)
            if samples > 0:
                self.tracking_aoi.reshape(aoi_l, aoi_r)
                self.tracking_aoi.render(self.video_stream.view_frame())

    def save(self):
        pass

    def load(self):
        pass
