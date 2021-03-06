import cv2
import numpy as np

from engines import VideoStream, RectangleStream, WindowStream, AreaOfInterest
from datasets import TrainingSet2 as TrainingSet
from classifiers import Classifications, NearestNeighbor, NeuralNetwork
from support import display
from classifiers_mxnet import MXNetClassifier

import color_constants as cc


# noinspection PyAttributeOutsideInit,PyArgumentList
class StateManager(object):
    def __init__(self, video_stream: VideoStream, rectangle_stream: RectangleStream, window_stream: WindowStream):
        self.video_stream = video_stream
        self.rectangle_stream = rectangle_stream
        self.window_stream = window_stream
        # ##############################################################################################################
        self.supported_classes = Classifications.IS_CLASS_20
        # CLASS 1: Environment
        # CLASS 2: Path
        # CLASS 3: Gravel
        # CLASS 4:
        # CLASS 5:
        # CLASS 6:
        # CLASS 7:
        # CLASS 8:
        # ##############################################################################################################
        self.classifier = MXNetClassifier(inputs=self.rectangle_stream.area, classes=self.supported_classes)
        self.train_classifier = self.train_classifier_mxnet
        self.query_classifier = self.query_classifier_mxnet
        self.frame_sourse = None
        self.source_in_color = False
        self.block = self.block_3
        # ##############################################################################################################
        self.training_set = TrainingSet()
        self.tracking_aoi = AreaOfInterest(0.1, 0.1, 0.5, self.video_stream.wh(), cc.YELLOW1, 2)
        self.driving_aoi = AreaOfInterest(0.5, 0.7, 0.5, self.video_stream.wh(), cc.BLUE, -1)
        # ##############################################################################################################
        self.window_stream.attach_state_manager_callback(self.state_manager_callback)
        # ##############################################################################################################
        self.key_bindings = {
            ord('h'): ('[h] display_key_bindings', self.display_key_bindings, {}),
            ord('q'): ('[q] quit_application', self.quit_application, {}),
            ord('r'): ('[r] video_stream.toggle_auto_grab', self.video_stream.toggle_auto_grab, {}),
            ord('R'): ('[R] restart_video_stream', self.restart_video_stream, {}),
            ord(','): ('[,] video_stream.toggle_grab, -1', self.video_stream.toggle_grab, {'direction': -1}),
            ord('.'): ('[.] video_stream.toggle_grab,  1', self.video_stream.toggle_grab, {'direction':  1}),
            ord('='): ('[=] rectangle_stream.select, rows,  1', self.rectangle_stream.select, {'rows':  1}),
            ord('-'): ('[-] rectangle_stream.select, rows, -1', self.rectangle_stream.select, {'rows': -1}),
            ord('+'): ('[+] rectangle_stream.select, margin,  1', self.rectangle_stream.select, {'margin': -1}),
            ord('_'): ('[_] rectangle_stream.select, margin, -1', self.rectangle_stream.select, {'margin':  1}),
            2490368: ('[up] rectangle_stream.select, rows,  1, locked', self.rectangle_stream.select, {'rows':  1, 'locked': True}),
            2621440: ('[down] rectangle_stream.select, rows, -1, locked', self.rectangle_stream.select, {'rows': -1, 'locked': True}),
            2424832: ('[right] rectangle_stream.select, margin,  1, locked', self.rectangle_stream.select, {'margin':  1, 'locked': True}),
            2555904: ('[left] rectangle_stream.select, margin, -1, locked', self.rectangle_stream.select, {'margin': -1, 'locked': True}),
            # ##########################################################################################################
            ord('g'): ('[g] toggle_show_grid', self.toggle_show_grid, {}),
            ord('G'): ('[G] toggle_render_grid_content', self.toggle_render_grid_content, {}),
            ord('m')-ord('`'): ('^[m] toggle_view_frame', self.toggle_view_frame, {}),
            # ##########################################################################################################
            ord('c'): ('[c] state_toggle_classification', self.state_toggle_classification, {}),
            ord('c')-ord('`'): ('^[c] state_toggle_classification_override', self.state_toggle_classification_override, {}),
            ord('p'): ('[p] state_paint_mode', self.state_paint_mode, {}),
            ord('u'): ('[u] undo_classification', self.undo_classification, {}),
            # ##########################################################################################################
            ord('t')-ord('`'): ('^[t] training_agent', self.training_agent, {}),
            # ##########################################################################################################
            ord('t'): ('[t] training_model', self.training_model, {}),
            ord('k'): ('[k] toggle_kernel_selection', self.toggle_kernel_selection, {}),
            ord('K'): ('[k] toggle_kernel_query', self.toggle_kernel_query, {}),
            ord('C'): ('[C] state_reset_classifications', self.state_reset_classifications, {}),
            ord('s'): ('[s] state_save_classifications', self.state_save_classifications, {}),
            ord('l'): ('[l] state_load_classifications', self.state_load_classifications, {}),
            ord('B'): ('[B] state_toggle_blanking', self.state_toggle_blanking, {}),
            ord('b'): ('[b] cycle_blanking, direction,  1', self.cycle_blanking, {'direction':  1}),
            ord('n'): ('[n] cycle_blanking, direction, -1', self.cycle_blanking, {'direction': -1}),
            ord('b')-ord('`'): ('^[b] cycle_block_mode', self.cycle_block_mode, {}),
            ord('r')-ord('`'): ('^[r] remember_class', self.remember_class, {}),
            ord('j'): ('[j] classifications_generate', self.classifications_generate, {}),
            ord('0'): ('[0] state_select_classifications, classification, 0', self.state_select_classifications, {'classification': 0}),
            ord('1'): ('[1] state_select_classifications, classification, 1', self.state_select_classifications, {'classification': 1}),
            ord('2'): ('[2] state_select_classifications, classification, 2', self.state_select_classifications, {'classification': 2}),
            ord('3'): ('[3] state_select_classifications, classification, 3', self.state_select_classifications, {'classification': 3}),
            ord('4'): ('[4] state_select_classifications, classification, 4', self.state_select_classifications, {'classification': 4}),
            ord('5'): ('[5] state_select_classifications, classification, 5', self.state_select_classifications, {'classification': 5}),
            ord('6'): ('[6] state_select_classifications, classification, 6', self.state_select_classifications, {'classification': 6}),
            ord('7'): ('[7] state_select_classifications, classification, 7', self.state_select_classifications, {'classification': 7}),
            ord('8'): ('[8] state_select_classifications, classification, 8', self.state_select_classifications, {'classification': 8}),
            ord('9'): ('[9] state_select_classifications, classification, 9', self.state_select_classifications, {'classification': 9}),
        }
        # ##############################################################################################################
        self.grid_enabled = True
        self.grid_rendered = True
        # ##############################################################################################################
        self.classify = False
        self.classification_override = False
        self.paint_mode = False
        self.re_offset = []
        self.classification = Classifications.IS_CLASS_1
        self.hidden_classes_set = []
        self.remembered_classes_set = []
        self.blank = False
        self.block_mode = False
        self.re_block_mode = self.block_mode
        # ##############################################################################################################
        self.selecting_kernel = False
        self.querying_kernel = False
        # ##############################################################################################################
        self.kernel_minimum = 0
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
        self.re_frame_counter = 0
        # ##############################################################################################################

    def accept(self, key):
        if key == -1:
            pass
        elif key in self.key_bindings:
            description, method, kwargs = self.key_bindings[key]
            return method(**kwargs)
        else:
            print('key: {} {} {}'.format(key, bin(key), hex(key)))
        return True

    # ##################################################################################################################
    def state_manager_callback(self, event, x, y):
        if self.classify or self.selecting_kernel or self.querying_kernel:
            if event == cv2.EVENT_LBUTTONUP:
                self.validate_class_selection(x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.paint_mode is True:
                self.validate_class_selection(x, y)
    # ##################################################################################################################

    def display_key_bindings(self):
        for description, method, kwargs in self.key_bindings.values():
            print(description)

    @staticmethod
    def quit_application():
        return False

    def toggle_show_grid(self):
        self.grid_enabled = not self.grid_enabled

    def toggle_render_grid_content(self):
        self.grid_rendered = not self.grid_rendered

    def toggle_kernel_selection(self):
        self.selecting_kernel = not self.selecting_kernel
        self.video_stream.kernel = None
        print('MODE: Kernel selection {}'.format('Enabled' if self.selecting_kernel else 'Disabled'))

    def toggle_kernel_query(self):
        self.querying_kernel = not self.querying_kernel
        print('MODE: Kernel query {}'.format('Enabled' if self.selecting_kernel else 'Disabled'))

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

    def state_toggle_classification_override(self):
        self.classification_override = not self.classification_override

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

    def restart_video_stream(self):
        self.video_stream.restart()
        self.video_stream.toggle_grab()

    # ##################################################################################################################
    def validate_class_selection(self, x, y):
        offset, xy, sub_frame = self.rectangle_stream.sub_frame_from_xy(self.frame_sourse, x, y, self.source_in_color)
        if self.paint_mode is False:
            print('color at x={}, y={}: {}'.format(x, y, self.video_stream.pick_color(x, y)))
        if self.classify and sub_frame is not None and offset not in self.re_offset:
            self.push_class_selection(self.video_stream.frame_counter, offset, xy, sub_frame)
            self.re_offset.append(offset)
        if self.selecting_kernel:
            self.video_stream.kernel = (xy, sub_frame)
            cv2.imwrite('kernels/kernel_s.png', sub_frame.astype(np.uint8))
        if self.querying_kernel:
            mn, mx = np.min(sub_frame), np.max(sub_frame)
            dt = mx-mn
            topography = sub_frame-np.min(sub_frame)
            contour = topography.flat[1:] - topography.flat[:-1]
            # contour, counts = np.unique(topography, return_counts=True)
            # contour_deltas = contour[1:] - contour[:-1]
            print('sub_frame', sub_frame, 'len', len(sub_frame))
            print('topography', topography, 'len', len(topography))
            print('mn, mx, dt', mn, mx, dt)
            print('contour', contour, 'len', len(contour))
            # print('contour_deltas', contour_deltas, 'len', len(contour_deltas))
            # ##########################################################################################################
            # deltas_set, deltas_counts = np.unique(contour, return_counts=True)
            smoothness = contour[1:] - contour[:-1]
            nonzero = np.count_nonzero(smoothness)
            zero = len(smoothness) - nonzero
            m = 1-((nonzero - zero)/len(smoothness))
            print('smoothness', smoothness, 'm', m)
            # deltas = np.unique(contour_deltas)
            # contour_entropy = 0 #self.entropy(deltas_set, deltas_counts)
            bins = 256
            bincount = np.bincount(topography.flat, minlength=bins)
            binstretch = np.arange(1, 257)
            g = bincount * binstretch
            m = np.sum(g)
            # m = np.count_nonzero(bincount)
            # p = bincount / len(topography)
            # p = p[p > 0]
            # contour_entropy = -np.sum(p * np.log2(p), axis=0)
            # contour_entropy = np.round(contour_entropy, 2)

            # print('contour_deltas', contour_deltas, 'len', len(contour_deltas), 'entropy ==>', contour_entropy)
            # print('deltas_set', deltas_set, 'len', len(deltas_set))
            # print('deltas_counts', deltas_counts, 'len', len(deltas_counts))
            print('bincount', bincount, 'm', m)
            print('p', g, 'len', len(g))
            # print('p', p * np.log2(p), 'len', len(p * np.log2(p)))
            self.kernel_minimum = m


    def push_class_selection(self, frame_number, offset, xy, frame_data):
        if not self.source_in_color:
            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2BGR)
        classification, data = self.classifier.prepare(self.classification, frame_data, dropout=False)
        self.training_set.push(frame_number, classification, offset, xy, data)
        classification, data = self.classifier.prepare(self.classification, cv2.flip(frame_data, 0), dropout=False)
        self.training_set.push(frame_number, classification, offset, xy, data)
        classification, data = self.classifier.prepare(self.classification, cv2.flip(frame_data, 1), dropout=False)
        self.training_set.push(frame_number, classification, offset, xy, data)

    def undo_classification(self):
        self.training_set.pop(self.video_stream.frame_counter)

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

    def cycle_blanking(self, direction=1):
        classification = self.classification
        classification += direction
        if classification > self.supported_classes:
            classification = Classifications.IS_NAC
        elif classification < Classifications.IS_NAC:
            classification = self.supported_classes
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
    def training_agent(self):

        pass

    # ##################################################################################################################
    def training_model(self):
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
        training_data, training_labels = self.training_set.flatten()
        dataset = [
            data for data in zip(training_data, training_labels)
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
            if self.source_in_color:
                frame = self.frame_sourse[y_t:y_b, x_l:x_r, :]
            else:
                frame = self.frame_sourse[y_t:y_b, x_l:x_r]
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            label, data = self.classifier.prepare(0, frame)
            batch.append((data, label))
        Y = self.classifier.predict(batch)
        for y in Y:
            i = np.argmax(y)
            predictions.append(i)
        return predictions

    def query_classifier_motex(self):
        predictions = []
        for i, rectangle in enumerate(self.rectangle_stream):
            xl, xr, yt, yb = rectangle[0], rectangle[2], rectangle[1], rectangle[3]
            rect1 = self.video_stream.frame_store[0][yt:yb, xl:xr]
            rect2 = self.video_stream.frame_store[1][yt:yb, xl:xr]
            steps_ratio, minimum_ratio = self.array_shift_sum_2(rect1, rect2)
            classification = int(20 * minimum_ratio)
            predictions.append(classification)
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

    def class_aoi_circumscribed(self, predictions, classes):
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

    def class_aoi_inscribed(self, predictions, classes):
        samples = 0
        aoi_b = []
        aoi_t = []
        bxy, txy = None, None
        if len(self.rectangle_stream) > 0:
            s1, s2, s3, s4 = self.rectangle_stream.grid_shape
            columns = s4-s3
            rows = s2-s1
            grid = [(c, r) for c in range(s4-s3) for r in range(s2-s1)]
            state = 0
            xy = None
            for (c, r) in grid:
                if r == 0:
                    state = 0
                    if txy is not None:
                        aoi_t.append(txy)
                    xy = None
                    txy = None
                i = r*columns+c
                allowed_class = predictions[i] in classes
                if allowed_class:
                    rectangle = self.rectangle_stream.rectangles_flattened[i]
                    xy = self.rectangle_stream.center(rectangle)
                if allowed_class and r == 0:
                    state = 1
                    aoi_b.append(xy)
                if allowed_class and state == 1:
                    txy = xy
                if not allowed_class and state == 1:
                    state = 2
            if state in [1, 2] and txy is not None:
                aoi_t.append(txy)
            samples = len(aoi_b)+len(aoi_t)
        return samples, aoi_b, aoi_t

    def show_grid(self):
        if self.grid_enabled:
            self.rectangle_stream.render_on(self.video_stream.view_frame())

    def show_classification_selection(self):
        labels, points = self.training_set.flatten(
            frame_number=self.video_stream.frame_counter, select=('labels', 'xy')
        )
        radius = 5
        for i, point in enumerate(points):
            classification = int(labels[i])
            class_color = self.class_colors[classification]
            cv2.circle(self.video_stream.view_frame(), point, radius, class_color.BGR(), thickness=-1)

    # ##################################################################################################################
    def block_1(self):
        self.show_grid()
        if self.classify:
            if self.classification_override:
                predictions = self.query_classifier()
                self.show_predictions(predictions)
            self.show_classification_selection()
        elif len(self.classifier) > 0:
            predictions = self.query_classifier()
            if self.grid_rendered:
                self.show_predictions(predictions)
            samples, aoi_l, aoi_r = self.class_aoi_circumscribed(predictions, self.remembered_classes_set)
            if samples > 0:
                self.tracking_aoi.reshape(aoi_l, aoi_r)
                self.tracking_aoi.render(self.video_stream.view_frame())
            samples, aoi_l, aoi_r = self.class_aoi_inscribed(predictions, self.remembered_classes_set)
            if samples > 0:
                self.driving_aoi.reshape(aoi_l, aoi_r)
                self.driving_aoi.render(self.video_stream.view_frame(), fill=True)

    def array_shift_sum_1(self, rect1, rect2):
        h, w = rect1.shape
        s = 6
        f1 = np.zeros((h, w+s))
        f1[:,3:w+3] = rect1
        f2 = np.zeros((h, w+s))
        f2[:,6:w+6] = rect2
        source_balance = np.sum(rect1)
        local_minimum = np.sum(cv2.absdiff(f1, f2))
        local_steps = 0
        for i in range(s):
            diff = cv2.absdiff(f1, f2)
            diff_sum = np.sum(diff)
            if diff_sum < local_minimum:
                local_steps = i
                local_minimum = diff_sum
            f2 = np.roll(f2, -1)
        minimum_ratio = (local_minimum if local_minimum < source_balance else source_balance)/source_balance
        step_ratio = local_steps / s
        return step_ratio, minimum_ratio

    def array_shift_sum_2(self, rect1, rect2):
        source_balance = np.sum(rect1)
        # local_minimum = np.sum(cv2.absdiff(rect1, rect2))
        # local_minimum = np.sum(cv2.bitwise_and(rect1, rect2))
        local_minimum = np.sum(((rect1 - rect2) + 255)//2)
        local_steps = 0
        local_ratio = (local_minimum if local_minimum < source_balance else source_balance)/source_balance
        return local_steps, local_ratio

    def block_2(self):
        self.show_grid()
        if self.re_frame_counter == self.video_stream.frame_counter:
            return
        self.re_frame_counter = self.video_stream.frame_counter
        predictions = self.query_classifier_motex()
        if self.grid_rendered:
            self.show_predictions(predictions)
            samples, aoi_l, aoi_r = self.class_aoi_circumscribed(predictions, self.remembered_classes_set)
            if samples > 0:
                self.tracking_aoi.reshape(aoi_l, aoi_r)
                self.tracking_aoi.render(self.video_stream.view_frame())

    def new_frame_query(self):
        for i, rectangle in enumerate(self.rectangle_stream):
            x, y = self.rectangle_stream.center(rectangle)
            offset, xy, frame = self.rectangle_stream.sub_frame_from_xy(self.frame_sourse, x, y, self.source_in_color)
            mn, mx = np.min(frame), np.max(frame)
            dt = mx-mn
            topography = frame-np.min(frame)
            # contour, counts = np.unique(topography, return_counts=True)
            # contour_deltas = contour[1:] - contour[:-1]
            # smoothness = contour_deltas[1:] - contour_deltas[:-1]
            # nonzero = np.count_nonzero(smoothness)
            # zero = len(smoothness) - nonzero
            # m = zero - nonzero
            contour = topography.flat[1:] - topography.flat[:-1]
            smoothness = contour[1:] - contour[:-1]
            nonzero = np.count_nonzero(smoothness)
            zero = len(smoothness) - nonzero
            m = 1-((nonzero - zero)/len(smoothness))

            bins = 256
            bincount = np.bincount(topography.flat, minlength=bins)
            binstretch = np.arange(1, 257)
            g = bincount * binstretch
            m = np.sum(g)

            # bins = np.max(contour_deltas)
            # bincount = np.bincount(contour_deltas, minlength=bins)
            # p = bincount / len(contour_deltas)
            # p = p[p > 0]
            # contour_entropy = -np.sum(p * np.log2(p), axis=0)
            # contour_entropy = np.round(contour_entropy, 2)

            if m <= self.kernel_minimum:
                class_color = self.class_colors[-1]
            else:
                class_color = self.class_colors[0]
            radius = 3
            cv2.circle(self.video_stream.view_frame(), (x, y), radius, class_color.BGR(), thickness=-1)

    def block_3(self):
        self.show_grid()
        if self.querying_kernel:
            self.new_frame_query()


    # ##################################################################################################################
    def show(self):
        if self.source_in_color:
            self.frame_sourse = self.video_stream.color_frame()
        else:
            self.frame_sourse = self.video_stream.gray_frame()
        self.block()

    def save(self):
        pass

    def load(self):
        pass
