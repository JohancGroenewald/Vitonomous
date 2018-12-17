import pickle


class TrainingSet1(object):
    FILE_NAME = None

    def __init__(self):
        self.FILE_NAME = f'{self.__class__.__name__}.pickle'
        self.database = {}

    def __len__(self):
        lengths = []
        for dictionary in self.database.values():
            lengths.extend([
                len(values) for values in dictionary.values()
            ])
        return sum(lengths)

    def clear(self):
        self.database = {}

    def push(self, index, key, value):
        if index in self.database:
            if key in self.database[index]:
                self.database[index][key].append(value)
            else:
                self.database[index][key] = [value]
        else:
            self.database[index] = {key: [value]}

    def pop(self, index):
        raise NotImplementedError

    def flatten(self, index=None, encoded_classes=0):
        keys = []
        values = []
        labels = []
        for i in self.database:
            if index is not None and i != index:
                continue
            for key in self.database[i]:
                for value in self.database[i][key]:
                    if encoded_classes > 0:
                        encoded = [0.01]*encoded_classes
                        encoded[key-1] = 0.99
                        keys.append(encoded)
                    else:
                        keys.append(key)
                    values.append(value)
                    labels.append(key)
        return keys, values, labels

    def save(self):
        with open(self.FILE_NAME,'wb') as outfile:
            pickle.dump(self.database, outfile)

    def load(self):
        with open(self.FILE_NAME,'rb') as infile:
            self.database = pickle.load(infile)


class TrainingSet2:
    FILE_NAME = None

    def __init__(self):
        self.FILE_NAME = f'{self.__class__.__name__}.pickle'
        self.database = []

    def __len__(self):
        return len(self.database)

    def push(self, frame_number, classification, offset, xy, data):
        self.database.append((frame_number, classification, offset, xy, data))

    def pop(self, frame_number):
        if len(self.database) > 0:
            if self.database[-1][0] == frame_number:
                self.database.pop()

    def clear(self):
        self.database.clear()

    def flatten(self, frame_number=None, select=('data', 'labels')):
        classification, offset, xy, data = [], [], [], []
        if frame_number is None:
            for (_frame_number, _classification, _offset, _xy, _data) in self.database:
                classification.append(_classification)
                offset.append(_offset)
                xy.append(_xy)
                data.append(_data)
        else:
            for (_frame_number, _classification, _offset, _xy, _data) in self.database:
                if frame_number == _frame_number:
                    classification.append(_classification)
                    offset.append(_offset)
                    xy.append(_xy)
                    data.append(_data)
        return_values = []
        for selected in select:
            if selected == 'labels':
                return_values.append(classification)
            elif selected == 'offset':
                return_values.append(offset)
            elif selected == 'xy':
                return_values.append(xy)
            elif selected == 'data':
                return_values.append(data)
        return tuple(return_values)

    def save(self):
        with open(self.FILE_NAME,'wb') as outfile:
            pickle.dump(self.database, outfile)

    def load(self):
        with open(self.FILE_NAME,'rb') as infile:
            self.database = pickle.load(infile)
