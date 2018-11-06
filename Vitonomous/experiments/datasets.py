import pickle


class DataSet(object):
    FILE_NAME = 'data_set.pickle'

    def __init__(self):
        self.database = {}

    def __len__(self):
        length = sum([len(values) for values in self.database.values()])
        return length

    def clear(self):
        self.database = {}

    def push(self, key, value):
        if key in self.database:
            self.database[key].append(value)
        else:
            self.database[key] = [value]

    def flatten(self):
        keys = []
        values = []
        for key in self.database:
            for value in self.database[key]:
                keys.append(key)
                values.append(value)
        return keys, values

    def save(self):
        with open(self.FILE_NAME,'wb') as outfile:
            pickle.dump(self.database, outfile)

    def load(self):
        with open(self.FILE_NAME,'rb') as infile:
            self.database = pickle.load(infile)
