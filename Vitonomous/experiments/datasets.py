
class DataSet(object):
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
