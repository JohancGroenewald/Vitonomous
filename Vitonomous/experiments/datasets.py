import pickle


class TrainingSet(object):
    FILE_NAME = 'training_set.pickle'

    def __init__(self):
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

    def flatten(self):
        keys = []
        values = []
        for index in self.database:
            for key in self.database[index]:
                for value in self.database[index][key]:
                    keys.append(key)
                    values.append(value)
        return keys, values

    def save(self):
        with open(self.FILE_NAME,'wb') as outfile:
            pickle.dump(self.database, outfile)

    def load(self):
        with open(self.FILE_NAME,'rb') as infile:
            self.database = pickle.load(infile)
