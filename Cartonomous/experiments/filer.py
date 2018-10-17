import os
import numpy as np
import shutil

from sources import Sources


def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def shuffle_in_unison(a, b):
    # courtesy http://stackoverflow.com/users/190280/josh-bleecher-snyder
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def move_files(input, output):
    """
    Input: folder with dataset, where every class is in separate folder
    Output: all images, in format class_number.jpg; output path should be absolute
    """
    index = -1
    for root, dirs, files in os.walk(input):
        print('** ', root, dirs, files)
        path = root.split('\\')
        path[0] += '\\'
        print('Working with path ', path)
        print('Path index ', index)
        file_num = 0
        for file in files:
            file_name, file_extension = os.path.splitext(file)
            if file_extension == '.png' or file_extension == '.PNG':
                source = os.path.join(*path, file)
                if os.path.isfile(source):
                    file = '{}{}{}'.format(path[-1], str(file_num), file_extension)
                    destination = os.path.join(output, file)
                    print(destination)
                    shutil.copy(source, destination)
                file_num += 1
        index += 1


def create_text_file(input_path, out_path, percentage):
    """
    Creating train.txt and val.txt for feeding Caffe
    """

    images, labels = [], []
    os.chdir(input_path)

    for item in os.listdir('.'):
        # print(os.path.join('.', item))
        if not os.path.isfile(os.path.join('.', item)):
            continue
        try:
            label = int(item.split('_')[0])
            images.append(item)
            labels.append(label)
        except:
            continue

    images = np.array(images)
    labels = np.array(labels)
    images, labels = shuffle_in_unison(images, labels)

    X_train = images[0:int(len(images) * percentage)]
    y_train = labels[0:int(len(labels) * percentage)]

    X_test = images[int(len(images) * percentage):]
    y_test = labels[int(len(labels) * percentage):]

    os.chdir(out_path)

    train_file = open("train.txt", "w")
    for i, l in zip(X_train, y_train):
        train_file.write(i + " " + str(l) + "\n")

    testfile = open("val.txt", "w")
    for i, l in zip(X_test, y_test):
        testfile.write(i + " " + str(l) + "\n")

    train_file.close()
    testfile.close()


def remove_previous_training_data(out_path):
    file_list = os.listdir(out_path)
    for f in file_list:
        file_path = os.path.join(out_path, f)
        if os.path.isfile(file_path):
            os.remove(file_path)


def main():
    source_path = os.path.join(Sources.ROOT, Sources.SOURCE, Sources.CLASSES)
    source_path = source_path.replace('/', chr(92))
    destination_path = os.path.join(Sources.ROOT, Sources.SOURCE, Sources.TRAIN)
    destination_path = destination_path.replace('/', chr(92))
    remove_previous_training_data(destination_path)
    move_files(source_path, destination_path)
    create_text_file(destination_path, './', 0.85)


main()
