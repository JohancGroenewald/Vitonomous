import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn
# noinspection PyUnresolvedReferences
from mxnet.contrib.ndarray import MultiBoxPrior


class MXNetClassifier:
    BATCH_SIZE = 5
    EPOCHS = 500
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    CUDA = False
    LOG_INTERVAL = 100

    def __init__(self, inputs, classes):
        # define network
        # 'relu', 'sigmoid', 'softrelu', 'softsign', 'tanh'
        inputs *= 3
        dense_layer_inputs = 512
        self.net = nn.Sequential()
        with self.net.name_scope():
            self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            self.net.add(gluon.nn.Conv2D(channels=5, kernel_size=2))
            self.net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
            self.net.add(gluon.nn.Activation(activation='relu'))
            self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

            # self.net.add(gluon.nn.Conv2D(channels=10, kernel_size=2))
            # self.net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
            # self.net.add(gluon.nn.Activation(activation='relu'))
            # self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

            self.net.add(gluon.nn.Flatten())

            self.net.add(nn.Dense(dense_layer_inputs))
            self.net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
            self.net.add(gluon.nn.Activation(activation='relu'))

            self.net.add(nn.Dense(classes))
        self.len = 0
        self.dropout = np.random.rand(inputs)
        self.keep_prob = 0.1
        self.dropout = self.dropout < self.keep_prob

    def __len__(self):
        return self.len

    # data
    def prepare(self, label, data, dropout=False):
        # data = data.ravel().astype(np.float32)/255
        data = np.transpose(data.astype(np.float32), (2, 0, 1))/255
        label = float(label)
        if dropout:
            data = np.multiply(data, self.dropout)
        return label, data

    # train_data = gluon.data.DataLoader(
    #     gluon.data.vision.MNIST('./data', train=True, transform=transformer),
    #     batch_size=opt.batch_size, shuffle=True, last_batch='discard')
    #
    # val_data = gluon.data.DataLoader(
    #     gluon.data.vision.MNIST('./data', train=False, transform=transformer),
    #     batch_size=opt.batch_size, shuffle=False)

    # train
    def test(self, val_data, ctx=mx.cpu()):
        metric = mx.metric.Accuracy()
        for data, label in val_data:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            output = self.net(data)
            metric.update([label], [output])

        return metric.get()

    def train(self, dataset, epochs=10000, ctx=mx.cpu()):
        train_data = gluon.data.DataLoader(dataset, batch_size=150, last_batch='keep', shuffle=True)
        self.net.initialize(mx.init.Xavier(magnitude=2.5), ctx=ctx)
        trainer = gluon.Trainer(
            # self.net.collect_params(), 'adadelta', {'rho': 0.9, 'epsilon': 1e-05}
            # self.net.collect_params(), 'adagrad', {'eps': 1e-07}
            self.net.collect_params(), 'adam', {
                'learning_rate': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-08, 'lazy_update': True
            }
            # self.net.collect_params(), 'signum', {'learning_rate': 0.3, 'momentum': 0.5, 'wd_lh': 0.0}
        )
        metric = mx.metric.Accuracy()
        loss = gluon.loss.SoftmaxCrossEntropyLoss()

        re_acc = 0
        acc_stable = 0
        for epoch in range(epochs):
            # reset data iterator and metric at begining of epoch.
            metric.reset()
            for i, (data, label) in enumerate(train_data):
                # Copy data to ctx if necessary
                data = data.as_in_context(ctx)
                label = label.as_in_context(ctx)
                # Start recording computation graph with record() section.
                # Recorded graphs can then be differentiated with backward.
                with autograd.record():
                    output = self.net(data)
                    L = loss(output, label)
                    L.backward()
                # take a gradient step with batch_size equal to data.shape[0]
                trainer.step(data.shape[0])
                # update metric at last.
                metric.update([label], [output])

                if i % self.LOG_INTERVAL == 0 and i > 0:
                    name, acc = metric.get()
                    print('[Epoch %d Batch %d] Training: %s=%f'%(epoch, i, name, acc))

            name, acc = metric.get()
            if abs(acc - re_acc) <= 0.05:
                acc_stable += 1
            else:
                acc_stable = 0
            print('[Epoch %d] Training: %s=%f, acc_stable:%d'%(epoch, name, acc, acc_stable))
            if acc_stable > 100:
                print('[Epoch {}] Acc stable exit'.format(epoch))
                break
            re_acc = acc

            # name, val_acc = self.test(ctx)
            # print('[Epoch %d] Validation: %s=%f'%(epoch, name, val_acc))

        # self.net.save_parameters('mnist.params')
        self.len = 1

    # noinspection PyUnresolvedReferences
    def predict(self, dataset, ctx=mx.cpu()):
        pred_data = gluon.data.DataLoader(dataset, batch_size=200, last_batch='keep')
        predictions = []
        for data, label in pred_data:
            data = data.as_in_context(ctx)
            output = self.net(data)
            # predictions.extend(mx.nd.argmax(output, axis=1)[0])
            predictions.extend(output.asnumpy())
        return predictions


class MXNetSSD(gluon.Block):

    def __init__(self, num_classes, **kwargs):
        super(MXNetSSD, self).__init__(**kwargs)
        # anchor box sizes for 4 feature scales
        self.anchor_sizes = [[.2, .272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        # anchor box ratios for 4 feature scales
        self.anchor_ratios = [[1, 2, .5]] * 5
        self.num_classes = num_classes

        with self.name_scope():
            self.body, self.downsamples, self.class_preds, self.box_preds = self.ssd_model(4, num_classes)

    def forward(self, x):
        default_anchors, predicted_classes, predicted_boxes = self.ssd_forward(
            x, self.body, self.downsamples, self.class_preds, self.box_preds, self.anchor_sizes, self.anchor_ratios
        )
        # we want to concatenate anchors, class predictions, box predictions from different layers
        anchors = self.concat_predictions(default_anchors)
        box_preds = self.concat_predictions(predicted_boxes)
        class_preds = self.concat_predictions(predicted_classes)
        # it is better to have class predictions reshaped for softmax computation
        class_preds = nd.reshape(class_preds, shape=(0, -1, self.num_classes + 1))

        return anchors, class_preds, box_preds

    @staticmethod
    def ssd_forward(x, body, downsamples, class_preds, box_preds, sizes, ratios):
        # extract feature with the body network
        x = body(x)

        # for each scale, add anchors, box and class predictions,
        # then compute the input to next scale
        default_anchors = []
        predicted_boxes = []
        predicted_classes = []

        for i in range(5):
            default_anchors.append(MultiBoxPrior(x, sizes=sizes[i], ratios=ratios[i]))
            predicted_boxes.append(MXNetSSD.flatten_prediction(box_preds[i](x)))
            predicted_classes.append(MXNetSSD.flatten_prediction(class_preds[i](x)))
            if i < 3:
                x = downsamples[i](x)
            elif i == 3:
                # simply use the pooling layer
                x = nd.Pooling(x, global_pool=True, pool_type='max', kernel=(4, 4))

        return default_anchors, predicted_classes, predicted_boxes

    @staticmethod
    def flatten_prediction(pred):
        return nd.flatten(nd.transpose(pred, axes=(0, 2, 3, 1)))

    @staticmethod
    def concat_predictions(preds):
        return nd.concat(*preds, dim=1)

    @staticmethod
    def class_predictor(num_anchors, num_classes):
        """return a layer to predict classes"""
        return nn.Conv2D(num_anchors * (num_classes + 1), 3, padding=1)

    @staticmethod
    def box_predictor(num_anchors):
        """return a layer to predict delta locations"""
        return nn.Conv2D(num_anchors * 4, 3, padding=1)

    @staticmethod
    def down_sample(num_filters):
        """stack two Conv-BatchNorm-Relu blocks and then a pooling layer to halve the feature size"""
        out = nn.HybridSequential()
        for _ in range(2):
            out.add(nn.Conv2D(num_filters, 3, strides=1, padding=1))
            out.add(nn.BatchNorm(in_channels=num_filters))
            out.add(nn.Activation('relu'))
        out.add(nn.MaxPool2D(2))
        return out

    @staticmethod
    def body():
        """return the body network"""
        out = nn.HybridSequential()
        for nfilters in [16, 32, 64]:
            out.add(MXNetSSD.down_sample(nfilters))
        return out

    @staticmethod
    def ssd_model(num_anchors, num_classes):
        """return SSD modules"""
        downsamples = nn.Sequential()
        class_preds = nn.Sequential()
        box_preds = nn.Sequential()

        downsamples.add(MXNetSSD.down_sample(128))
        downsamples.add(MXNetSSD.down_sample(128))
        downsamples.add(MXNetSSD.down_sample(128))

        for scale in range(5):
            class_preds.add(MXNetSSD.class_predictor(num_anchors, num_classes))
            box_preds.add(MXNetSSD.box_predictor(num_anchors))

        return MXNetSSD.body(), downsamples, class_preds, box_preds

