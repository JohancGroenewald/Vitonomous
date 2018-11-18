import argparse
import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn

# Parse CLI arguments

parser = argparse.ArgumentParser(description='MXNet Gluon MNIST Example')
parser.add_argument('--batch-size', type=int, default=5,
                    help='batch size for training and testing (default: 100)')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Train on GPU with CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
opt = parser.parse_args()


class MxnetClassifier:
    def __init__(self, inputs, classes):
        # define network
        # 'relu', 'sigmoid', 'softrelu', 'softsign', 'tanh'
        # inputs *= 3
        inputs = 512
        self.net = nn.Sequential()
        with self.net.name_scope():
            self.net.add(gluon.nn.Conv2D(channels=5, kernel_size=2))
            self.net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
            self.net.add(gluon.nn.Activation(activation='relu'))
            self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

            self.net.add(gluon.nn.Conv2D(channels=10, kernel_size=2))
            self.net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
            self.net.add(gluon.nn.Activation(activation='relu'))
            self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

            self.net.add(gluon.nn.Flatten())

            self.net.add(nn.Dense(inputs))
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

                if i % opt.log_interval == 0 and i > 0:
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


# if __name__ == '__main__':
#     if opt.cuda:
#         ctx = mx.gpu(0)
#     else:
#         ctx = mx.cpu()
#     train(opt.epochs, ctx)
