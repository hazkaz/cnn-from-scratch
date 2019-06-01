import mnist
import numpy as np

from Softmax import Softmax
from Conv import Conv3x3
from MaxPool import Pool2x2

# The mnist package handles the MNIST dataset for us!
# Learn more at https://github.com/datapythonista/mnist
train_images = mnist.train_images()[:1000]
train_labels = mnist.train_labels()[:1000]

conv = Conv3x3(8)
pool = Pool2x2()
softmax = Softmax(13 * 13 * 8, 10)


def forward(image, label):
    output = conv.forward((image / 255) - 0.5)
    output = pool.max_pool(output)
    output = softmax.forward(output)

    loss = -np.log(output[label])
    acc = 1 if np.argmax(output) == label else 0

    return output, loss, acc


def main():
    print('MNIST CNN initialized')

    loss = 0
    num_correct = 0

    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        _, l, acc = forward(im, label)
        # print(l,acc)
        loss += l
        num_correct += acc

        if i % 100 == 99:
            print(
                '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                (i + 1, loss / 100, num_correct)
            )
            loss = 0
            num_correct = 0


if __name__=="__main__":
    main()
# output = conv.forward(train_images[0])
# print(output.shape) # (26, 26, 8)
# pool = Pool2x2()
# output = pool.max_pool(output)
# print(output)
# print(output.shape)
