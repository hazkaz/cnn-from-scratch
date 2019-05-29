import mnist
from conv import Conv3x3
from pool import Pool2x2
# The mnist package handles the MNIST dataset for us!
# Learn more at https://github.com/datapythonista/mnist
train_images = mnist.train_images()
train_labels = mnist.train_labels()

conv = Conv3x3(8)
output = conv.forward(train_images[0])
print(output.shape) # (26, 26, 8)
pool = Pool2x2()
output = pool.max_pool(output)
print(output.shape)