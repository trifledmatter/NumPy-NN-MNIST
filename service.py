import networks.base as net
import loaders.mnist as mnist

training_data, validation_data, test_data = mnist.load_data_wrapper()

net = net.Network([784, 30, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

net = net.Network([784, 0, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
