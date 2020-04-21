import sys
from pathlib import Path
path = Path.cwd()
sys.path.append(str(path.parent))
import numpy as np
from NeuralNetwork import NeuralNet


train_x = np.array([np.array([0, 1]), np.array([1, 0]),
                    np.array([1, 1]), np.array([0, 0])]).T
test_x = np.array([np.array([1, 0])]).reshape((2, 1))
train_y = np.array([1, 1, 0, 0]).reshape((1, 4))
test_y = np.array([1]).reshape((1, 1))


layers_dims = [2, 3, 3, 1]  # 4-layer model
layer4Model = NeuralNet.NeuralNet(layers_dims)
layer4Model.hyperInit = 1
layer4Model.fit(train_x, train_y, num_iterations=2000, learning_rate=0.1,
                Lambda=0.0, print_cost=True, init="he")


print("Training set Accuracy")
pred_train = layer4Model.predict(train_x, train_y, show=True)
print("Testing set Accuracy")
pred_test = layer4Model.predict(test_x, test_y, show=True)
