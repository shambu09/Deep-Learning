from NeuNet.NeuralNetwork import NeuralNet, HelperFuncs
import pathlib


train_x_orig, train_y, test_x_orig, test_y, classes = HelperFuncs.load_data(
    pathlib.Path.cwd())

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]
# The "-1" makes reshape flatten the remaining dimensions


train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.


layers_dims = [12288, 20, 10, 5, 5, 5, 1]  # 4-layer model
layer4Model = NeuralNet.NeuralNet(layers_dims)
layer4Model.hyperInit = 1
layer4Model.fit(train_x, train_y, num_iterations=4000, learning_rate=0.005,
                Lambda=1.0, print_cost=True, init="he")


print("Training set Accuracy")
pred_train = layer4Model.predict(train_x, train_y)
print("Testing set Accuracy")
pred_test = layer4Model.predict(test_x, test_y)


"""
{                                       {
# 3000                                   #3000
lambda = 0.1 init= "he"                 lambda = 0.1 , init="random"
Training set Accuracy                   Training set Accuracy
Accuracy: 0.8468899521531099            Accuracy: 0.6555023923444976
Testing set Accuracy                    Testing set Accuracy
Accuracy: 0.5800000000000001            Accuracy: 0.3400000000000001
}                                       }



{                                       {
# 3100                                   #6000
lambda = 0.4 init= "he"                 lambda = 0.6 , init="he"
Training set Accuracy                   Training set Accuracy
Accuracy: 0.985645                      Accuracy: 0.990430
Testing set Accuracy                    Testing set Accuracy
Accuracy: 0.84                          Accuracy: 0.86
}                                       }
"""
