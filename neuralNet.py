import numpy as np
import nnfs
from nnfs.datasets import spiral_data

np.random.seed(0)



class LayerDesign:
    def __init__(self,nSamples,nNeurons):
        self.weights = 0.1 * np.random.randn(nSamples,nNeurons)
        self.biases = np.zeros((1, nSamples))
        
    def forward(self, inputs):
        self.output = np.dot(inputs,self.weights.T) + self.biases

class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class Acitvation_SoftMax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sampleLosses = self.forward(output,y)
        data_loss = np.mean(sampleLosses)
        return data_loss
    
class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_cliped = np.clip(y_pred,1e-7,1-1e-7)
        

        if len(y_true.shape) ==1 :
            correct_confidences = y_pred_cliped[range(samples), y_true]
        elif len(y_true.shape) ==2 :
            correct_confidences = np.sum(y_pred_cliped*y_true, axis=1) # Elementwise multiplication 

        neg_log_likelihood = -np.log(correct_confidences)

        return neg_log_likelihood
    


'''
Upto Relu
'''
# #generate spiral data with 100 data points per class with 3 classes.
# X, y = spiral_data(100,3)

# layer1 = LayerDesign(5,2)
# activation1 = Activation_ReLU()

# layer1.forward(X)
# activation1.forward(layer1.output)
# print(layer1.output)
# print("*************")
# print(activation1.output)

'''
Upto SoftMax
'''
X, y = spiral_data(samples = 100, classes =3)
dense1 = LayerDesign(3,2)
activation1 = Activation_ReLU()

dense2 = LayerDesign(3,3)
activation2 = Acitvation_SoftMax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)


lossFunction = CategoricalCrossEntropy()
loss = lossFunction.calculate(activation2.output,y)

print(loss)

# Now that we have found the losses, to minimize this loss 
# is where the optimization part comes in.