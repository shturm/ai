# https://github.com/llSourcell/neural_networks/blob/master/simple_af_network.ipynb


import numpy as np
"""
x is input, h is hidden, o is output
--
X H 
X H O
X H
  H
"""
input_data = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
output_labels = np.array([[0],
            [1],
            [1],
            [0]])

np.random.seed(5)

print(input_data)
print(output_labels)

# sigmoid function
def activate(x,deriv=False):
    if(deriv==True): return x*(1-x)
    return 1/(1+np.exp(-x))

# weights, w0 3,4 means 3 input neurons (comes from dim1 size of input_data) should map to 4 neurons in the next layer. 
# rows in w0 are  weights for a layer0 neuron to each neuron in layer1
# columns in w0 are weights of a layer0 neuron to each neuron in layer1
weights0 = 2*np.random.random((3,4)) - 1
weights1 = 2*np.random.random((4,1)) - 1

print(weights0)
print(weights1)

# train over the whole dataset 60k times = 60k epochs
for j in range(60000):

	# Forward propagate through layers 0, 1, and 2
    layer0 = input_data
    layer1 = activate(np.dot(layer0,weights0))
    layer2 = activate(np.dot(layer1,weights1))

    #calculate error for layer 2
    layer2_error = output_labels - layer2
    
    #Use it to compute the gradient
    layer2_gradient = layer2_error*activate(layer2,deriv=True)

    #calculate error for layer 1
    layer1_error = layer2_gradient.dot(weights1.T)
    
    #Use it to compute its gradient
    layer1_gradient = layer1_error * activate(layer1,deriv=True)
    
    #update the weights using the gradients
    weights1 += layer1.T.dot(layer2_gradient)
    weights0 += layer0.T.dot(layer1_gradient)
    
    if (j% 10000) == 0: print ("Error:" + str(np.mean(np.abs(layer2_error))))


# predict
for i in range(len(input_data)):
    layer1 = activate(np.dot(np.array(input_data[i]), weights0))
    layer2 = activate(np.dot(layer1,weights1))
    print("Predict " + str(input_data[i]) +" => "+ str(layer2)+" Expected "+str(output_labels[i])) # what is syn0 ?

