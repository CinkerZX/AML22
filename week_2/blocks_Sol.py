import numpy as np

#######################################################
# put `w1_sigmoid_forward` and `w1_sigmoid_grad_input` here #
#######################################################
def w1_sigmoid_forward(x_input):
    """sigmoid nonlinearity
    # Arguments
        x_input: np.array of size `(n_objects, n_in)`
    # Output
        the output of sigmoid layer
        np.array of size `(n_objects, n_in)`
    """
    #################
    ### YOUR CODE ###
    #################
    output = 1/(1+np.exp(-x_input))
    return output

def w1_sigmoid_grad_input(x_input, grad_output):
    """sigmoid nonlinearity gradient. 
        Calculate the partial derivative of the loss 
        with respect to the input of the layer
    # Arguments
        x_input: np.array of size `(n_objects, n_in)`
        grad_output: np.array of size `(n_objects, n_in)` 
            dL / df
    # Output
        the partial derivative of the loss 
        with respect to the input of the function
        np.array of size `(n_objects, n_in)` 
        dL / dh
    """
    #################
    ### YOUR CODE ###  #df/dh * grad_output
    #################
    return np.multiply(w1_sigmoid_forward(x_input)*(1-w1_sigmoid_forward(x_input)), grad_output)


#######################################################
# put `w1_nll_forward` and `w1_nll_grad_input` here    #
#######################################################
def w1_nll_forward(target_pred, target_true):
    """Compute the value of NLL
        for a given prediction and the ground truth
    # Arguments
        target_pred: predictions - np.array of size `(n_objects, 1)`
        target_true: ground truth - np.array of size `(n_objects, 1)`
    # Output
        the value of NLL for a given prediction and the ground truth
        scalar
    """
    #################
    ### YOUR CODE ###
    #################
    (n_obj, s) = target_pred.shape
    a = np.ones((n_obj,1))
#     print(target_pred.shape)
#     X[X[:,feature_index]<split_value,:]
    target_pred[target_pred[:,0]==0,:] = 0.00000001
    target_pred[target_pred[:,0]==1,:] = 0.99999999
    
#     target_pred[] = target_pred + np.multiply(a, b)
    output = np.dot(np.transpose(target_true), np.log(target_pred))+np.dot(np.transpose(a-target_true), np.log(a-target_pred))
    return -output/n_obj


def w1_nll_grad_input(target_pred, target_true):
    """Compute the partial derivative of NLL
        with respect to its input
    # Arguments
        target_pred: predictions - np.array of size `(n_objects, 1)`
        target_true: ground truth - np.array of size `(n_objects, 1)`
    # Output
        the partial derivative 
        of NLL with respect to its input
        np.array of size `(n_objects, 1)`
    """
    #################
    ### YOUR CODE ###
    #################
    (n_obj, s) = target_pred.shape
    a = np.ones((n_obj,1))
    denominator = np.multiply(target_pred, a-target_pred)
    denominator = np.clip(denominator, a_min = 0.00001, a_max = np.inf)
    grad_input = (target_pred-target_true)/denominator
    return grad_input/n_obj