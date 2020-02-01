%derivative of ReLU activation function
function gradient = relu_diff(activated_output)
    gradient = activated_output;
    gradient(gradient > 0) = 1;
