%ReLU activation function
function activated_output = relu(output)
    activated_output = output;
    activated_output(activated_output < 0) = 0;
