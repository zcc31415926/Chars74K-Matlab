%softmax algorithm
function softmaxed_output = softmax(output)
    softmaxed_output = exp(output) / sum(exp(output));
