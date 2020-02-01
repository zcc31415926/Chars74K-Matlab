%derivative of softmax operation
function gradient = softmax_diff(softmaxed_output)
    gradient = zeros(length(softmaxed_output), length(softmaxed_output));
    for i = 1 : length(softmaxed_output)
        for j = 1 : length(softmaxed_output)
            if i == j
                gradient(i, j) = softmaxed_output(i) * (1 - softmaxed_output(i));
            else
                gradient(i, j) = -softmaxed_output(i) * softmaxed_output(j);
            end
        end
    end
