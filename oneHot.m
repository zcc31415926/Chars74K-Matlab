%one-hot vector generation algorithm
function one_hot_vector = oneHot(index, dim)
    one_hot_vector = zeros(dim, 1);
    one_hot_vector(index) = 1;
