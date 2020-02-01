%derivative of cross entropy loss function
function gradient = loss_diff(logit, label)
    logit(logit < 1e-8) = 1e-8;
    logit(logit > 1 - 1e-8) = 1 - 1e-8;
    gradient = -label ./ logit + (1 - label) ./ (1 - logit);
    %gradient = gradient / length(label);
