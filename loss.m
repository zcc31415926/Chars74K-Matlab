%cross entropy loss function
function loss_value = loss(logit, label)
    logit(logit < 1e-8) = 1e-8;
    logit(logit > 1 - 1e-8) = 1 - 1e-8;
    log_logit = log(logit);
    log_1_logit = log(1 - logit);
    loss_value = -sum(label .* log_logit) - sum((1 - label) .* log_1_logit);
    %loss_value = loss_value / length(label);
