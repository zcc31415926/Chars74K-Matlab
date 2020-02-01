%gradient clip algorithm
%set upper_bound and lower_bound 0 to turn off gradient clip
function clipped_gradient = clipGradient(gradient, upper_bound, lower_bound)
    clipped_gradient = gradient;
    if upper_bound ~= 0 || lower_bound ~= 0
        clipped_gradient(clipped_gradient > upper_bound) = upper_bound;
        clipped_gradient(clipped_gradient < lower_bound) = lower_bound;
    end
