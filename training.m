%training process of CAPITAL letters recognition
%dataset: Chars74K
%data: 128*128 single-channel images, 1016*26 in total
%800 images for training and 216 for validation

pkg load image;

num_iter = 2000;
batch_size = 16;
lr = 0.0005;
momentum = 0.9;

%network structure: 32*32 -> 256 -> 64 -> 26
W1 = randn(256, 32 * 32) * 0.1;
b1 = randn(256, 1) * 0.1;
W2 = randn(64, 256) * 0.1;
b2 = randn(64, 1) * 0.1;
W3 = randn(26, 64) * 0.1;
b3 = randn(26, 1) * 0.1;
history_gradient_W1 = zeros(size(W1));
history_gradient_W2 = zeros(size(W2));
history_gradient_W3 = zeros(size(W3));
history_gradient_b1 = zeros(size(b1));
history_gradient_b2 = zeros(size(b2));
history_gradient_b3 = zeros(size(b3));
if exist('./workspace.mat', 'file')
    load('./workspace.mat');
end

dataset_path = '/home/charlie/Data/Chars74K/English/Fnt/Sample0';
loss_file = fopen('./loss_value.txt', 'a');

loss_value = -1;

for iter = 1 : num_iter
    accumulated_gradient_W1 = zeros(size(W1));
    accumulated_gradient_W2 = zeros(size(W2));
    accumulated_gradient_W3 = zeros(size(W3));
    accumulated_gradient_b1 = zeros(size(b1));
    accumulated_gradient_b2 = zeros(size(b2));
    accumulated_gradient_b3 = zeros(size(b3));
    accumulated_loss = 0;

    for sample = 1 : batch_size

        letter_index = randi(26);
        img_index = randi(800);
        image = processImage(dataset_path, letter_index, img_index);

        input = reshape(image, [32 * 32, 1]);
        label = oneHot(letter_index, 26);

        %forward calculation
        y1 = W1 * input + b1;
        y1_ = relu(y1);
        y2 = W2 * y1_ + b2;
        y2_ = relu(y2);
        y3 = W3 * y2_ + b3;
        prediction = softmax(y3);

        %loss calculation
        loss_value = loss(prediction, label);
        accumulated_loss = accumulated_loss + loss_value;

        %gradient backprop
        gradient_prediction = loss_diff(prediction, label);

        gradient_y3 = softmax_diff(prediction) * gradient_prediction;
        gradient_W3 = gradient_y3 * y2_';
        gradient_b3 = gradient_y3;
        gradient_y2_ = W3' * gradient_y3;

        gradient_y2 = gradient_y2_ .* relu_diff(y2_);
        gradient_W2 = gradient_y2 * y1_';
        gradient_b2 = gradient_y2;
        gradient_y1_ = W2' * gradient_y2;

        gradient_y1 = gradient_y1_ .* relu_diff(y1_);
        gradient_W1 = gradient_y1 * input';
        gradient_b1 = gradient_y1;

        accumulated_gradient_W1 = accumulated_gradient_W1 + gradient_W1;
        accumulated_gradient_W2 = accumulated_gradient_W2 + gradient_W2;
        accumulated_gradient_W3 = accumulated_gradient_W3 + gradient_W3;
        accumulated_gradient_b1 = accumulated_gradient_b1 + gradient_b1;
        accumulated_gradient_b2 = accumulated_gradient_b2 + gradient_b2;
        accumulated_gradient_b3 = accumulated_gradient_b3 + gradient_b3;
    end

    %batch-wise training with gradient clip
    average_gradient_W1 = clipGradient(accumulated_gradient_W1 / batch_size, 1, -1);
    average_gradient_W2 = clipGradient(accumulated_gradient_W2 / batch_size, 1, -1);
    average_gradient_W3 = clipGradient(accumulated_gradient_W3 / batch_size, 1, -1);
    average_gradient_b1 = clipGradient(accumulated_gradient_b1 / batch_size, 1, -1);
    average_gradient_b2 = clipGradient(accumulated_gradient_b2 / batch_size, 1, -1);
    average_gradient_b3 = clipGradient(accumulated_gradient_b3 / batch_size, 1, -1);

    %parameter update (momentum)
    if sum(sum(history_gradient_W1)) == 0
        W1 = W1 - lr * average_gradient_W1;
        W2 = W2 - lr * average_gradient_W2;
        W3 = W3 - lr * average_gradient_W3;
        b1 = b1 - lr * average_gradient_b1;
        b2 = b2 - lr * average_gradient_b2;
        b3 = b3 - lr * average_gradient_b3;
    else
        W1 = W1 - lr * (momentum * history_gradient_W1 + (1 - momentum) * average_gradient_W1);
        W2 = W2 - lr * (momentum * history_gradient_W2 + (1 - momentum) * average_gradient_W2);
        W3 = W3 - lr * (momentum * history_gradient_W3 + (1 - momentum) * average_gradient_W3);
        b1 = b1 - lr * (momentum * history_gradient_b1 + (1 - momentum) * average_gradient_b1);
        b2 = b2 - lr * (momentum * history_gradient_b2 + (1 - momentum) * average_gradient_b2);
        b3 = b3 - lr * (momentum * history_gradient_b3 + (1 - momentum) * average_gradient_b3);
    end

    history_gradient_W1 = average_gradient_W1;
    history_gradient_W2 = average_gradient_W2;
    history_gradient_W3 = average_gradient_W3;
    history_gradient_b1 = average_gradient_b1;
    history_gradient_b2 = average_gradient_b2;
    history_gradient_b3 = average_gradient_b3;

    if rem(iter, 10) == 0
        disp(['iter: ', num2str(iter), ' loss: ', num2str(accumulated_loss / batch_size)]);
        fprintf(loss_file, '%6.3f\n', accumulated_loss / batch_size);
    end
    if rem(iter, 100) == 0
        save('workspace.mat');
    end
end

fclose(loss_file);
