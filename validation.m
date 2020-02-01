%validation process of CAPITAL letters recognition
%dataset: Chars74K
%data: 128*128 single-channel images, 1016*26 in total
%800 images for training and 216 for validation

pkg load image;

num_sample = 100;

dataset_path = '/home/charlie/Data/Chars74K/English/Fnt/Sample0';

if exist('./workspace.mat', 'file')
    load('./workspace.mat');
else
    disp('no workspace MAT file found.');
    exit;
end

correct_counter = zeros(26, 1);

for letter_index = 1 : 26
    for sample = 1 : num_sample
        img_index = randi(216) + 800;
        image = processImage(dataset_path, letter_index, img_index);

        input = reshape(image, [32 * 32, 1]);

        %forward calculation
        y1 = W1 * input + b1;
        y1_ = relu(y1);
        y2 = W2 * y1_ + b2;
        y2_ = relu(y2);
        y3 = W3 * y2_ + b3;
        prediction = softmax(y3);
        [max_value, max_index] = max(prediction);

        if max_index == letter_index
            correct_counter(letter_index) = correct_counter(letter_index) + 1;
        end
    end
end

accuracy_file = fopen('./accuracy.txt', 'a');
disp('accuracy:');
for letter_index = 1 : 26
    disp([num2str(letter_index), ': ', num2str(correct_counter(letter_index) / num_sample)]);
    fprintf(accuracy_file, '%6.2f\n', correct_counter(letter_index) / num_sample);
end
fclose(accuracy_file);
