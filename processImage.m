%dataset path: $DataDir$/Chars74K/English/Fnt/Sample0xx/img0xx-0yyyy.png
%dir range: Sample011(A) - Sample036(Z)
%1016 images in each dir (each character type)

%letter_index: 1(A) - 26(Z)
%img_index: 1-1016
%return: 32*32 image
function image = processImage(dataset_path, letter_index, img_index)
    letter_index_in_filename = num2str(letter_index + 10);
    img_index_in_filename = num2str(img_index, '%04d');
    image = imread(strcat(dataset_path, letter_index_in_filename, ...
        '/img0', letter_index_in_filename, '-0', img_index_in_filename, '.png'));
    image = double(imresize(image, 0.25)) / 255;
