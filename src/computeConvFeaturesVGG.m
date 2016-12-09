function [H] = computeConvFeaturesVGG(img, net)

global depth;
input_data = {prepare_image(img)};

net.forward(input_data);

%H = net.blobs('conv3_3').get_data();
if(depth == 1)
    H = net.blobs('pool1').get_data();
elseif(depth == 2)
    H = net.blobs('pool2').get_data();
end

H = permute(H, [2, 1, 3]);

end

% ------------------------------------------------------------------------
function crops_data = prepare_image(im)
% ------------------------------------------------------------------------
% caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat contains mean_data that
% is already in W x H x C with BGR channels
global model_dir;
global depth;
mean_data = [104 117 123];

% Convert an image returned by Matlab's imread to im_data in caffe's data
% format: W x H x C with BGR channels
im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
im_data = single(im_data);  % convert from uint8 to single

% mean(:,:,1) = repmat([104.00698793], size(im_data, 1), size(im_data, 2));
% mean(:,:,2) = repmat([116.66876762], size(im_data, 1), size(im_data, 2));
% mean(:,:,3) = repmat([122.67891434], size(im_data, 1), size(im_data, 2));

im_data = permute(im_data, [2, 1, 3]);  % flip width and height
im_size = size(im_data);
im_data = imresize(im_data, [224 224], 'bilinear');  % resize im_data

for i = 1:3
    im_data(:,:,i) = im_data(:,:,i) - mean_data(1,i);
end

im_data = imresize(im_data, im_size(1:2)*depth, 'bilinear');  % resize im_data

%im_data = imresize(im_data, 4, 'bilinear');  % resize im_data
% oversample (4 corners, center, and their x-axis flips)
crops_data = im_data;
end