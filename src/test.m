I = imread('../data/Basketball/img/0001.jpg');
H = computeFeatures(I, 4);

[size_x, size_y, ~] = size(H);

yf = getLabelImage(size_x, size_y, 20);

lambda = 1e-4;

A = yf./(