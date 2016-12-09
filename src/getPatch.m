function [patch] = getPatch(img, pos, patch_size)

x = floor(pos(2) + (1:patch_size(2)) - floor(patch_size(2)/2));
y = floor(pos(1) + (1:patch_size(1)) - floor(patch_size(1)/2));

x(x < 1) = 1;
y(y < 1) = 1;
x(x > size(img, 2)) = size(img,2);
y(y > size(img, 1)) = size(img,1);

patch = img(y, x, :);

end