function [H] = computeFeatures(img, cell_size, cos_window)

img = single(img);
img = img/255;
H = double(fhog(img, cell_size, 9));
H(:,:,end) = [];

if(~isempty(cos_window))
    H = bsxfun(@times, H, cos_window);
end

end