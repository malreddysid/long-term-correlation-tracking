function [pos, max_response] = refine_pos_rf(im, pos, svm_struct, app_model, config)

max_response = config.max_response;

motion_model_patch_size = config.motion_model_patch_size;
app_model_patch_size = config.app_model_patch_size;

A_t = app_model.A_t;
xf_t = app_model.xf_t; 

[feat, pos_samples, ~, weights]= det_samples2(im, pos, motion_model_patch_size, config.detc);


scores = svm_struct.w'*feat + svm_struct.b;

scores = scores.*reshape(weights,1,[]);

tpos = round(pos_samples(:, find(scores==max(scores),1)));

if isempty(tpos),  return; end

tpos = reshape(tpos,1,[]);

% figure(2), imshow(im),
% hold on, plot(tpos(2), tpos(1), 'xg');

if size(im,3) > 1
    im=rgb2gray(im);
end

cell_size = 4;
patch = getPatch(im, tpos, app_model_patch_size);
zf_t = fft2(computeFeatures(patch, cell_size, []));
[~,max_response] = getNewPos(zf_t, xf_t, A_t);


if max_response>1.5*config.max_response && max(scores)>0
% if max_response>config.appearance_thresh && max(scores)>0
    fprintf('position update by detector\n')
    pos = tpos;
else
    max_response = config.max_response;
    fprintf('No update\n')
end