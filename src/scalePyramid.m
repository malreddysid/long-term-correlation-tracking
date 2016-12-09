function [scale_pyr,scale] = scalePyramid(target_size,N,im,pos,cell_size,scale_window,current_scale)

global A_scale;
count = 1;

P = target_size(1,1);
Q = target_size(1,2);
scaleModelFactor = 1;
if prod(target_size) > 512
    scaleModelFactor = sqrt(512/prod(target_size));
end

xt = pos(1,1);
yt = pos(1,2);

for i = 1:N
    
    n = ceil(N/2)-i;
    
    %scale
    s = current_scale*A_scale^n;
    
    % extract patch
    [patch] = getPatch(im,[xt yt],[floor(s*P) floor(s*Q)]);
    
    % resize patch to P,Q
    re_patch = imResample(patch,[floor(scaleModelFactor*P) floor(scaleModelFactor*Q)]);
    
    % Compute HOG features
    [H] = computeFeatures(re_patch,cell_size,[]);
 
    % Creating target scale pyramid
    scale_pyr(:,i) = H(:)*scale_window(i);
    scale{count} = A_scale^n;
    count = count + 1;
    
end




end