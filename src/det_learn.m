

function [svm_struct] = det_learn( im, pos, motion_model_patch_size, det_config, svm_struct )
%
[feat, posamples, labels]=det_samples(im, pos, motion_model_patch_size, det_config);

idx_p = labels>0.9;
idx_n = labels<0.1;

feat = feat(:, idx_p|idx_n);



labels(idx_p) = 1;
labels(idx_n) = -1;
labels = labels(idx_p|idx_n);



if isempty(svm_struct)
    
[svm_struct.w, svm_struct.b]= vl_svmtrain(feat, labels, 0.5) ;

else  
    svm_struct.w = onlineSVMTrain(feat', labels, 0.5, svm_struct.w);   
end


end


function [w] = onlineSVMTrain(X, y, C, w)
%TRAIN_PA Train/update passive-aggressive online learner
%
%   w = train_pa(X, y, C, w)
%
% The function trains or updates a linear classifier using the
% passive-aggressive online learning algorithm. The variable C is the
% agressiveness parameter. The functions assumes the labels are in [-1, +1].
%


[N, D] = size(X);
if ~exist('w', 'var') || isempty(w)
    w = randn(D, 1);
    iter = 10;
else
    iter = 1;
end
if any(~(y == -1 | y == 1))
    error('Labels should be in [-1, 1].');
end        

% Perform updates
for i=1:iter
    for n=1:N
        % Perform prediction and suffer loss
        loss = max(0, 1 - y(n) .* (X(n,:) * w));
        % Update weights
        if loss > 0
           w = w + (loss ./ (sum(X(n,:) .^ 2) + (1 ./ (2 .* C)))) * (X(n,:)' * y(n)); 
        end
    end
end

end

