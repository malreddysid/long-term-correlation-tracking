function [opt_scale,sf] = getOptimalScale(scale_pyr,scale,s_num,s_den)

global lambda_s;
[~,N] = size(scale_pyr);


sf = fft(scale_pyr,[],2);
scale_response = real(ifft(sum(s_num .* sf, 1) ./ (s_den + lambda_s)));

s_ind = find(scale_response == max(scale_response(:)), 1);

% Optimal scale value
opt_scale = scale{s_ind};


end