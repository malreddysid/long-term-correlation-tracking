function [diff, max_value] = getNewPos(zf, xf, A)

global kernel_width;
xzkf = computeGaussianCorrelation(zf, xf, kernel_width);

yf = fftshift(real(ifft2(A.*xzkf)));



max_value = max(yf(:));

[dx, dy] = find(yf == max_value, 1);
diff = [dx, dy];

end