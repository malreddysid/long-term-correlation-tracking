function feature = calcIIF(I,kernel_size,nbins)


src = I;
dst = zeros(size(I));
mask = zeros(size(I));

step = 256/nbins;

for i = 1:nbins
    A = src>= (i-1)*step;
    B = src<= i*step;
    temp = 255*(A&B);
    mask = mask + temp;
    
    h = fspecial('average', kernel_size);
    temp_blr = imfilter(temp, h, 'replicate');
    dst = dst + (mask.*temp_blr)*(1/255.0);
    
end


feature = dst;

end