function getRects(file_path)

files = dir(strcat(file_path, '*.jpg'));
num_files = length(files);
ours = csvread('../Jogging/rects_conv.txt');
iptsetpref('ImshowBorder','tight');
fCount = num_files;
detlaT = 0.033;
current_image_name = files(1).name;
current_image_path = strcat(file_path, current_image_name);
imshow(imread(current_image_path)); hold on;
rectangle('Position', [ours(1,1) ours(1,2) ours(1,3) ours(1,4)]...
        , 'EdgeColor', 'r','LineWidth',3);
drawnow;


f = getframe(gcf);
[im,map] = rgb2ind(f.cdata,256,'nodither');
im(1,1,1,fCount) = 0;
k = 1;
for i = 1:1:num_files
    current_image_name = files(i).name;
    current_image_path = strcat(file_path, current_image_name);
    imshow(imread(current_image_path)); hold on;
    rectangle('Position', [ours(i,1) ours(i,2) ours(i,3) ours(i,4)]...
        , 'EdgeColor', 'r','LineWidth',3);
    drawnow;
    f = getframe(gcf);
    im(:,:,1,k) = rgb2ind(f.cdata,map,'nodither');
    k = k+1;
end


imwrite(im,map,'Human_conv.gif','DelayTime',detlaT,'LoopCount',inf);

end