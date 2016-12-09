
GT = csvread('../data/Jogging/groundtruth_rect.txt');

ours = csvread('../Jogging/rects.txt');
ours_conv = csvread('../Jogging/rects_conv.txt');
ours_hog = csvread('../Jogging/rects_hog.txt');

num_frames = size(GT, 1);

assert(size(GT, 1) == size(ours, 1));
assert(size(GT, 1) == size(ours_conv, 1));

overlap_threshold = 0:1:50;
num_t = numel(overlap_threshold);
num_correct = zeros(1, num_t);

for i = 1:num_t
    num_correct(i) = sum(abs(ours(:,1) - GT(:,1)) <= overlap_threshold(i) &...
        abs(ours(:,2) - GT(:,2)) <= overlap_threshold(i))/num_frames;
end

plot(overlap_threshold, num_correct, 'r', 'LineWidth', 1);
title('Accuracy vs Distance Precision');
xlabel('Distance Precision');
ylabel('Accuracy');
hold on;

for i = 1:num_t
    num_correct(i) = sum(abs(ours_conv(:,1) - GT(:,1)) <= overlap_threshold(i) &...
        abs(ours_conv(:,2) - GT(:,2)) <= overlap_threshold(i))/num_frames;
end

plot(overlap_threshold, num_correct, 'b', 'LineWidth', 1);


for i = 1:num_t
    num_correct(i) = sum(abs(ours_hog(:,1) - GT(:,1)) <= overlap_threshold(i) &...
        abs(ours_hog(:,2) - GT(:,2)) <= overlap_threshold(i))/num_frames;
end

plot(overlap_threshold, num_correct, 'g', 'LineWidth', 1);

legend('No Detection','ResNet','HOG');