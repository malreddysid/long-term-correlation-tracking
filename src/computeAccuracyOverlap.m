
GT = csvread('../data/Jogging/groundtruth_rect.txt');

ours = csvread('../Jogging/rects.txt');
ours_conv = csvread('../Jogging/rects_conv.txt');
ours_hog = csvread('../Jogging/rects_hog.txt');

num_frames = size(GT, 1);

assert(size(GT, 1) == size(ours, 1));
assert(size(GT, 1) == size(ours_conv, 1));

area_overlap = rectint(GT, ours);
area_overlap = diag(area_overlap);

GT_area = GT(:,3).*GT(:,4);

overlap_threshold = 0:0.01:1;
num_t = numel(overlap_threshold);
num_correct = zeros(1, num_t);

for i = 1:num_t
    num_correct(i) = sum(area_overlap > GT_area.*overlap_threshold(i))/num_frames;
end

plot(overlap_threshold, num_correct, 'r', 'LineWidth', 1);
title('Accuracy vs Overlap Threshold');
xlabel('Overlap Threshold');
ylabel('Accuracy');
hold on;

area_overlap = rectint(GT, ours_conv);
area_overlap = diag(area_overlap);

GT_area = GT(:,3).*GT(:,4);

overlap_threshold = 0:0.01:1;
num_t = numel(overlap_threshold);
num_correct = zeros(1, num_t);

for i = 1:num_t
    num_correct(i) = sum(area_overlap > GT_area.*overlap_threshold(i))/num_frames;
end

plot(overlap_threshold, num_correct, 'b', 'LineWidth', 1);

area_overlap = rectint(GT, ours_hog);
area_overlap = diag(area_overlap);

GT_area = GT(:,3).*GT(:,4);

overlap_threshold = 0:0.01:1;
num_t = numel(overlap_threshold);
num_correct = zeros(1, num_t);

for i = 1:num_t
    num_correct(i) = sum(area_overlap > GT_area.*overlap_threshold(i))/num_frames;
end

plot(overlap_threshold, num_correct, 'g', 'LineWidth', 1);



legend('No Detection','ResNet','HOG');