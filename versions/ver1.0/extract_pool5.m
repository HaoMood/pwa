% Forward pass to extract raw VGG-16 pool5 features for Oxford dataset.
%
% The pool5 features are saved in .csv format. One .csv file for one image.
%
% This file is modified from SCDA:
%     http://lamda.nju.edu.cn/code_SCDA.ashx 
% Since relu5-2 features are not used in general image retrieval, those part of
% codes (codes to compute L28 features) are deleted.


% Author     Hao Zhang
% Copyright  Copyright @2017 LAMDA
% Date       2017-09-15
% Email      zhangh0214@gmail.com
% License    CC BY-SA 3.0
% Status     Development
% Updated    2017-09-15
% Version    1.0


clear, clc, close all
fprintf('Forward pass to extract raw VGG-16 pool5 features.\n')

% Path to save the raw pool5 features.
project_root = '/data/zhangh/project/pwa/';
image_path = '/data/zhangh/data/oxbuild/images/';
pool5_data_path = [project_root, 'data/pool5/data/'];
pool5_shape_path = [project_root, 'data/pool5/shape/'];

% Setting the GPU device
opt.gpu = 7;
g = gpuDevice(opt.gpu);
reset(g);

% load CNN model
run('/data/zhangh/usr/matconvnet-1.0-beta24/matlab/vl_setupnn.m');
% The pre-trained model--VGG-19
net = load('/data/zhangh/model/matconvnet/imagenet-vgg-verydeep-16.mat');
net.layers(end-5:end)=[]; % Removing the fully connected layers
net = vl_simplenn_move(net, 'gpu');

% Using the RGB average values obtained from ImageNet
net.normalization.averageImage = ones(224,224,3);
net.normalization.averageImage(:,:,1) = net.normalization.averageImage(:,:,1) .* net.meta.normalization.averageImage(1,1);
net.normalization.averageImage(:,:,2) = net.normalization.averageImage(:,:,2) .* net.meta.normalization.averageImage(1,2);
net.normalization.averageImage(:,:,3) = net.normalization.averageImage(:,:,3) .* net.meta.normalization.averageImage(1,3);
fprintf('CNN model is ready.\n');

% Load Oxford datasets.
oxford_image_names = dir(image_path);
m = length(oxford_image_names);
for i = 1 : m
    image_name = oxford_image_names(i).name;
    if image_name(end) ~= 'g'
        continue;
    end
    if mod(i, 100) == 0
        fprintf('Process %d/%d\n', i, m)
    end

    %% original image. We keep the original image size.
    im = imread([image_path image_name]);
    im_ = single(im);
    [h, w, c] = size(im_);
    if c > 2
        im_ = im_ - imresize(net.normalization.averageImage,[h,w]);
    else    
        im_ = bsxfun(@minus,im_,imresize(net.normalization.averageImage,[h,w]));
    end
    
    % Save pool5 features onto disk. Each with size HW*D.
    res = vl_simplenn(net, gpuArray(im_)) ;
    tmp_1 = gather(res(32).x);
    [h_pool5, w_pool5, c_pool5] = size(tmp_1);
    csvwrite([pool5_data_path image_name(1:end-4) '.csv'], ...
             reshape(tmp_1, h_pool5 * w_pool5, c_pool5));
    csvwrite([pool5_shape_path image_name(1:end-4) '.csv'], size(tmp_1));
end
fprintf('Done. Bye.\n')
