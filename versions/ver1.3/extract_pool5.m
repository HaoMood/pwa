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
% Updated    2017-09-17
% Version    1.3


function [] = extract_pool5()
    % Main function of the program.
    clear, clc, close all

    % Path to save the raw pool5 features.
    project_root = '/data/zhangh/project/pwa/';
    all_image_path = '/data/zhangh/data/oxbuild/images/';
    crop_image_path = '/data/zhangh/data/oxbuild/cropped-queries/';
    all_pool5_data_path = [project_root, 'data/pool5/data-all/'];
    all_pool5_shape_path = [project_root, 'data/pool5/shape-all/'];
    crop_pool5_data_path = [project_root, 'data/pool5/data-crop/'];
    crop_pool5_shape_path = [project_root, 'data/pool5/shape-crop/'];

    % Setting the GPU device.
    opt.gpu = 2;
    g = gpuDevice(opt.gpu);
    reset(g);

    net = prepareNet();
    getPool5(net, all_image_path, all_pool5_data_path, all_pool5_shape_path);
    getPool5(net, crop_image_path, crop_pool5_data_path, crop_pool5_shape_path);
end


function net = prepareNet()
    % Prepare the pre-trained VGG-16 model.
    %
    % Returns:
    %     net: VGG-16 model.
    fprintf('Prepare CNN model.\n');
    % load CNN model.
    run('/data/zhangh/usr/matconvnet-1.0-beta24/matlab/vl_setupnn.m');
    % The pre-trained model.
    net = load('/data/zhangh/model/matconvnet/imagenet-vgg-verydeep-16.mat');
    net.layers(end-5 : end)=[]; % Removing the fully connected layers
    net = vl_simplenn_move(net, 'gpu');

    % Using the RGB average values obtained from ImageNet.
    net.normalization.averageImage = ones(224, 224, 3);
    net.normalization.averageImage(:, :, 1) = ...
        net.normalization.averageImage(:, :, 1) ...
        .* net.meta.normalization.averageImage(1, 1);
    net.normalization.averageImage(:, :, 2) = ...
        net.normalization.averageImage(:, :, 2) ...
        .* net.meta.normalization.averageImage(1, 2);
    net.normalization.averageImage(:, :, 3) = ...
        net.normalization.averageImage(:, :, 3) ...
        .* net.meta.normalization.averageImage(1, 3);
end


function [] = getPool5(net, image_path, data_path, shape_path)
    % Extract pool5 features for full images/cropped query images.
    % 
    % Args:
    %     net: VGG-16 model.
    %     image_path, str: Path for .jpg images.
    %     data_path, str: Path to save pool5 features.
    %     shape_path, str: Path to save pool5 shapes.
    fprintf('Extract raw VGG-16 pool5 features for %s.\n', image_path)
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
            im_ = im_ - imresize(net.normalization.averageImage, [h, w]);
        else    
            im_ = bsxfun(@minus,im_,imresize(...
                net.normalization.averageImage, [h, w]));
        end
        
        % Save pool5 features onto disk. Each with size HW*D.
        res = vl_simplenn(net, gpuArray(im_));
        tmp_1 = gather(res(32).x);
        [h_pool5, w_pool5, c_pool5] = size(tmp_1);
        csvwrite([data_path image_name(1 : end-4) '.csv'], ...
                 reshape(tmp_1, h_pool5 * w_pool5, c_pool5));
        csvwrite([shape_path image_name(1 : end-4) '.csv'], size(tmp_1));
    end
end
