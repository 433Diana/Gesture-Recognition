clear
clc
time = cputime;

figdir = 'figure\';
figname = 'pppunch265.png';
bits = '8bit';
word = 8;
fraction = 4;
gesture = ["OK","NOTHING","PEACE","PUNCH","STOP"];

if (strcmp(bits,'32bit') == 0) && ... 
   (strcmp(bits,'int8') == 0)  && ...
   (strcmp(bits,'8bit') == 0)
    error('Wrong quantization bits.');
end

image = imread(append(figdir,figname));
img = single(image(:,:,1));
if (strcmp(bits,'int8') == 1)
    img = int8(img-128);
elseif (strcmp(bits,'32bit') == 1)
    img = img/255;
else
    img = single(fi(img/255,1,word,fraction));
end

%***************************-----CNN-----****************************%
%-------------------------------conv1--------------------------------%
load(append('variable_',bits,'\conv1_kernel_',bits,'.mat'));
load(append('variable_',bits,'\conv1_bias_',bits,'.mat'));

if (strcmp(bits,'32bit') == 1)
    conv1_kernel = conv1_kernel_32bit;
    conv1_bias = conv1_bias_32bit;
elseif (strcmp(bits,'int8') == 1)
    conv1_kernel = conv1_kernel_int8;
    conv1_bias = conv1_bias_int8; 
else 
    conv1_kernel = conv1_kernel_8bit;
    conv1_bias = conv1_bias_8bit; 
end 

if (strcmp(bits,'int8') == 1)
    conv1_out = int8(CNN_Conv(img,conv1_kernel,conv1_bias,1,1));
elseif (strcmp(bits,'32bit') == 1)
    conv1_out = single(CNN_Conv(img,conv1_kernel,conv1_bias,1,1));
else
    conv1_out = single(fi(CNN_Conv(img,conv1_kernel,conv1_bias,1,1),1,word,fraction));
end
%--------------------------------relu--------------------------------%
relu1_out = CNN_Relu(conv1_out);
%------------------------------maxpool-------------------------------%
maxpool1_out = CNN_MaxPool(relu1_out,2,2,0);

%-------------------------------conv2--------------------------------%
load(append('variable_',bits,'\conv2_kernel_',bits,'.mat'));
load(append('variable_',bits,'\conv2_bias_',bits,'.mat'));

if (strcmp(bits,'32bit') == 1)
    conv2_kernel = conv2_kernel_32bit;
    conv2_bias = conv2_bias_32bit;
elseif (strcmp(bits,'int8') == 1)
    conv2_kernel = conv2_kernel_int8;
    conv2_bias = conv2_bias_int8;
else
    conv2_kernel = conv2_kernel_8bit;
    conv2_bias = conv2_bias_8bit;
end 

if (strcmp(bits,'int8') == 1)
    conv2_out = int8(CNN_Conv(maxpool1_out,conv2_kernel,conv2_bias,1,1));
elseif (strcmp(bits,'32bit') == 1)
    conv2_out = single(CNN_Conv(maxpool1_out,conv2_kernel,conv2_bias,1,1));
else
    conv2_out = single(fi(CNN_Conv(maxpool1_out,conv2_kernel,conv2_bias,1,1),1,word,fraction));
end
%--------------------------------relu--------------------------------%
relu2_out = CNN_Relu(conv2_out);
%------------------------------maxpool-------------------------------%
maxpool2_out = CNN_MaxPool(relu2_out,2,2,0);

% %-------------------------------conv3--------------------------------%
% load(append('variable_',bits,'\conv3_kernel.mat'));
% load(append('variable_',bits,'\conv3_bias.mat'));
% conv3_out = CNN_Conv(maxpool2_out,conv3_kernel,conv3_bias,1,1);
% %--------------------------------relu--------------------------------%
% relu3_out = CNN_Relu(conv3_out);
% %------------------------------maxpool-------------------------------%
% maxpool3_out = CNN_MaxPool(relu3_out,2,2,0);

% %-------------------------------conv4--------------------------------%
% load(append('variable_',bits,'\conv4_kernel.mat'));
% load(append('variable_',bits,'\conv4_bias.mat'));
% conv4_out = CNN_Conv(maxpool3_out,conv4_kernel,conv4_bias,1,1);
% %--------------------------------relu--------------------------------%
% relu4_out = CNN_Relu(conv4_out);

%-------------------------------permute------------------------------%
permute_out = permute(maxpool2_out,[3 2 1]);
%-------------------------------reshape------------------------------%
flatten_out = reshape(permute_out,[],1);
%--------------------------------cls---------------------------------%
load(append('variable_',bits,'\dense1_kernel_',bits,'.mat'));
load(append('variable_',bits,'\dense1_bias_',bits,'.mat'));

if (strcmp(bits,'32bit') == 1)
    dense1_kernel = dense1_kernel_32bit;
    dense1_bias = dense1_bias_32bit;
elseif (strcmp(bits,'int8') == 1)
    dense1_kernel = dense1_kernel_int8;
    dense1_bias = dense1_bias_int8;
else
    dense1_kernel = dense1_kernel_8bit;
    dense1_bias = dense1_bias_8bit;
end 

if (strcmp(bits,'int8') == 1)
    cls_dense1_out = single(int8(dense1_kernel * flatten_out + dense1_bias));
elseif (strcmp(bits,'32bit') == 1)
    cls_dense1_out = single(dense1_kernel * flatten_out + dense1_bias);
else
    cls_dense1_out = single(fi(dense1_kernel * flatten_out + dense1_bias,1,word,fraction));
end

% cls_relu_out = CNN_Relu(cls_dense1_out);
% 
% load(append('variable_',bits,'\dense2_kernel.mat'));
% load(append('variable_',bits,'\dense2_bias.mat'));
% cls_dense2_out = dense2_kernel * cls_relu_out + dense2_bias;
% cls_dense2_out = single(cls_dense2_out);%GREEDY(cls_linear2_out,8)
output = softmax(cls_dense1_out);

[prob,index] = max(output);
disp(gesture(index));

imshow(append(figdir,figname));
title(append('Predict:',gesture(index)));

time = cputime - time;
fprintf('Runtime is %.2f seconds.\n',time);
