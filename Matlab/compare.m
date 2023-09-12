clear
clc
time = cputime;

figdir = 'figure\';
gesture = ["OK","NOTHING","PEACE","PUNCH","STOP"];
word = 8;
fraction = 3;

img_path_list = dir(strcat(figdir,'*.png'));
img_num = length(img_path_list);

output_32bit = zeros(img_num,1);
output_8bit = zeros(img_num,1);
output_gd = zeros(img_num,1);
output_gd(1:(img_num/5),1) = 1;
output_gd(1+(img_num/5):(2*img_num/5),1) = 2;
output_gd(1+(2*img_num/5):(3*img_num/5),1) = 3;
output_gd(1+(3*img_num/5):(4*img_num/5),1) = 4;
output_gd(1+(4*img_num/5):(5*img_num/5),1) = 5;

for j = 1:img_num
    if mod(j,100) == 0
        disp(append('Original model: ',string(j),'/',string(length(img_path_list))));
    end

    figname = img_path_list(j).name;
    image = imread(append(figdir,figname));
    img = single(image(:,:,1));
    img = img/255;

    bits = '8bit';

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
    
    output = softmax(cls_dense1_out);
    [prob,index] = max(output);
    output_8bit(j) = index;
end

for j = 1:img_num
    if mod(j,100) == 0
        disp(append('Quantization model: ',string(j),'/',string(length(img_path_list))));
    end

    figname = img_path_list(j).name;
    image = imread(append(figdir,figname));
    img = single(image(:,:,1));
    img = img/255;

    bits = '32bit';

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
    
    output = softmax(cls_dense1_out);
    [prob,index] = max(output);
    output_32bit(j) = index;
end

acc_32bit = length(find(output_32bit==output_gd))/img_num;
acc_8bit = length(find(output_8bit==output_gd))/img_num;

fprintf('Accuracy of original model is %f.\n',acc_32bit);
fprintf('Accuracy of quantization model is %f.\n',acc_8bit);
fprintf('Accuracy is reduced by %f after quantization.\n',(acc_32bit-acc_8bit));

time = cputime - time;
fprintf('Runtime is %.2f seconds.\n',time);
