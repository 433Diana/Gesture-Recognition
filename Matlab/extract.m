clear

model = 'quan_model.hdf5';
info = h5info(model);

conv1_kernel_32bit = h5read(model,'/quant_conv2d/conv2d/kernel:0');
conv1_bias_32bit = h5read(model,'/quant_conv2d/conv2d/bias:0');
save('variable_32bit\conv1_kernel_32bit.mat','conv1_kernel_32bit');
save('variable_32bit\conv1_bias_32bit.mat','conv1_bias_32bit');

conv1_max = h5read(model,'/quant_conv2d/quant_conv2d/kernel_max:0');
conv1_min = h5read(model,'/quant_conv2d/quant_conv2d/kernel_min:0');
conv1_kernel_int8 = zeros(size(conv1_kernel_32bit));
conv1_bias_int8 = zeros(size(conv1_bias_32bit));

for i = 1:size(conv1_max)
    conv1_kernel_int8(i,:,:,:) = int8(conv1_kernel_32bit(i,:,:,:)/conv1_max(i)*256);
    conv1_bias_int8(i) = int8(conv1_bias_32bit(i)/conv1_max(i)*256);
end
save('variable_int8\conv1_kernel_int8.mat','conv1_kernel_int8');
save('variable_int8\conv1_bias_int8.mat','conv1_bias_int8');

conv1_kernel_8bit = single(fi(conv1_kernel_32bit,1,8,7));
conv1_bias_8bit = single(fi(conv1_bias_32bit,1,8,7));
save('variable_8bit\conv1_kernel_8bit.mat','conv1_kernel_8bit');
save('variable_8bit\conv1_bias_8bit.mat','conv1_bias_8bit');


conv2_kernel_32bit = h5read(model,'/quant_conv2d_1/conv2d_1/kernel:0');
conv2_bias_32bit = h5read(model,'/quant_conv2d_1/conv2d_1/bias:0');
save('variable_32bit\conv2_kernel_32bit.mat','conv2_kernel_32bit');
save('variable_32bit\conv2_bias_32bit.mat','conv2_bias_32bit');

conv2_max = h5read(model,'/quant_conv2d_1/quant_conv2d_1/kernel_max:0');
conv2_min = h5read(model,'/quant_conv2d_1/quant_conv2d_1/kernel_min:0');
conv2_kernel_int8 = zeros(size(conv2_kernel_32bit));
conv2_bias_int8 = zeros(size(conv2_bias_32bit));

for i = 1:size(conv2_max)
    conv2_kernel_int8(i,:,:,:) = int8(conv2_kernel_32bit(i,:,:,:)/conv2_max(i)*256);
    conv2_bias_int8(i) = int8(conv2_bias_32bit(i)/conv2_max(i)*256);
end
save('variable_int8\conv2_kernel_int8.mat','conv2_kernel_int8');
save('variable_int8\conv2_bias_int8.mat','conv2_bias_int8');
conv2_kernel_8bit = single(fi(conv2_kernel_32bit,1,8,7));
conv2_bias_8bit = single(fi(conv2_bias_32bit,1,8,7));
save('variable_8bit\conv2_kernel_8bit.mat','conv2_kernel_8bit');
save('variable_8bit\conv2_bias_8bit.mat','conv2_bias_8bit');


dense1_kernel_32bit = h5read(model,'/quant_dense/dense/kernel:0');
dense1_bias_32bit = h5read(model,'/quant_dense/dense/bias:0');
save('variable_32bit\dense1_kernel_32bit.mat','dense1_kernel_32bit');
save('variable_32bit\dense1_bias_32bit.mat','dense1_bias_32bit');

dense1_max = h5read(model,'/quant_dense/quant_dense/kernel_max:0');
dense1_min = h5read(model,'/quant_dense/quant_dense/kernel_min:0');
dense1_kernel_int8 = zeros(size(dense1_kernel_32bit));
dense1_bias_int8 = zeros(size(dense1_bias_32bit));

for i = 1:size(dense1_max)
    dense1_kernel_int8(:,:) = int8(dense1_kernel_32bit(:,:)/dense1_max(i)*256);
    dense1_bias_int8(:) = int8(dense1_bias_32bit(:)/dense1_max(i)*256);
end
save('variable_int8\dense1_kernel_int8.mat','dense1_kernel_int8');
save('variable_int8\dense1_bias_int8.mat','dense1_bias_int8');
dense1_kernel_8bit = single(fi(dense1_kernel_32bit,1,8,7));
dense1_bias_8bit = single(fi(dense1_bias_32bit,1,8,7));
save('variable_8bit\dense1_kernel_8bit.mat','dense1_kernel_8bit');
save('variable_8bit\dense1_bias_8bit.mat','dense1_bias_8bit');
