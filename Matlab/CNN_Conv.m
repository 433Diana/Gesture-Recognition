function [ Conv_out ] = CNN_Conv(img,kernel,bias,conv_stride,conv_padding)
%img:           input image
%kernel:        convolution kernel
%conv_stride:   convolution stride
%conv_padding:  convolution padding

img_w       = size(img,1);      %width of the image
img_h       = size(img,2);      %height of the image
img_num     = size(img,3);      %channels of the image
kernel_size = size(kernel,3);   %size of the convolution kernel
kernel_num  = size(kernel,1);   %number of the convolution kernel

Conv_in(img_w+2*conv_padding,img_h+2*conv_padding,img_num) = 0; %size of the image after padding
Conv_in = single(Conv_in);
Conv_in((conv_padding+1):(conv_padding+img_w),(conv_padding+1):(conv_padding+img_h),:) = img; %image after padding
Conv_out(floor((img_w-kernel_size+2*conv_padding)/conv_stride+1),floor((img_h-kernel_size+2*conv_padding)/conv_stride+1),kernel_num) = 0; %size of the image after convolution
Conv_out = single(Conv_out);

for i = 1:kernel_num
    for j = 1:img_num
         Conv_out(:,:,i) = Conv_out(:,:,i) + CNN_Conv_1ch(Conv_in(:,:,j),kernel(i,j,:,:),conv_stride);
    end
end
for i = 1:kernel_num
    Conv_out(:,:,i) = Conv_out(:,:,i) + bias(i);
end

Conv_out = single(Conv_out);

end