function[Conv_out] = CNN_Conv_1ch(img,kernel,conv_stride)
%img:           input image
%kernel:        convolution kernel
%conv_stride:   convolution stride

img_w       = size(img,1);      %width of the image
img_h       = size(img,2);      %height of the image
kernel_size = size(kernel,3);   %size of the convolution kernel

Conv_out    = zeros(floor((img_w-kernel_size)/conv_stride+1),floor((img_h-kernel_size)/conv_stride+1)); %size of the output
Conv_out    = single(Conv_out);
kernel      = squeeze(kernel);  %remove a single dimension (channel dimension)
kernel      = kernel';

for i = 1:conv_stride:img_w-kernel_size+1
    for j = 1:conv_stride:img_h-kernel_size+1
        Conv_out((i-1)/conv_stride+1,(j-1)/conv_stride+1) = sum(sum(img(i:i+kernel_size-1,j:j+kernel_size-1).*kernel));
    end
end
end