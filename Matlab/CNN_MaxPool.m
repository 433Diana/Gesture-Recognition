function [maxpool_out] = CNN_MaxPool(img,kernel_size,stride,padding)
%img:           input image
%kernel_size:   pooling kernel
%stride:        pooling stride
%padding:       pooling padding

width   = size(img,1); %width of the image
height  = size(img,2); %height of the image
channel = size(img,3); %channels of the image

maxpool_in(width+2*padding,height+2*padding,channel) = 0; %size of the image after padding
maxpool_in = single(maxpool_in);
maxpool_in((padding+1):(padding+width),(padding+1):(padding+height),:) = img; %image after padding
maxpool_out(floor((width-kernel_size+2*padding)/stride+1),floor((height-kernel_size+2*padding)/stride+1),channel) = 0; %size of the image after convolution
maxpool_out = single(maxpool_out);

for i = 1:channel
    for j = 1:size(maxpool_out,1)
        for k = 1:size(maxpool_out,2)
            m = j * stride - 1;
            n = k * stride - 1;
            maxpool_out(j,k,i) = max(max(maxpool_in([m m+1],[n n+1],i)));
        end
    end
end


maxpool_out = single(maxpool_out);

end