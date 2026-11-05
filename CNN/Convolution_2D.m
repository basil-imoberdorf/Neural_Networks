clear
close all


% Define filter weights
x_filter = [1,0,1;0,1,0;1,0,1];
x_filter = [0,1,0;1,1,1;0,1,0];
x_filter = [1,0,-1;1,0,-1;1,0,-1];
% x_filter = -[1,0,-1;1,0,-1;1,0,-1];
% x_filter = -[-1,-1,-1;-1,8,-1;-1,-1,-1]; % edge erkennung
% x_filter = -[0,-1,0;-1,4,-1;0,-1,0]; % edge erkennung

% Define base shape
fractal_base = [0,0,0,0,0
                0,1,0,1,0
                0,0,1,0,0
                0,1,0,1,0
                0,0,0,0,0];

% Stack shape
img = repmat(fractal_base, 5);


img = imread('..\Images\abstract.JPG');
img = compImag(img, 3, 'avg');  % compress image to reduce dimensionality
img = mean(img, 3)./255;  % convert color image to gray-scale


kernel_size = 3;
padding = 0;
stride = 1;

start_index = floor(kernel_size/2);

[x_size, y_size] = size(img);

x_out_size = floor((x_size + 2*padding - kernel_size)/stride) + 1;
y_out_size = floor((y_size + 2*padding - kernel_size)/stride) + 1;

conv = zeros(x_out_size, y_out_size);

% apply padding with zeros
if padding > 0
    img = [zeros(padding, x_size);img;zeros(padding, x_size)];
    img = [zeros(2*padding + x_size, padding),img, zeros(2*padding + x_size, padding)];
end

% Plot Shape
imagesc(img)

for i = 1:x_out_size
    for j = 1:y_out_size
        
        % Extract patch
        x_start_index = stride*(j-1) + 1; 
        x_end_index = stride*(j-1) + kernel_size;

        y_start_index = stride*(i-1) + 1;
        y_end_index = stride*(i-1) + kernel_size;

        patch = img(y_start_index:y_end_index, x_start_index:x_end_index);

        % Convolution
        conv(i,j) = sum(patch.*x_filter, 'all');
        
    end
end


figure
imagesc(conv)


































