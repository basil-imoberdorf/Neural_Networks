clear
close all
% rng(42)

% Define input and output dimensions as well as the number of neurons per layer
input_dim = 14;
output_dim = 1;
dense_NeuralNet_design = [input_dim, 64, 16, output_dim];  % [Input, Hidden_Layer_1, Hidden_Layer_2, ... , Hidden_Layer_n, Output]

Num_Hidden_Layers  = numel(dense_NeuralNet_design) - 2;  % -2 to remove input and output layer from consideration
Num_Forward_Steps  = Num_Hidden_Layers + 1;  % we need to perform n-1 matrix multiplications + activation function to do one forward pass where n is the number of hidden layers + input and output layer, so n is total layers

% Initialize containers for weights, gradients and outputs at each layer
weights_cell = initialize_weights(dense_NeuralNet_design, 'xavier', 0);
weights_set_to_zero_cell = initialize_weights(dense_NeuralNet_design, 'zeros', 0);
weights_gradients_cell = weights_set_to_zero_cell;  % cell to store each individal gradient
weights_gradients_cell_batch = weights_set_to_zero_cell;  % cell to store the sum of gradients of a mini-batch. After the gradients of each training exaple in the mini-batch has been summed up the average is formed 

m_momentum_cell = weights_set_to_zero_cell; % cell array to hold the exponential moving average momentum vectors (first moment) 
v_squared_gradients_cell = weights_set_to_zero_cell; % cell array to hold the squared gradients (second moment) 

hiddenOutputs = cell(Num_Hidden_Layers, 1);
Layer_Derivative = cell(Num_Forward_Steps, 1);

% Import and display training image
tarim = imread('Images\euler.JPG');
tarim = compImag(tarim, 10, 'avg');  % compress image to reduce dimensionality
tarim = mean(tarim, 3)./255;  % convert color image to gray-scale

[tarDimX, tarDimY] = size(tarim);  % dimensions of image
tarim_size = [tarDimX, tarDimY];
cmap = magma; % <-- chose colormap

% artifical grayscale "image"
quadrant = [zeros(10,10),ones(10,10);ones(10,10),zeros(10,10)];
tarim = repmat(quadrant, 12); % chess board
[tarDimX, tarDimY] = size(tarim);
tarim_size = [tarDimX, tarDimY];

% Feature engineering: Computing Fourier features
x1_input  = ((1:tarDimX)./tarDimX) * 2*pi - pi;
x2_input  = ((1:tarDimY)./tarDimY) * 2*pi - pi;
x3_input  = sin(x1_input);       
x4_input  = sin(x2_input);       
x5_input  = cos(x1_input);       
x6_input  = cos(x2_input);
x7_input  = sin(2*x1_input);
x8_input  = sin(2*x2_input);
x9_input  = cos(2*x1_input);
x10_input = cos(2*x2_input);   
x11_input = sin(8*x1_input);
x12_input = sin(8*x2_input);
x13_input = cos(8*x1_input);
x14_input = cos(8*x2_input); 

% Create an matrix to pass all pixels features at once through the network. Bro we can predict the whole image in one go
[X1, X2] = meshgrid(x1_input, x2_input);
x1_img_vec = X1(:)'; % flatten to get one big ass vector to reach every pixel
x2_img_vec = X2(:)'; % flatten to get one big ass vector to reach every pixel
x3_img_vec  = sin(x1_img_vec);       
x4_img_vec  = sin(x2_img_vec);       
x5_img_vec  = cos(x1_img_vec);       
x6_img_vec  = cos(x2_img_vec);
x7_img_vec  = sin(2*x1_img_vec);
x8_img_vec  = sin(2*x2_img_vec);
x9_img_vec  = cos(2*x1_img_vec);
x10_img_vec = cos(2*x2_img_vec);   
x11_img_vec = sin(8*x1_img_vec);
x12_img_vec = sin(8*x2_img_vec);
x13_img_vec = cos(8*x1_img_vec);
x14_img_vec = cos(8*x2_img_vec); 
bias_filler_test = ones(1, length(x1_img_vec));
Full_Image_Input = [bias_filler_test; x1_img_vec; x2_img_vec; x3_img_vec; x4_img_vec; x5_img_vec; x6_img_vec; x7_img_vec; x8_img_vec; x9_img_vec; x10_img_vec; x11_img_vec; x12_img_vec; x13_img_vec; x14_img_vec];

% Presets and hyperparameters
Record_Video = 0;
max_iterations = 1000000;
iter_plot_ratio = 1000;
learn_rate = 0.0001;
emv_momentum = 0.9; % define what fraction of the previous gradient should be used for the current gradient. Only the remaining part, 1-momentum_share, is contributed by the current gradient. Is the beta term here: m_i+1 = beta*m_i + (1-beta)*gradients 
emv_sqgrad = 0.995; % Exponential Moving Average coefficient. Is the beta term here: v_i+1 = beta*v_i + (1-beta)*gradients.^2 
batch_size = 16;
bias_filler = ones(1, batch_size);
num_output_elements = output_dim * batch_size; % to divide by the correct factor to get the MEAN of the squared errors

emv_momentum_inv = 1-emv_momentum; % same idea as for above. save a computation in this case the scalar that defines the fraction of the current gradient used. depending on cache is bad mkay
emv_sqgrad_inv = 1-emv_sqgrad; % same idea as for above. save a computation in this case the scalar that defines the fraction of the current gradient used. depending on cache is bad mkay
eps = 1e-8; % for numerical stability. To avoid dividing by zero when rescaling the learing rate per parameter

Loss_sum = 0;
plot_index = 0;  % to use as index to keep track of the loss history, allows pre-allocate a array of the correct size given the iter_plot_ratio. avoids over allocating when iter_plot_ratio > 1
loss_history = zeros(max_iterations/iter_plot_ratio, 1);  % container to plot the evolution of the loss
mse_scaler = tarDimX*tarDimY;  % the sum of squared errors is divided by this number to get the MEAN squared error of the image but not the mini-batches which include only a subset 
test_image = zeros(tarDimX, tarDimY); % pre-allocate
full_tarim = tarim';
full_tarim = full_tarim(:)';

% % Set-up plot figure
% weights_handle = setup_weights_plot();
[fig1, ay, img_handle] = make_ll_canvas(tarim, cmap);
%[neuro_handle, line_cell] = setup_NN_canvas(dense_NeuralNet_design);
loss_plt = setup_loss_evolution_canvas(iter_plot_ratio);

if Record_Video == 1
    Filename = 'ChessBoard_14x32x32x1_magma2';
    Movie = VideoWriter(Filename,'MPEG-4');
    Movie.FrameRate = 30;
    Movie.Quality = 100;
    open(Movie)
end

%%%
tic
for t = 1:max_iterations
    random_sample_index_X = randperm(tarDimX, batch_size); % get indices to shuffle dataset to prevent the network from learning the input sequence pattern
    random_sample_index_Y = randperm(tarDimY, batch_size); % get indices to shuffle dataset to prevent the network from learning the input sequence pattern
    % Extract the random nonrepeating inputs for the current mini-batch
    Inputs = [bias_filler; x1_input(random_sample_index_X); x2_input(random_sample_index_Y); x3_input(random_sample_index_X); x4_input(random_sample_index_Y); x5_input(random_sample_index_X); x6_input(random_sample_index_Y); x7_input(random_sample_index_X); x8_input(random_sample_index_Y); x9_input(random_sample_index_X); x10_input(random_sample_index_Y); x11_input(random_sample_index_X); x12_input(random_sample_index_Y); x13_input(random_sample_index_X); x14_input(random_sample_index_Y)];

    % Feed forward part
    [Outputs, hiddenOutputs, Loss, Error] = feedForward(Inputs, tarim(sub2ind(tarim_size, random_sample_index_X, random_sample_index_Y)), weights_cell, hiddenOutputs, Num_Hidden_Layers, bias_filler, num_output_elements);

    % Calculate the the gradients to adjust the weights via backpropagation of the derivative of the loss function
    weights_gradients_cell_batch = backProp(Error, Inputs, Outputs, hiddenOutputs, weights_cell, Layer_Derivative, weights_gradients_cell_batch, Num_Hidden_Layers, bias_filler, batch_size);
    % check_gradients_mlp(1e-6, Inputs, tarim(sub2ind(tarim_size, random_sample_index_X, random_sample_index_Y)), weights_cell, hiddenOutputs, weights_gradients_cell_batch, Num_Hidden_Layers, bias_filler, num_output_elements)

    % Update weights
    for i = 1:Num_Forward_Steps
        % Compute the average gradients from the sum of gradients in the mini-batch
        average_mini_batch_gradients = weights_gradients_cell_batch{i};

        % Compute first and second moments to propagate information from previous gradients into the current update
        m_momentum_cell{i} = emv_momentum*m_momentum_cell{i}  +  emv_momentum_inv*average_mini_batch_gradients;
        v_squared_gradients_cell{i} = emv_sqgrad*v_squared_gradients_cell{i}  +  emv_sqgrad_inv*average_mini_batch_gradients.^2;

        % Bias correction to mitigate the bias towards the zero vector initialization
        m_momentum_hat = m_momentum_cell{i}./(1 - emv_momentum^t);
        v_squared_gradients_hat = v_squared_gradients_cell{i}./(1 - emv_sqgrad^t);

        % Update weights
        weights_cell{i} = weights_cell{i} - learn_rate./(sqrt(v_squared_gradients_hat) + eps) .* m_momentum_hat;
    end
    
    % Plot model output
    if mod(t, iter_plot_ratio)==0
        plot_index = plot_index + 1;
        
        % Get the image prediciton in one go braa via matrix pass
        [Image_Pred, ~, Loss, ~] = feedForward_tanh(Full_Image_Input, full_tarim, weights_cell, hiddenOutputs, Num_Hidden_Layers, bias_filler_test, mse_scaler); 
        loss_history(plot_index) = Loss; % save loss for plotting 

        % Update Plots
        img_handle.CData = reshape(Image_Pred, [tarDimY, tarDimX])';

        %update_weight_colors(line_cell, m_momentum_cell, Num_Hidden_Layers)
        % update_weight_colors(line_cell, weights_cell, Num_Hidden_Layers)
        % update_weight_colors(line_cell, weights_gradients_cell_batch, Num_Hidden_Layers)
        update_loss_evolution(loss_plt, plot_index, loss_history)

        % bar3(weights_handle, m_momentum_cell{2}');

        pause(0)

        % Capture and directly write frames for movie
        if Record_Video == 1
            fname = 'image_tiff2'; % full name of image
            print(fig1, '-dtiff','-r330',fname)     % save image with '-r200' resolution
            I = imread([fname '.tif']);       % read saved image
            writeVideo(Movie,im2frame(I))
        end
    end
end
fprintf('\n\t Vectorized Training Time: %.2f s \n\n', toc);

% Close video object
if Record_Video == 1
    close(Movie)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Feed-Forward with Tanh as activation function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Outputs, hiddenOutputs, Loss, Error] = feedForward_tanh(Inputs, Targets, weights_cell, hiddenOutputs, Num_Hidden_Layers, bias_filler, num_input_elements)
    
    % Pass the inputs into the first hidden layer
    hiddenOutputs{1} = tanh(weights_cell{1} * Inputs);  
    
    % Pass the activations of the first hidden layer throgh the  remaining hidden layers
    for i = 2:Num_Hidden_Layers
        % hiddenOutputs{i}  = tanh(weights_cell{i} * [1;hiddenOutputs{i-1}]);
        hiddenOutputs{i}  = tanh(weights_cell{i} * [bias_filler;hiddenOutputs{i-1}]); % simultaneous multi forward pass
    end
    
    % Pass the the activations from the last hidden layer to the output layer
    if Num_Hidden_Layers==0  % if there are no hidden layers, e.g perceptron
        Outputs = hiddenOutputs{1}; % the first hidden output, already computed above, is already the output
    else
        % Outputs = sigmoid(weights_cell{end} * [1;hiddenOutputs{end}]);
        Outputs = sigmoid(weights_cell{end} * [bias_filler;hiddenOutputs{end}]); % simultaneous multi forward pass
    end

    % Compute the loss via the difference between the actual targets and the predicted outputs, called the error
    Error = Targets - Outputs;
    Loss = 0.5*sum(Error.^2, 'all')/num_input_elements;  % compute scalar loss value as Mean Squared Error
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Backpropagation with tanh as hidden layer activation function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function weights_gradients_cell = backProp_tanh(Error, Inputs, Outputs, hiddenOutputs, weights_cell, Layer_Derivative, weights_gradients_cell, Num_Hidden_Layers, bias_filler, batch_size)
    
    % Compute the derivative at the output layer. Meaning the derivative of the loss function with respect to the output times the derivative of the output with respect to the final weighted sum
    Layer_Derivative{end} = -Error .* Outputs.*(1 - Outputs);  % dL/dy*dy/dz, Outputs*(1 - Outputs) is the derivative of the sigmoid activation function
    
    % Propagate the derivative/errors through the hidden layers to the input layer stating with the last hidden layer right behind the output layer
    for i = Num_Hidden_Layers:-1:1
        Layer_Derivative{i} = weights_cell{i+1}(:,2:end)'*Layer_Derivative{i+1} .* (1 - hiddenOutputs{i}.^2);
    end
    
    batch_normalizer = 1/batch_size;
    % Compute gradients of weights to map from input to first hidden layer. If there are no subsequent hidden layers then this first hidden layer is also the output layer.
    weights_gradients_cell{1} = batch_normalizer .* (Layer_Derivative{1} * Inputs'); % The one for the bias neuron is already inside. Was done outside this funciton Inputs = [1;Inputs]

    % Compute gradients of weights in hidden layer(s)
    for i = 2:Num_Hidden_Layers
        weights_gradients_cell{i} = batch_normalizer .* (Layer_Derivative{i} * [bias_filler;hiddenOutputs{i-1}]');
    end
    
    % Compute gradients for output layer (if hidden layers exist) 
    if Num_Hidden_Layers > 0  % If there are no hidden layers the output layer is the same as the first (hidden) layer. Equaly the last hidden layer is also always the output layer
        % Compute gradients of weights in output layer
        weights_gradients_cell{end} = batch_normalizer .* (Layer_Derivative{end} * [bias_filler;hiddenOutputs{end}]');
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Activation Functions and their derivatives
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = sigmoid(x)
y = 1./(1 + exp(-x));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%  % Figues for Images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [fig1, ay, img] = make_ll_canvas(targets, cmap)
% Figure to plot the current prediction of the image 
fig1 = figure('Position', [1350, 50, 800, 800], 'ToolBar','none');
ay = axes(fig1, 'Position', [0,0,1,1,]);
img = imshow(targets, 'Parent', ay);
colormap(ay, cmap)

% Figure for the input/target image to compare the progress
fig2 = figure('Position', [600, 50, 700, 600], 'ToolBar','none');
ax = axes(fig2, 'Position', [0,0,1,1,]);
imshow(targets, 'Parent', ax);
colormap(ax, cmap)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function apx = setup_weights_plot()
fig = figure('Position',[200,600,700,700]);
apx = axes(fig);
% hold(apx, 'on')

colormap(apx, magma)
apx.View = [45,10];
apx.ZLim = [-0.7, 0.7];

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%




% Set-up a figure for the neural network visualization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [neuro_handle, line_cell] = setup_NN_canvas(dense_NeuralNet_design)
% The following algorithm creates a figure with scatter points for the neurons as well as the in and outputs for the defined dense neural network.
% The parameter "b" defines the fraction of the the left and right side of the axis with limits [0,1] where the inputs and outputs are aligned. The hidden layers are then evenly spaced between the in and outputs
% The parameter "h" defines the fraction of the the bottom and top side of the axis to be left free. The neurons are then evenly spaced between the in and outputs

fig = figure('Position',[900,700,1500,600], 'ToolBar', 'none');
ax = axes(fig, 'Position',[0,0,1,1]);
hold(ax, 'on');

% The following algorithm creates a figure with scatter points for the neurons as well as the in and outputs for the defined dense neural network.
% The parameter "b" defines the fraction of the the left and right side of the axis with limits [0,1] where the inputs and outputs are aligned. The hidden layers are then evenly spaced between the in and outputs
% The parameter "h" defines the fraction of the the bottom and top side of the axis to be left free. The neurons are then evenly spaced between the in and outputs

num_neurons_per_layer = dense_NeuralNet_design;
num_neurons_per_layer(1:end-1) = dense_NeuralNet_design(1:end-1) + 1; % +1 to acocunt for the bias neuron
num_layers = numel(num_neurons_per_layer);
num_hidden_layers = num_layers - 2; % without input and output layer

%%%%%
% create the coordinates for the scatter points that represent the neurons 
b = 0.1; % side padding, must be a value between 0 and <0.5
h = 0.1; % height padding, must be a value between 0 and <0.5

layer_x_spacing = (1 - 2*b)/(num_hidden_layers + 1);

x_coordinate = zeros(sum(num_neurons_per_layer), 1);
y_coordinate = zeros(sum(num_neurons_per_layer), 1);

runner = 0;
for i = 1:num_layers
    for j = 1:num_neurons_per_layer(i)
        runner = runner + 1;
        x_coordinate(runner) = b + layer_x_spacing*(i-1);

        if num_neurons_per_layer(i) == 1
            y_coordinate(runner) = 0.5;
        else
            layer_y_spacing = (1 - 2*h)/(num_neurons_per_layer(i) - 1);
            y_coordinate(runner) = 1 - (h + layer_y_spacing*(j-1));
        end
    end

end

% make color matrix for neurons
index_bias = cumsum(num_neurons_per_layer(1:end-1)); % get the index of the bias neurons to plot them separately. makes coloring of the active neurons later easier
x_coordi_neuro = x_coordinate(setdiff(1:sum(num_neurons_per_layer), index_bias));
x_coordi_bias = x_coordinate(index_bias);
y_coordi_neuro = y_coordinate(setdiff(1:sum(num_neurons_per_layer), index_bias));
y_coordi_bias = y_coordinate(index_bias);



%%%%%
% create the lines that connect the neurons
linwidth = 1.2;
line_cell = cell(num_layers-1, 1);  % the handle for each line is stored in this nested cell. The top level contains the cells wich contain all lines connecting two layers, so there are num_layers-1 top level cells
for i = 1:num_layers-1
    runner_v2 = 0;
    line_cell{i} = cell(num_neurons_per_layer(i) * dense_NeuralNet_design(i+1), 1); % there are num_neurons_layer_k * (num_neurons_layer_k+1 - 1) layer-to-layer connections

    if i == num_layers-1
        by_pass_bias = 0;  % for the last layer where there is no bias we need to set this to 0 to avoid connecting the last output
    else
        by_pass_bias = 1; 
    end
  

    for j = 1:num_neurons_per_layer(i)
        for k = 1:num_neurons_per_layer(i+1) - by_pass_bias
            runner_v2 = runner_v2 + 1;
            if i == 1
                line_cell{i}{runner_v2} = line(ax, [x_coordinate(j), x_coordinate(index_bias(i) + k)], [y_coordinate(j), y_coordinate(index_bias(i) + k)], 'LineWidth', linwidth);
            else
                line_cell{i}{runner_v2} = line(ax, [x_coordinate(index_bias(i-1) + j), x_coordinate(index_bias(i) + k)], [y_coordinate(index_bias(i-1) + j), y_coordinate(index_bias(i) + k)], 'LineWidth', linwidth);
            end
        end
    end
end
%%%%%

% plot the neurons on the axis
neuro_handle = scatter(ax, x_coordi_neuro, y_coordi_neuro, 500, [0, 0.6, 0.9], 'filled', 'MarkerEdgeColor', 'w');
scatter(ax, x_coordi_bias, y_coordi_bias, 500, [1, 1, 0], 'filled', 'MarkerEdgeColor', 'w');  % since the bias neurons wont change color it is much easier to make a separate scatter object for them. 

ax.XLim = [0, 1];
ax.YLim = [0, 1];
ax.Color = 'k';
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function update_neuron_colors(dense_NeuralNet_design, Inputs, Outputs, hiddenOutputs, neuro_handle, Num_Hidden_Layers, n_inputs, n_outputs)
    % function to assign colors to the cooresponding lines and neurons
    neuron_state_vector = zeros(sum(dense_NeuralNet_design), 1);

    for i = 1:Num_Hidden_Layers + 2
        index_bias = cumsum(dense_NeuralNet_design);
        if i == 1
            neuron_state_vector(1:n_inputs) = Inputs;
        elseif i == Num_Hidden_Layers + 2
            neuron_state_vector(end-n_outputs + 1:end) = Outputs;
        else
            neuron_state_vector(index_bias(i-1)+1:index_bias(i)) = hiddenOutputs{i-1};
        end
    end
    neuro_handle.CData = redblueColormap(neuron_state_vector);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function update_weight_colors(line_cell, weights_gradients_cell, Num_Hidden_Layers)
    % function to assign colors to the lines/weights, they are colored relative to each other, no grouping by layers. should make differences in the magnitude of the weight across layers visible
    num_weights = zeros(numel(line_cell), 1);
    for i = 1:numel(line_cell)
        num_weights(i) = numel(line_cell{i});
    end
    num_weights = cumsum(num_weights);
    
    weights_state_vector = zeros(num_weights(end), 1);

    for i = 1:Num_Hidden_Layers + 1
        if i == 1
            weights_state_vector(1:num_weights(i)) = weights_gradients_cell{i}(:);
        else
            weights_state_vector(num_weights(i-1) + 1:num_weights(i)) = weights_gradients_cell{i}(:);
        end
    end
    
    % Get colors
    weights_color_vector = redblueColormap(weights_state_vector);
    
    % Apply each color to the desired line
    runner = 0;
    for i = 1:numel(line_cell)
        for j = 1:numel(line_cell{i})
            runner = runner + 1;
            line_cell{i}{j}.Color = weights_color_vector(runner, :);
        end
    end
    %%%%%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Loss-Evolution 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function loss_plt = setup_loss_evolution_canvas(iter_plot_ratio)
% Initialize plot for the loss evolution
fig = figure('Position', [50, 700, 840, 490], 'Color', 'k');
ay = axes(fig, 'Position', [0.12, 0.13, 0.82, 0.84]);

loss_plt = plot(ay, 0, 0, 'Color', 'c', 'LineWidth', 2);

ay.FontSize = 15;
ay.XLabel.String = ['Number of mini-batches / weight updates [', num2str(iter_plot_ratio), 'x]'];
ay.YLabel.String = 'Loss';
ay.XLabel.FontSize = 16;
ay.YLabel.FontSize = 16;
ay.XColor = 'y';
ay.YColor = 'y';
ay.Color = 'k';

ay.XGrid = 'on';
ay.YGrid = 'on';
ay.YAxis.Scale = 'log';

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function update_loss_evolution(loss_plt, e, loss_history)
loss_plt.XData = 1:e;
loss_plt.YData = loss_history(1:e);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
