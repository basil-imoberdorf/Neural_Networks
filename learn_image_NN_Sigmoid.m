clear
close all

% Define input and output dimensions as well as the number of neurons per layer
n_inputs = 10;
n_outputs = 1;
dense_NeuralNet_design = [n_inputs, 32, 32, 32, 32, n_outputs];  % [Input, Hidden_Layer_1, Hidden_Layer_2, ... , Hidden_Layer_n, Output]

Num_Hidden_Layers  = numel(dense_NeuralNet_design) - 2;  % -2 to remove input and output layer from consideration
Num_Forward_Steps  = Num_Hidden_Layers + 1;  % we need to perform n-1 matrix multiplications + activation function to do one forward pass where n is the number of hidden layers + input and output layer, so n is total layers

% Initialize containers for weights, gradients and outputs at each layer
weights_cell = initialize_weights(dense_NeuralNet_design, 'xavier', 0);
weights_gradients_cell = initialize_weights(dense_NeuralNet_design, 'zeros', 0);
weights_gradients_cell_batch = weights_gradients_cell;
weights_momentum_cell = weights_gradients_cell;
weights_set_to_zero_cell = weights_gradients_cell;

hiddenOutputs = cell(Num_Hidden_Layers, 1);
Layer_Derivative = cell(Num_Forward_Steps, 1);

% Import and show training image
tarim = imread('Images\euler.JPG');
tarim = compImag(tarim, 12, 'avg');  % compress image to reduce dimensionality
tarim = mean(tarim, 3)./255;  % convert color image to gray-scale
[tarDimX, tarDimY] = size(tarim);  % dimensions of image
cmap = magma; % <-- chose colormap

% % artifical grayscale "image"
% quadrant = [zeros(10,10),ones(10,10);ones(10,10),zeros(10,10)];
% tarim = repmat(quadrant, 3); % chess board
% tarDim = size(tarim);

% Feature engineering: Computing Fourier features
x1_input  = ((0:tarDimX)./tarDimX) * 2*pi - pi;
x2_input  = ((0:tarDimY)./tarDimY) * 2*pi - pi;
x3_input  = sin(x1_input);       
x4_input  = sin(x2_input);       
x5_input  = cos(x1_input);       
x6_input  = cos(x2_input);
x7_input  = sin(2*x1_input);
x8_input  = sin(2*x2_input);
x9_input  = cos(2*x1_input);
x10_input = cos(2*x2_input);   

% Presets and hyperparameters
max_epochs = 1000000;
iter_plot_ratio = 1000;
learn_rate = 1;
momentum_share = 0.99; % define what fraction of the previous gradient should be used for the current gradient. Only the remaining part, 1-momentum_share, is contributed by the current gradient
batch_size = 4;

momentum_share_gradient = 1-momentum_share; % same idea as for above. save a computation in this case the scalar that defines the fraction of the current gradient used. depending on cache is bad mkay
gradient_scaler = learn_rate/batch_size; % this variabl combines the constant relation that is applied to each weight update. This saves a divison. maybe it helps a bit in performance
batch_loss = 0;
Loss_sum = 0;
plot_index = 0;  % to use as index to keep track of the loss history, allows pre-allocate a array of the correct size given the iter_plot_ratio. avoids over allocating when iter_plot_ratio > 1
loss_history = zeros(max_epochs/iter_plot_ratio, 1);  % container to plot the evolution of the loss
mse_scaler = tarDimX*tarDimY;  % the sum of squared errors is divided by this number to get the MEAN squared error
test_image = zeros(tarDimX, tarDimY); % pre-allocate

% Set-up plot figure
%weights_handle = setup_weights_plot(weights_cell{1}');
[ay, img_handle] = make_ll_canvas(tarim, cmap);
% [neuro_handle, line_cell] = setup_NN_canvas(dense_NeuralNet_design);
loss_plt = setup_loss_evolution_canvas(iter_plot_ratio);

%%%
tic
for e = 1:max_epochs
    random_sample_index_X = randperm(tarDimX, batch_size); % get indices to shuffle dataset to prevent the network from learning the input sequence pattern
    random_sample_index_Y = randperm(tarDimY, batch_size); % get indices to shuffle dataset to prevent the network from learning the input sequence pattern
    for i = 1:batch_size
        index_x = random_sample_index_X(i);
        index_y = random_sample_index_Y(i);
        % Inputs = [1; x1_input(index_x); x2_input(index_y)];
        % Inputs = [1; x1_input(index_x); x2_input(index_y); x3_input(index_x); x4_input(index_y); x5_input(index_x); x6_input(index_y)];
        Inputs = [1; x1_input(index_x); x2_input(index_y); x3_input(index_x); x4_input(index_y); x5_input(index_x); x6_input(index_y); x7_input(index_x); x8_input(index_y); x9_input(index_x); x10_input(index_y)];
        
        % Feed forward part
        [Outputs, hiddenOutputs, Loss, Error] = feedForward(Inputs, tarim(index_x, index_y), weights_cell, hiddenOutputs, Num_Hidden_Layers);
       
        % Calculate the the gradients to adjust the weights via backpropagation of the derivative of the loss function
        weights_gradients_cell = backProp(Error, Inputs, Outputs, hiddenOutputs, weights_cell, Layer_Derivative, weights_gradients_cell, Num_Hidden_Layers);
        %check_gradients(1e-6, Inputs, weights_cell, hiddenOutputs, Num_Hidden_Layers, weights_gradients_cell, targets(random_sample_index(i)))

        % Compute the sum of the gradients of the weights for each layer for the current batch
        for p = 1:Num_Forward_Steps  % the number of forward steps is equal to the number of weight matrices because each forward step involves first a matrix multipliaction of said weights with the activations of the previous layer
            weights_gradients_cell_batch{p} = weights_gradients_cell_batch{p} + weights_gradients_cell{p};
        end
    end

    % Update weights
    for i = 1:Num_Forward_Steps
        weights_momentum_cell{i} = (momentum_share.*weights_momentum_cell{i} + momentum_share_gradient*weights_gradients_cell_batch{i});
        weights_cell{i} = weights_cell{i} - gradient_scaler .* weights_momentum_cell{i};
    end

    % Reset the accumulated gradients of the current batch
    weights_gradients_cell_batch = weights_set_to_zero_cell;
    
    % Plot model output
    if mod(e, iter_plot_ratio)==0
        plot_index = plot_index + 1;
        for i = 1:tarDimX
            for j = 1:tarDimY
                % Inputs = [1; x1_input(i); x2_input(j)];
                % Inputs = [1; x1_input(i); x2_input(j); x3_input(i); x4_input(j); x5_input(i); x6_input(j)];
                Inputs = [1; x1_input(i); x2_input(j); x3_input(i); x4_input(j); x5_input(i); x6_input(j); x7_input(i); x8_input(j); x9_input(i); x10_input(j)];
                [test_image(i, j), ~, Loss, ~] = feedForward(Inputs, tarim(i, j), weights_cell, hiddenOutputs, Num_Hidden_Layers);
                Loss_sum = Loss_sum + Loss; % sum up the loss to estimate the average loss over the whole image           
            end
        end
        loss_history(plot_index) = Loss_sum/mse_scaler;
        Loss_sum = 0; % reset loss for next run
        
        % Update Plots
        img_handle.CData = test_image;

        % update_weight_colors(line_cell, weights_momentum_cell, Num_Hidden_Layers)
        % update_weight_colors(line_cell, weights_cell, Num_Hidden_Layers)
        update_loss_evolution(loss_plt, plot_index, loss_history)

        % bar3(weights_handle, weights_momentum_cell{2}');
        % zlim([-0.07, 0.07]);
        % colormap(gca, plasma);

        % bar3(weights_handle, weights_cell{2}');
        % zlim([-0.5, 0.5]);

        pause(0)
    end
end
toc


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize weights 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function weights_cell = initialize_weights(NeuralNet_design_vector, method, initial_bias_weight)
% This function returns random weights as starting point for training. Each matrices contains also weights for the bias term
% Arguments:
% NeuralNet_design_vector: A vector of integers that defines with the number of elements how many layers the network shoul have the value of each entry defines how many nodes the layer has 
% method: A string that defines the weight initialization methdos. Must be one of the following:
%   'gauss':  draw weights randomly from a standard normal distribution, W∼N(0,1) 
%   'xavier': normalize the weights ensures that the variance of activations remains consistent across layers W∼N(0,sqrt(2/(n_in + n_out + 1))
%   'He':     draw weights randomly from a standard normal distribution, W∼N(0,sqrt(2/(n_in + 1))
%   'zeros':  set all weights to zero. Not used for training but to initalize a cell for restetting the summed gradients cell back to zero before starting the next epoch
% initial_bias_weight: A real number that defines how the bias weights are initialized. 0 is recommended for sigmoid, tanh, softsign etc. For ReLU small positive biases might be preferred because they can help activate neurons early in training and avoid the "dead neuron" problem.

% Pre-allocate cell array. This is a minor performance boost mkay^^
weights_cell = cell(length(NeuralNet_design_vector) - 1, 1);

% Loop between each layer to initilaze the weights iteratively. [I-W1->H1-W2->...-Wn->O]
for i = 2:length(NeuralNet_design_vector)

    n_in = NeuralNet_design_vector(i-1); % number of input neurons = number of neurons in the previous layer
    n_out = NeuralNet_design_vector(i);  % number of output neurons = number of neurons in the next layer

    switch method
        case 'gauss'  % naive method
            weights_cell{i-1} = [initial_bias_weight*ones(n_out, 1), randn(n_out, n_in)]; % combine weights for the bias, initialized at 0, and weights for the activations/inputs

        case 'xavier' % for activation function like sigmoid() and tanh() 
            weights_cell{i-1} = [initial_bias_weight*ones(n_out, 1), randn(n_out, n_in)] .* sqrt(2 / (n_out + n_in));  

        case 'He'  % use this initialization in combination with the ReLU activation function
            weights_cell{i-1} = [initial_bias_weight*ones(n_out, 1), randn(n_out, n_in)] .* sqrt(2 / n_in);  

        case 'lecun' % used for tanh(). Is the same as xavier if n_in = n_out
            weights_cell{i-1} = [initial_bias_weight*ones(n_out, 1), randn(n_out, n_in)] * sqrt(1 / n_in);

        case 'zeros'  % Not for training! Only used to get matrices with zeros in the shape of the weigth matrices
            weights_cell{i-1} = zeros(n_out, n_in + 1);  % + 1 is for the bias term
    end
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Feed-Forward 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Outputs, hiddenOutputs, Loss, Error] = feedForward(Inputs, Targets, weights_cell, hiddenOutputs, Num_Hidden_Layers)
    
    % Pass the inputs into the first hidden layer
    hiddenOutputs{1} = sigmoid(weights_cell{1} * Inputs);  
    
    % Pass the activations of the first hidden layer throgh the  remaining hidden layers
    for i = 2:Num_Hidden_Layers
        hiddenOutputs{i}  = sigmoid(weights_cell{i} * [1;hiddenOutputs{i-1}]);
    end
    
    % Pass the the activations from the last hidden layer to the output layer
    if Num_Hidden_Layers==0  % if there are no hidden layers, e.g perceptron
        Outputs = hiddenOutputs{1}; % the first hidden output, already computed above, is already the output
    else
        Outputs = sigmoid(weights_cell{end} * [1;hiddenOutputs{end}]);
    end

    % Compute the loss from the targets and the outputs
    Error = Targets - Outputs;
    Loss = 0.5*sum(Error.^2);
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Backpropagation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function weights_gradients_cell = backProp(Error, Inputs, Outputs, hiddenOutputs, weights_cell, Layer_Derivative, weights_gradients_cell, Num_Hidden_Layers)
    
    % Compute the error of the output layer. Meaning the derivative of the loss function times the derivative of the final activation function
    Layer_Derivative{end} = -Error * Outputs.*(1 - Outputs);  % dL/dy*dy/dz, Outputs*(1 - Outputs) is the derivative of the sigmoid activation function
    
    % Propagate the derivative/errors through the hidden layers stating with the last hidden layer right behind the output layer
    for i = Num_Hidden_Layers:-1:1
        Layer_Derivative{i} = weights_cell{i+1}(:,2:end)'*Layer_Derivative{i+1} .* hiddenOutputs{i}.*(1 - hiddenOutputs{i});
    end
    
    % Compute gradients of weights to map from input to first hidden layer. If there are no subsequent hidden layers then this first hidden layer is also the output layer.
    weights_gradients_cell{1} = Layer_Derivative{1} * Inputs'; % The bias neuron is already the inside. No need for [1;Inputs]

    % Compute gradients of weights in hidden layer(s)
    for i = 2:Num_Hidden_Layers
        weights_gradients_cell{i} = Layer_Derivative{i} * [1;hiddenOutputs{i-1}]';
    end
    
    % Compute gradients for output layer (if hidden layers exist) 
    if Num_Hidden_Layers > 0  % If there are no hidden layers the output layer is the same as the first (hidden) layer. Equaly the last hidden layer is also always the output layer
        % Compute gradients of weights in output layer
        weights_gradients_cell{end} = Layer_Derivative{end} * [1;hiddenOutputs{end}]';
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
function [ay, img] = make_ll_canvas(targets, cmap)
% Figure to plot the current prediction of the image 
fig1 = figure('Position', [1350, 50, 700, 600], 'ToolBar','none');
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


% Gradient Checking to verfiy backpropagation implementation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function check_gradients(epsilon, Inputs, weights_cell, hiddenOutputs, Num_Hidden_Layers, weights_gradients_cell, Targets)

% This function computes nummerically the gradients and compares them with the gradients obtained by backpropagation. If the match the backpropagation algorithm works as intended.
% First copy the weight matrices to manipulate them separately
numerical_weights_gradients_cell = weights_gradients_cell;

for k = 1:Num_Hidden_Layers + 1 
    weights_cell_positive_perturbed = weights_cell;
    weights_cell_negative_perturbed = weights_cell;
    dim = size(weights_cell{k});
    for i = 1:dim(1)
        for j = 1:dim(2)
            % Perturb each paramter one at a time and compute the loss
            weights_cell_positive_perturbed{k}(i,j) = weights_cell{k}(i,j) + epsilon;
            weights_cell_negative_perturbed{k}(i,j) = weights_cell{k}(i,j) - epsilon;

            [~, ~, Loss_positive_perturbed, ~] = feedForward(Inputs, Targets, weights_cell_positive_perturbed, hiddenOutputs, Num_Hidden_Layers);
            [~, ~, Loss_negative_perturbed, ~] = feedForward(Inputs, Targets, weights_cell_negative_perturbed, hiddenOutputs, Num_Hidden_Layers);
            
            % Compute the numerical approximation of the derivative for the corresponding weight
            numerical_weights_gradients_cell{k}(i,j) = (Loss_positive_perturbed - Loss_negative_perturbed)/(2*epsilon);

            % Reset the weights for the next round to not carry over disturbances from previous tests
            weights_cell_positive_perturbed = weights_cell;
            weights_cell_negative_perturbed = weights_cell;
        end
    end

    gradient_differences = (numerical_weights_gradients_cell{k} - weights_gradients_cell{k})
    percent_error = (numerical_weights_gradients_cell{k} - weights_gradients_cell{k})./numerical_weights_gradients_cell{k} * 100
end
end




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
                line_cell{i}{runner_v2} = line(ax, [x_coordinate(j), x_coordinate(index_bias(i) + k)], [y_coordinate(j), y_coordinate(index_bias(i) + k)], 'LineWidth', 2);
            else
                line_cell{i}{runner_v2} = line(ax, [x_coordinate(index_bias(i-1) + j), x_coordinate(index_bias(i) + k)], [y_coordinate(index_bias(i-1) + j), y_coordinate(index_bias(i) + k)], 'LineWidth', 2);
            end
        end
    end
end
%%%%%

% plot the neurons on the axis
neuro_handle = scatter(ax, x_coordi_neuro, y_coordi_neuro, 500, 'filled', 'MarkerEdgeColor', 'w');
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
