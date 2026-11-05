clear
close all

% Define the number of neurons per layer
n_inputs = 2;
n_outputs = 1;
dense_NeuralNet_design = [n_inputs, 10, 10, 10, n_outputs];  % [Input, Hidden_1, Hidden_2, ..., Output]

Num_Hidden_Layers  = numel(dense_NeuralNet_design) - 2;  % -2 to remove input and output layer from consideration
Num_Forward_Steps  = Num_Hidden_Layers + 1;  % we need to perform n-1 matrix multiplications + activation function to do one forward pass where n is the number of hidden layers + input and output layer, so n is total layers

% Initialize containers for weights, gradients and outputs at each layer
weights_cell = initialize_weights(dense_NeuralNet_design, 'xavier', 0);
weights_set_to_zero_cell = initialize_weights(dense_NeuralNet_design, 'zeros', 0);
weights_gradients_cell = weights_set_to_zero_cell;
weights_gradients_cell_batch = weights_set_to_zero_cell;  % cell to store the sum of gradients of a mini-batch. After the gradients of each training exaple in the mini-batch has been summed up the average is formed 

m_momentum_cell = weights_set_to_zero_cell; % cell array to hold the exponential moving average momentum vectors (first moment) 
v_squared_gradients_cell = weights_set_to_zero_cell; % cell array to hold the squared gradients (second moment) 

hiddenOutputs = cell(Num_Hidden_Layers, 1);
hiddenOutputs_withBias = cell(Num_Hidden_Layers, 1);
Layer_Derivative = cell(Num_Forward_Steps, 1);

% training data
n = 1000;
lb = -6;
ub =  6;

freq = 1.5;
x_inputs1 = linspace(lb, ub, n);
x_inputs2 = x_inputs1.^2;
x_inputs = [x_inputs1;x_inputs2];

x_test1 = linspace(lb, ub, n/10);
x_test2 = x_test1.^2;
x_test = [x_test1;x_test2];


targets = (sin(freq*pi*x_inputs1)+1).*1.5 + 0.1*x_inputs1;
y_test = ones(n/10, 1);
y_test_true = (sin(freq*pi*x_test1)+1).*1.5 + 0.1*x_test1;

% Input scaling
x_inputs_avg = mean(x_inputs, 2);
x_test_avg   = mean(x_test, 2);

x_inputs_std = std(x_inputs, 0, 2);
x_test_std = std(x_test, 0, 2);

x_inputs = (x_inputs - x_inputs_avg)./x_inputs_std;
x_test = (x_test - x_test_avg)./x_test_std;
x_test_withBias = [ones(1, length(x_test1)); x_test];


% Presets and hyperparameters
max_epochs = 1000000;
iter_plot_ratio = 1000;
learn_rate = 0.002;
emv_momentum = 0.9; % define what fraction of the previous gradient should be used for the current gradient. Only the remaining part, 1-momentum_share, is contributed by the current gradient. Is the beta term here: m_i+1 = beta*m_i + (1-beta)*gradients 
emv_sqgrad = 0.999; % Exponential Moving Average coefficient. Is the beta term here: v_i+1 = beta*v_i + (1-beta)*gradients.^2 
batch_size = 16;

emv_momentum_inv = 1-emv_momentum; % same idea as for above. save a computation in this case the scalar that defines the fraction of the current gradient used. depending on cache is bad mkay
emv_sqgrad_inv = 1-emv_sqgrad; % same idea as for above. save a computation in this case the scalar that defines the fraction of the current gradient used. depending on cache is bad mkay
eps = 1e-8; % for numerical stability. To avoid dividing by zero when rescaling the learing rate per parameter

batch_loss = 0;
loss_history = zeros(max_epochs, 1);  % container to plot the evolution of the loss

% Set-up plot figure
output_handle = make_ll_canvas(x_inputs, x_test, targets, y_test);
weights_handle = setup_weights_plot();
[neuro_handle, line_cell] = setup_NN_canvas(dense_NeuralNet_design);

%%%
tic
for t = 1:max_epochs
    random_sample_index = randperm(n, batch_size); % get indices to shuffle dataset to prevent the network from learning the input sequence pattern
    for i = 1:batch_size
        Inputs = x_inputs(:,random_sample_index(i));
        % For performance reasons the following helper variables are comupted ONCE and then passed to the feedForward() and backProp() funcitons to avoid 
        Inputs = [1;Inputs];
        
        % Feed forward part
        [Outputs, hiddenOutputs, Loss, Error] = feedForward(Inputs, targets(random_sample_index(i)), weights_cell, hiddenOutputs, Num_Hidden_Layers);
       
        % calculate the the gradients to adjust the weights via backpropagation of the derivative of the loss function
        weights_gradients_cell = backProp(Error, Inputs, Outputs, hiddenOutputs, weights_cell, Layer_Derivative, weights_gradients_cell, Num_Hidden_Layers);
        %check_gradients(1e-6, Inputs, weights_cell, hiddenOutputs, Num_Hidden_Layers, weights_gradients_cell, targets(random_sample_index(i)))

        % Compute the sum of the gradients of the weights for each layer for the current batch
        for p = 1:Num_Forward_Steps  % the number of forward steps is equal to the number of weight matrices because each forward step involves first a matrix multipliaction of said weights with the activations of the previous layer
            weights_gradients_cell_batch{p} = weights_gradients_cell_batch{p} + weights_gradients_cell{p};
        end

        % sum up the loss for each training instance
        batch_loss = batch_loss + Loss;
    end

    % Update weights
    for i = 1:Num_Forward_Steps
        % Compute the average gradients from the sum of gradients in the mini-batch
        average_mini_batch_gradients = weights_gradients_cell_batch{i}./batch_size;

        % Compute first and second moments to propagate information from previous gradients into the current update
        m_momentum_cell{i} = emv_momentum*m_momentum_cell{i}  +  emv_momentum_inv*average_mini_batch_gradients;
        v_squared_gradients_cell{i} = emv_sqgrad*v_squared_gradients_cell{i}  +  emv_sqgrad_inv*average_mini_batch_gradients.^2;

        % Bias correction to mitigate the bias towards the zero vector initialization
        m_momentum_hat = m_momentum_cell{i}./(1 - emv_momentum^t);
        v_squared_gradients_hat = v_squared_gradients_cell{i}./(1 - emv_sqgrad^t);

        % Update weights
        weights_cell{i} = weights_cell{i} - learn_rate./(sqrt(v_squared_gradients_hat) + eps) .* m_momentum_hat;
    end
    
    % Visualize loss evolution
    loss_history(t) = batch_loss/batch_size;
  
    % Plot model output
    if mod(t,200)==0

        update_weight_colors(line_cell, m_momentum_cell, Num_Hidden_Layers)
        % update_weight_colors(line_cell, weights_cell, Num_Hidden_Layers)

        for i = 1:100
            y_test(i) = feedForward(x_test_withBias(:, i), y_test_true(i), weights_cell, hiddenOutputs, Num_Hidden_Layers);
        end
        output_handle.YData = y_test;

        % bar3(weights_handle, weights_momentum_cell{2}');
        % zlim(weights_handle, [-0.07, 0.07]);
        % colormap(weights_handle, plasma);

        bar3(weights_handle, weights_gradients_cell_batch{2}');
        zlim(weights_handle, [-0.2, 0.2]);
        colormap(weights_handle, plasma);

        % imagesc(weights_momentum_cell{2}')
        % view([-10,10])
        pause(0)
    end

    % Reset the total loss and accumulated gradients of the current batch
    batch_loss = 0;
    weights_gradients_cell_batch = weights_set_to_zero_cell;
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
    % hiddenOutputs{1} = sigmoid(weightsCell{1} * [1;Inputs]);
    hiddenOutputs{1} = tanh(weights_cell{1} * Inputs);  
    
    % Pass the activations of the first hidden layer throgh the  remaining hidden layers
    for i = 2:Num_Hidden_Layers
        % hiddenOutputs{i}  = sigmoid(weights_cell{i} * [1;hiddenOutputs{i-1}]);
        hiddenOutputs{i}  = tanh(weights_cell{i} * [1;hiddenOutputs{i-1}]);
    end
    
    % Pass the the activations from the last hidden layer to the output layer
    if Num_Hidden_Layers==0  % if there are no hidden layers, e.g perceptron
        Outputs = hiddenOutputs{1}; % the first hidden output, already computed above, is already the output
    else
        % Outputs = sigmoid(weightsCell{end} * [1;hiddenOutputs{end}]);
        Outputs = (weights_cell{end} * [1;hiddenOutputs{end}]); % no final activation function
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
    %Layer_Derivative{end} = -Error * Outputs.*(1 - Outputs);  % dL/dy*dy/dz, Outputs*(1 - Outputs) is the derivative of the sigmoid activation function
    Layer_Derivative{end} = -Error;  % this is for when no activation function was used for the last layer, intended for regression problems
    
    % Propagate the derivative/errors through the hidden layers to the input layer stating with the last hidden layer right behind the output layer
    for i = Num_Hidden_Layers:-1:1
        % Layer_Derivative{i} = weights_cell{i+1}(:,2:end)'*Layer_Derivative{i+1} .* hiddenOutputs{i}.*(1 - hiddenOutputs{i});
        Layer_Derivative{i} = weights_cell{i+1}(:,2:end)'*Layer_Derivative{i+1} .* (1 - hiddenOutputs{i}.^2);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function output_handle = make_ll_canvas(x_inputs, x_test, targets, y_test)
fig = figure('Position',[1000,50,1100,600], 'Color', 'w');
atx = axes(fig, 'Position', [0.06, 0.12, 0.9, 0.87]);
hold(atx, 'on')


plot(atx, x_inputs(1, :), targets, 'LineWidth', 4, 'Color', 'r');
output_handle = plot(atx, x_test(1, :)', y_test, 'LineWidth', 2, 'Color', 'b');
atx.XLabel.String = 'x - input';    atx.XLabel.FontSize = 20;
atx.YLabel.String = 'y - output';   atx.YLabel.FontSize = 22;

range = max(targets) - min(targets);
atx.YLim = [min(targets) - 0.05*range, max(targets) + 0.05*range];
atx.Box = 'on';
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function apx = setup_weights_plot()
fig = figure('Position',[200,400,700,700]);
apx = axes(fig, 'Position', [0.07, 0, 0.88, 1]);
% apx.View = [45,10];
end



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




% Set-up figure for neural network visualization
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
        % pause(0)
    end
    %%%%%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
