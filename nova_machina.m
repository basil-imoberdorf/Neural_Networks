clear
close all

% Define the number of neurons per layer
n_inputs = 2;
dense_NeuralNet_design = [n_inputs, 2, 1];  % [Input, Hidden_1, Hidden_2, ..., Output]

Num_Hidden_Layers  = numel(dense_NeuralNet_design) - 2;  % -2 to remove input and output layer from consideration
Num_Forward_Steps  = Num_Hidden_Layers + 1;  % we need to perform n-1 matrix multiplications + activation function to do one forward pass where n is the number of hidden layers + input and output layer, so n is total layers

% Initialize containers for weights, gradients and outputs at each layer
weights_cell = init_weights(dense_NeuralNet_design, 'He');
weights_gradients_cell = init_weights(dense_NeuralNet_design, 'zeros');
weights_gradients_cell_total = weights_gradients_cell;
weights_set_to_zero_cell = weights_gradients_cell;

hiddenOutputs = cell(Num_Hidden_Layers, 1);
Layer_Derivative = cell(Num_Forward_Steps, 1);

% Hyperparameters for learning
learn_rate = 0.1;
batch_size = 4;
max_epochs = 50000;
epochal_loss = 0;
loss_history = zeros(batch_size, 1);  % container to plot the evolution of the loss 


% XOR training data
x1 = [1,1,0,0];
x2 = [1,0,1,0];
Targets = [0,1,1,0];
x_inputs = [x1; x2];



% % Initialize plot for the prediction
% fig = figure('Position', [800, 700, 1200, 500], 'Color', 'w');
% ax = axes(fig, 'Position', [0.1, 0.1, 0.88, 0.88]);
% hold(ax, 'on')

% plot(ax, x, Targets, 'Color', 'k', 'LineWidth', 2);  % True curve
% pred_plt = plot(ax, x, y_pred, 'Color', 'r', 'LineWidth', 2);  % prediciton curve by NN
% ax.XLabel.String = 'Predictior';  ax.XColor = 'k';
% ax.YLabel.String = 'Response';    ax.YColor = 'k';

% Initialize plot for the loss evolution
fig = figure('Position', [800, 100, 1200, 500], 'Color', 'w');
ay = axes(fig, 'Position', [0.1, 0.1, 0.88, 0.88]);

loss_plt = plot(ay, 0, 0, 'Color', 'r', 'LineWidth', 2);
ay.XLabel.String = 'Number of iterations';
ay.YLabel.String = 'Loss';
% ay.YAxis.Scale = 'log';

% random_sample_index = [1,2,3,4];
% Let's go
tic
for e = 1:max_epochs
    random_sample_index = randperm(4); % get indices to shuffle dataset to prevent the network from learning the input sequence pattern
    for i = 1:batch_size
        Inputs = x_inputs(:,random_sample_index(i));
        
        % Feed forward part
        [Outputs, hiddenOutputs] = feedForward(Inputs, weights_cell, hiddenOutputs, Num_Hidden_Layers);

        % calculate the the gradients to adjust the weights via backpropagation of the derivative of the loss function 
        [weights_gradients_cell, loss] = backProp(Inputs, hiddenOutputs, Outputs, Targets(random_sample_index(i)), weights_cell, Layer_Derivative, weights_gradients_cell, learn_rate, Num_Hidden_Layers);
        
        % Compute the sum of the gradients of the weights for each layer for the current batch
        for p = 1:Num_Forward_Steps  % the number of forward steps is equal to the number of weight matrices because each forward step involves first a matrix multipliaction of said weights with the activations of the previous layer
            weights_gradients_cell_total{p} = weights_gradients_cell_total{p} + weights_gradients_cell{p};
        end
        
        % sum up the loss for each training instance
        epochal_loss = epochal_loss + loss;

    end

    % Update weights in output layer
    for i = 1:Num_Forward_Steps
        weights_cell{i} = weights_cell{i} - learn_rate/batch_size .* weights_gradients_cell_total{i};
    end
    loss_history(e) = epochal_loss/batch_size;

    % Visualize loss evolution
    if mod(e, 100)==0 % update the plot every nth epoch
        loss_plt.XData = 1:e;
        loss_plt.YData = loss_history(1:e);
        pause(0) % make the changes to the plot's data become visible
    end

    % Reset the total loss and accumulated gradients of the current epoch
    epochal_loss = 0;
    weights_gradients_cell_total = weights_set_to_zero_cell;
end
training_time = toc;
fprintf("\n +----------------+ \n | Training Stats | \n +----------------+ \n");
fprintf(" | Epochs: %s  | \n +----------------+ \n",  num2str(round(max_epochs, 2)));
fprintf(" | Batch Size: %s  | \n +----------------+ \n",  num2str(round(batch_size, 2)));
fprintf(" | %s seconds   | \n +----------------+ \n",  num2str(round(training_time, 2)));

y1 = feedForward([0;0], weights_cell, hiddenOutputs, Num_Hidden_Layers);
y2 = feedForward([1;0], weights_cell, hiddenOutputs, Num_Hidden_Layers);
y3 = feedForward([0;1], weights_cell, hiddenOutputs, Num_Hidden_Layers);
y4 = feedForward([1;1], weights_cell, hiddenOutputs, Num_Hidden_Layers);

fprintf("\n +-----------------------------+ \n |          XOR Test           | \n +-----------------------------+ \n");
fprintf(" | Input | Target | Prediction | \n");
fprintf(" +-----------------------------+ \n")
fprintf(" | [0,0] |    0   |     %s   | \n", num2str(round(y1,2)));
fprintf(" +-----------------------------+ \n")
fprintf(" | [1,0] |    1   |     %s   | \n", num2str(round(y2,2)));
fprintf(" +-----------------------------+ \n")
fprintf(" | [0,1] |    1   |     %s   | \n", num2str(round(y3,2)));
fprintf(" +-----------------------------+ \n")
fprintf(" | [1,1] |    0   |     %s   | \n", num2str(round(y4,2)));
fprintf(" +-----------------------------+ \n\n")

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize weights 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function weights_cell = init_weights(NeuralNet_design_vector, method)
% This function returns random weights as starting point for training. Each matrices contains also weights for the bias term
% Arguments:
% NeuralNet_design_vector: A vector of integers that defines with the number of elements how many layers the network shoul have the value of each entry defines how many nodes the layer has 
% method: A string that defines the weight initialization methdos. Must be one of the following:
%   'gauss':  draw weights randomly from a standard normal distribution, W∼N(0,1) 
%   'xavier': normalize the weights ensures that the variance of activations remains consistent across layers W∼N(0,sqrt(2/(n_in + n_out + 1))
%   'He':     draw weights randomly from a standard normal distribution, W∼N(0,sqrt(2/(n_in + 1))
%   'zeros':  set all weights to zero. Not used for training but to initalize a cell for restetting the summed gradients cell back to zero before starting the next epoch

weights_cell = cell(length(NeuralNet_design_vector) - 1, 1);
for i = 2:length(NeuralNet_design_vector)

    n_in = NeuralNet_design_vector(i-1); % number of input neurons = number of neurons in the previous layer
    n_out = NeuralNet_design_vector(i);  % number of output neurons = number of neurons in the next layer

    switch method
        case 'gauss'  % naive method
            weights_cell{i-1} = randn(n_out, n_in + 1);  % + 1 is for the bias term

        case 'xavier' % for sigmoid() and tanh() activation functions
            weights_cell{i-1} = randn(n_out, n_in + 1) .* sqrt(2 / (n_out + n_in + 1));  % + 1 is for the bias term

        case 'He'  % use this initialization in combination with the ReLU activation function
            weights_cell{i-1} = randn(n_out, n_in + 1) .* sqrt(2 / (n_in + 1));  % + 1 is for the bias term

        case 'zeros'  % Not for training! Only used to get matrices with zeros in the shape of the weigth matrices
            weights_cell{i-1} = zeros(NeuralNet_design_vector(i), NeuralNet_design_vector(i-1) + 1);  % + 1 is for the bias term
    end
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Feed-Forward 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Output, hiddenOutputs] = feedForward(Inputs, weightsCell, hiddenOutputs, Num_Hidden_Layers)
    
    % Pass the inputs into the first hidden layer
    hiddenOutputs{1} = sigmoid(weightsCell{1} * [1;Inputs]);
    
    % Pass the activations of the first hidden layer throgh the  remaining hidden layers
    for i = 2:Num_Hidden_Layers
        hiddenOutputs{i}  = sigmoid(weightsCell{i} * [1;hiddenOutputs{i-1}]);
    end
    
    % Pass the the activations from the last hidden layer to the output layer
    if Num_Hidden_Layers==0  % if there are no hidden layers, e.g perceptron
        Output = hiddenOutputs{1};
    else
        Output = sigmoid(weightsCell{end} * [1;hiddenOutputs{end}]);
    end
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Backpropagation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [weights_gradients_cell, Loss] = backProp(Inputs, hiddenOutputs, Outputs, Targets, weightsCell, Layer_Derivative, weights_gradients_cell, learn_rate, Num_Hidden_Layers)
    
    % Compute the error of the output layer. Meaning the derivative of the loss function times the derivative of the final activation function
    Error = Targets - Outputs;
    Loss = sum(Error.^2);
    Layer_Derivative{end} = -Error * Outputs*(1 - Outputs);  % dL/dy*dy/dz, Outputs*(1 - Outputs) is the derivative of the sigmoid activation function
    % Layer_Derivative{end} = -Error;  % dL/dy,
    
    % Propagate the derivative/errors through the hidden layers stating with the last hidden layer right behind the output layer
    for i = Num_Hidden_Layers:-1:1
        Layer_Derivative{i} = weightsCell{i+1}(:,2:end)'*Layer_Derivative{i+1} .* hiddenOutputs{i}.*(1 - hiddenOutputs{i});
    end
    
    % Compute gradients of weights to map from input to first hidden layer. If there are no subsequent hidden layers then this first hidden layer is also the output layer.
    weights_gradients_cell{1} = Layer_Derivative{1} * [1;Inputs]';

    % Compute gradients of weights in hidden layer(s)
    for i = 2:Num_Hidden_Layers
        weights_gradients_cell{i} = Layer_Derivative{i} * [1;hiddenOutputs{i}]';
    end
    
    % Compute gradients for output layer (if hidden layers exist) 
    if Num_Hidden_Layers > 0  % If there are no hidden layers the output layer is the same as the first (hidden) layer. Equaly the last hidden layer is also always the output layer
        % Compute gradients of weights in output layer
        weights_gradients_cell{end} = Layer_Derivative{end} * [1;hiddenOutputs{end}]';
    end
 
    % % Update all weights
    % for i = 1:Num_Hidden_Layers + 1
    %     weightsCell{i} = weightsCell{i} - learn_rate*weights_gradients_cell{i};
    % end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Activation Functions and their derivatives
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = sigmoid(x)
y = 1./(1 + exp(-x));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = sigmoid_derivative(x)
% y = 1./(1 + exp(-x)) .* (1 - 1./(1 + exp(-x)));
y = exp(-y1Test_Data)./(1 + exp(-y1Test_Data)).^2; % is slightly faster than the above
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


















