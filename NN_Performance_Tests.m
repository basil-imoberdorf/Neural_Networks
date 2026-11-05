clear
close all

% Define the number of neurons per layer
n_inputs = 2;
n_outputs = 1;
dense_NeuralNet_design = [n_inputs, 10, n_outputs];  % [Input, Hidden_1, Hidden_2, ..., Output]

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

     
% XOR training data
x1 = [1,1,0,0];
x2 = [1,0,1,0];
Targets = [0,1,1,0];
x_inputs = [x1; x2];

% Presets
learn_rate = 0.8;
max_epochs = 10000;
batch_loss = 0;
tracker = 0;
loss_flag = true;

% the main criterion to asses the algorithm is the evolution of the loss in this code
repetitions = 10; % define the number of times the trainingis repeated and loss is recorded for the same hyperparameter settings to get an idea of the variability and avoid false conclusions
max_batch_size = 4;
loss_history = zeros(max_epochs, repetitions*max_batch_size);  % container to plot the evolution of the loss
momenta = [0,0.3,0.6,0.9];
momenta = [0,0.9,0.93,0.96];

completion_times_matrix = zeros(repetitions, max_batch_size);
convergence_times_matrix = zeros(repetitions, max_batch_size);

% % to plot the loss landscape using two weights
% weight_location1 = [1,1,1];
% weight_location2 = [1,10,1];
% surf_handle = make_ll_canvas();

% Grid Search of Hyperparameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for a = 1:max_batch_size  % test the batch size
    for reps = 1:repetitions
        tic
        tracker = tracker + 1;
        %%% Actively changed hyperparameters for learning
        momentum_share = momenta(a); % define what fraction of the previous gradient should be used for the current gradient. Only the remaining part, 1-momentum_share, is contributed by the current gradient
        batch_size = 3;
        momentum_share_gradient = 1-momentum_share; % same idea as for above. save a computation in this case the scalar that defines the fraction of the current gradient used. depending on cache is bad mkay
        gradient_scaler = learn_rate/batch_size; % this variabl combines the constant relation that is applied to each weight update. This saves a divison. maybe it helps a bit in performance
        %%%


        for e = 1:max_epochs
            random_sample_index = randperm(4); % get indices to shuffle dataset to prevent the network from learning the input sequence pattern
            for i = 1:batch_size
                Inputs = x_inputs(:,random_sample_index(i));

                % Feed forward part
                [Outputs, hiddenOutputs, Loss, Error] = feedForward(Inputs, Targets(random_sample_index(i)), weights_cell, hiddenOutputs, Num_Hidden_Layers);

                % calculate the the gradients to adjust the weights via backpropagation of the derivative of the loss function
                weights_gradients_cell = backProp(Error, Inputs, Outputs, hiddenOutputs, weights_cell, Layer_Derivative, weights_gradients_cell, Num_Hidden_Layers);

                % Compute the sum of the gradients of the weights for each layer for the current batch
                for p = 1:Num_Forward_Steps  % the number of forward steps is equal to the number of weight matrices because each forward step involves first a matrix multipliaction of said weights with the activations of the previous layer
                    weights_gradients_cell_batch{p} = weights_gradients_cell_batch{p} + weights_gradients_cell{p};
                end

                % sum up the loss for each training instance
                batch_loss = batch_loss + Loss;
            end

            % Update weights
            for i = 1:Num_Forward_Steps
                weights_momentum_cell{i} = (momentum_share.*weights_momentum_cell{i} + momentum_share_gradient*weights_gradients_cell_batch{i});
                weights_cell{i} = weights_cell{i} - gradient_scaler .* weights_momentum_cell{i};
            end

            % Visualize loss evolution
            loss_history(e, tracker) = batch_loss/batch_size;
            
            if e > 110 && loss_flag && (sum(loss_history(e-100:e, tracker) < 0.001)==100)
                loss_flag = false;
                convergence_times_matrix(reps, a) = toc;
                % convergence_times_matrix(reps, a) = e;
            end

            % Reset the total loss and accumulated gradients of the current epoch
            batch_loss = 0;
            weights_gradients_cell_batch = weights_set_to_zero_cell;
            
            % % Plot landscape
            % if mod(e,1000)==0
            %     plot_landscape(weight_location1, weight_location2, 40, Inputs, Targets(random_sample_index(i)), weights_cell, hiddenOutputs, Num_Hidden_Layers, surf_handle)
            % end
        end
        % Measure time of execution of the current training session
        completion_times_matrix(reps, a) = toc;
        loss_flag = true; % reset the loss flag for the next round to get the time when the loss fell below the threshold    

        % Reset weight cells
        weights_cell = initialize_weights(dense_NeuralNet_design, 'xavier', 0);
        weights_gradients_cell = weights_set_to_zero_cell;
        weights_gradients_cell_batch = weights_set_to_zero_cell;
        weights_momentum_cell = weights_set_to_zero_cell;

        hiddenOutputs = cell(Num_Hidden_Layers, 1);
        Layer_Derivative = cell(Num_Forward_Steps, 1);
    end
end

% Time statistics
completion_time_avg = mean(completion_times_matrix);
completion_time_std = std(completion_times_matrix);

convergence_time_avg = mean(convergence_times_matrix);
convergence_time_std = std(convergence_times_matrix);

fprintf("\n +---------------------+ \n | Algorithm Completion Time | \n +---------------------+ \n");
fprintf(" |Time  %s ±  %s | \n +---------------------+ \n",  num2str(round(completion_time_avg(1), 2)), num2str(round(completion_time_std(1), 3)));
fprintf(" |Time  %s ±  %s | \n +---------------------+ \n",  num2str(round(completion_time_avg(2), 2)), num2str(round(completion_time_std(2), 3)));
fprintf(" |Time  %s ±  %s | \n +---------------------+ \n",  num2str(round(completion_time_avg(3), 2)), num2str(round(completion_time_std(3), 3)));
fprintf(" |Time  %s ±  %s | \n +---------------------+ \n",  num2str(round(completion_time_avg(4), 2)), num2str(round(completion_time_std(4), 3)));

fprintf("\n +---------------------+ \n | Loss Convergence Time | \n +---------------------+ \n");
fprintf(" |Time  %s ±  %s | \n +---------------------+ \n",  num2str(round(convergence_time_avg(1), 2)), num2str(round(convergence_time_std(1), 3)));
fprintf(" |Time  %s ±  %s | \n +---------------------+ \n",  num2str(round(convergence_time_avg(2), 2)), num2str(round(convergence_time_std(2), 3)));
fprintf(" |Time  %s ±  %s | \n +---------------------+ \n",  num2str(round(convergence_time_avg(3), 2)), num2str(round(convergence_time_std(3), 3)));
fprintf(" |Time  %s ±  %s | \n +---------------------+ \n",  num2str(round(convergence_time_avg(4), 2)), num2str(round(convergence_time_std(4), 3)));

relative_completion_time = completion_time_avg./completion_time_avg(1)
relative_convergence_time = convergence_time_avg./convergence_time_avg(1)


errorbar(completion_time_avg, convergence_time_avg, convergence_time_std, convergence_time_std)
xlabel('training time')
ylabel('convergence time')

fig = figure('Position', [800, 50, 1600, 590], 'Color', 'w');
ay = axes(fig, 'Position', [0.07, 0.12, 0.9, 0.86]);
hold(ay, 'on');

% Plot results
cols = ['r';'g';'b';'c'];
for i = 1:max_batch_size  
    
    plot(ay, loss_history(:,((i-1)*repetitions)+1:i*repetitions), 'Color', cols(i))
    
end

ax.XLabel.String = 'Iterations [-]';
ax.YLabel.String = 'Loss [-]';

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
function [Outputs, hiddenOutputs, Loss, Error] = feedForward(Inputs, Targets, weightsCell, hiddenOutputs, Num_Hidden_Layers)
    
    % Pass the inputs into the first hidden layer
    hiddenOutputs{1} = sigmoid(weightsCell{1} * [1;Inputs]);
    
    % Pass the activations of the first hidden layer throgh the  remaining hidden layers
    for i = 2:Num_Hidden_Layers
        hiddenOutputs{i}  = sigmoid(weightsCell{i} * [1;hiddenOutputs{i-1}]);
    end
    
    % Pass the the activations from the last hidden layer to the output layer
    if Num_Hidden_Layers==0  % if there are no hidden layers, e.g perceptron
        Outputs = hiddenOutputs{1};
    else
        Outputs = sigmoid(weightsCell{end} * [1;hiddenOutputs{end}]);
    end

    % Compute the loss from the targets and the outputs
    Error = Targets - Outputs;
    Loss = 0.5*sum(Error.^2);
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Backpropagation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function weights_gradients_cell = backProp(Error, Inputs, Outputs, hiddenOutputs, weightsCell, Layer_Derivative, weights_gradients_cell, Num_Hidden_Layers)
    
    % Compute the error of the output layer. Meaning the derivative of the loss function times the derivative of the final activation function
    Layer_Derivative{end} = -Error * Outputs.*(1 - Outputs);  % dL/dy*dy/dz, Outputs*(1 - Outputs) is the derivative of the sigmoid activation function
    
    % Propagate the derivative/errors through the hidden layers stating with the last hidden layer right behind the output layer
    for i = Num_Hidden_Layers:-1:1
        Layer_Derivative{i} = weightsCell{i+1}(:,2:end)'*Layer_Derivative{i+1} .* hiddenOutputs{i}.*(1 - hiddenOutputs{i});
    end
    
    % Compute gradients of weights to map from input to first hidden layer. If there are no subsequent hidden layers then this first hidden layer is also the output layer.
    weights_gradients_cell{1} = Layer_Derivative{1} * [1;Inputs]';

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
function surf_handle = make_ll_canvas()
fig = figure('Position',[500,200,700,700]);
atx = axes(fig);

surf_handle = surf(atx, zeros(40),zeros(40),zeros(40), 'EdgeAlpha', 0.5);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot the loss landscape when looking at two single weights
function plot_landscape(weight_location1, weight_location2, res, Inputs, Targets, weights_cell, hiddenOutputs, Num_Hidden_Layers, surf_handle)


cell1 = weight_location1(1);
cell2 = weight_location2(1);

x1 = weight_location1(2);
x2 = weight_location2(2);

y1 = weight_location1(3);
y2 = weight_location2(3);


range = linspace(-8,8,res);

[X,Y] = meshgrid(range);
Loss = zeros(res);

for i = 1:res
    for j = 1:res
        
        weights_cell{cell1}(x1, y1) = range(i);
        weights_cell{cell2}(x2, y2) = range(j);

        [~, ~, loss, ~] = feedForward(Inputs, Targets, weights_cell, hiddenOutputs, Num_Hidden_Layers);
        Loss(i,j) = loss;

    end
end


surf_handle.XData = X;
surf_handle.YData = Y;
surf_handle.ZData = Loss;
pause(1/60)


end


