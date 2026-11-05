function weights_gradients_cell = backProp(Error, Inputs, Outputs, hiddenOutputs, weights_cell, Layer_Derivative, weights_gradients_cell, Num_Hidden_Layers, bias_filler, num_output_elements)
% This function computes the average gradient of a multilayer perceptron
%   Error:      Y-Y_hat
%   Inputs:     
%   Outputs:      
%   batch_size: number of inputs per mini-batch


% Compute the derivative at the output layer. Meaning the derivative of the loss function with respect to the output times the derivative of the output with respect to the final weighted sum
Layer_Derivative{end} = -Error./num_output_elements .* Outputs.*(1 - Outputs);  % dL/dy*dy/dz, Outputs*(1 - Outputs) is the derivative of the sigmoid activation function, num_output_elements=outut_dimension*mini-batch_size, for correct gradients

% Propagate the derivative/errors backwards through the hidden layers to the input layer stating with the last hidden layer right behind the output layer
for i = Num_Hidden_Layers:-1:1
    Layer_Derivative{i} = weights_cell{i+1}(:,2:end)'*Layer_Derivative{i+1} .* (1 - hiddenOutputs{i}.^2);
end

% Compute gradients of weights to map from input to first hidden layer. If there are no subsequent hidden layers then this first hidden layer is also the output layer.
weights_gradients_cell{1} = (Layer_Derivative{1} * Inputs'); % The one for the bias neuron is already inside. Was done outside this funciton Inputs = [1;Inputs]

% Compute gradients of weights in hidden layer(s)
for i = 2:Num_Hidden_Layers
    weights_gradients_cell{i} = (Layer_Derivative{i} * [bias_filler;hiddenOutputs{i-1}]');
end

% Compute gradients for output layer (if hidden layers exist)
if Num_Hidden_Layers > 0  % If there are no hidden layers the output layer is the same as the first (hidden) layer. Equaly the last hidden layer is also always the output layer
    % Compute gradients of weights in output layer
    weights_gradients_cell{end} = (Layer_Derivative{end} * [bias_filler;hiddenOutputs{end}]');
end
end