function [Outputs, hiddenOutputs, Loss, Error] = feedForward(Inputs, Targets, weights_cell, hiddenOutputs, Num_Hidden_Layers, bias_filler, num_output_elements)

% Pass the inputs into the first hidden layer
hiddenOutputs{1} = tanh(weights_cell{1} * Inputs);

% Pass the activations of the first hidden layer throgh the  remaining hidden layers
for i = 2:Num_Hidden_Layers
    hiddenOutputs{i}  = tanh(weights_cell{i} * [bias_filler;hiddenOutputs{i-1}]); % simultaneous multi forward pass
end

% Pass the the activations from the last hidden layer to the output layer
if Num_Hidden_Layers==0  % if there are no hidden layers, e.g perceptron
    Outputs = hiddenOutputs{1}; % the first hidden output, already computed above, is already the output
else
    Outputs = sig(weights_cell{end} * [bias_filler;hiddenOutputs{end}]); % sigmoid activation function on last layer
end

% Compute the loss via the difference between the actual targets and the predicted outputs, called the error
Error = (Targets - Outputs);
Loss = 0.5*sum(Error.^2, 'all')./num_output_elements;  % compute scalar loss value as Mean Squared Error
end