function [gradientCell,loss] = backpropNN(Inputs, Outputs, hiddenOutputsCell, weightsCell, Targets, ActFun, LossFun)
% This function executes the tuning of the paramters, the weights, by gradient descent of a cost function defined by the string in the argument LossFun
% Returns a cell with all the gradients for all the weights
% The function needs the final output of the network as well as the hidden outputs. Then it needs the target labels for each input and the cell containing all weights
% 
% Valid strings for LossFun are currently:   'MSE':      "mean square error"                           f(x) = 1/2 * (target - x)^2
%                                            'BCE':      "binary cross-entropy loss" or  log loss      f(x) = -(target*log(x) + (1-target)*log(1-x))    with log as the natural logarithm
% 
% Note: Default LossFun is 'MSE' and for ActFun default is 'tanh'
% Note: 'BCE' Needs Outputs that have been normalized between 1 and 0 and Targets that are eighter 1 or 0!
% Note: The name of the derivative function of the activaion function "ActFun" must be the same string but end with the letter d to be accessed by the algorithm!

nLayers = length(weightsCell) - 1;      % nLayer doesnt account for the input layer nor output layer
%generate cell for the hidden errors
L_errors = cell(1,nLayers + 1);         % the -1 accounts for the fact that the final output gets stored in a seperate variable
gradientCell = cell(1,nLayers + 1);     % gradietnsCell must have the same size as the weightsCell


%set default loss function
switch nargin
    case 5
        ActFun  = 'tanh';
        LossFun = 'MSE';
    case 6
        LossFun = 'MSE';
end

switch ActFun % Choose activation function
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TANH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case 'tanh'
        % calculate cost or "loss" and error(s) of output layer
        switch LossFun
            case 'MSE' % Mean Squared Error
                loss = sum(1/2 .* (Targets - Outputs).^2);
                % L_errors{nLayers+1} = -((Targets - Outputs) .* eval(sprintf('%sd(Outputs)',ActFun)));   % error of first layer
                %L_errors{nLayers+1} = -(Targets - Outputs) .* tanhd(Outputs);   % error of output layer
                L_errors{nLayers+1} = -(Targets - Outputs);   % error of first layer
            case 'BCE' % Binary Crossentropy Loss
                % protect the log function from bad entries
                Outputs(Outputs==0) = 10e-10;
                Outputs(Outputs==1) = 1 - 10e-10;
                loss = sum(-(Targets.*log(Outputs) + (1-Targets).*log(1-Outputs)));     %error of output layer
                L_errors{nLayers+1} = -(Targets - Outputs);
        end

        % compute errors, i mean derivatives, for the hidden layers
        for i = nLayers:-1:1
            if i==nLayers
                L_errors{i} = (weightsCell{i+1}' * L_errors{i+1}) .* tanhd(hiddenOutputsCell{i}); % matrix multiplication with transponed weights gives the sum of errors for all neurons one layer behind. Then make Hadamar product with derivatives of the activations on the layer behind
            else
                L_errors{i} = (weightsCell{i+1}' * L_errors{i+1}(2:end)) .* tanhd(hiddenOutputsCell{i});
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SIGMOID %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case 'sig'
        % calculate cost or "loss" and error(s) of output layer 
        switch LossFun
            case 'MSE'
                loss = sum(1/2 .* (Targets - Outputs).^2);
                L_errors{nLayers+1} = -(Targets - Outputs) .* sigd(Outputs);   % error of output layer

            case 'BCE'
                % protect the log function from bad entries
                Outputs(Outputs==0) = 10e-10;
                Outputs(Outputs==1) = 1 - 10e-10;
                loss = sum(-(Targets.*log(Outputs) + (1-Targets).*log(1-Outputs)));     %error of output layer
                L_errors{nLayers+1} = -(Targets - Outputs);
        end

        % compute errors, i mean derivatives, for the hidden layers
        for i = nLayers:-1:1
            if i==nLayers
                %L_errors{i} = (weightsCell{i+1}' * L_errors{i+1}) .* eval(sprintf('%sd(hiddenOutputsCell{i})',ActFun)); %matrix multiplication with transponed weights gives the sum of errors for all neurons one layer behind. Then make Hadamar product with derivatives of the activations on the layer behind
                L_errors{i} = (weightsCell{i+1}' * L_errors{i+1}) .* sigd(hiddenOutputsCell{i}); %matrix multiplication with transponed weights gives the sum of errors for all neurons one layer behind. Then make Hadamar product with derivatives of the activations on the layer behind
            else
                %L_errors{i} = (weightsCell{i+1}' * L_errors{i+1}(2:end)) .* eval(sprintf('%sd(hiddenOutputsCell{i})','tanh'));
                L_errors{i} = (weightsCell{i+1}' * L_errors{i+1}(2:end)) .* sigd(hiddenOutputsCell{i});
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Rectified Linear Unit %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case 'ReLU'
        % calculate cost or "loss" and error(s) of output layer
        switch LossFun
            case 'MSE'
                loss = sum(1/2 .* (Targets - Outputs).^2);
                L_errors{nLayers+1} = -(Targets - Outputs) .* ReLUd(Outputs);   % error of output layer

            case 'BCE'
                % protect the log function from bad entries
                Outputs(Outputs==0) = 10e-10;
                Outputs(Outputs==1) = 1 - 10e-10;
                loss = sum(-(Targets.*log(Outputs) + (1-Targets).*log(1-Outputs)));     % error of output layer
                L_errors{nLayers+1} = -(Targets - Outputs);
        end

        % Compute errors, i mean derivatives, for the hidden layers
        for i = nLayers:-1:1
            if i==nLayers
                L_errors{i} = (weightsCell{i+1}' * L_errors{i+1}) .* ReLUd(hiddenOutputsCell{i}); %matrix multiplication with transponed weights gives the sum of errors for all neurons one layer behind. Then make Hadamar product with derivatives of the activations on the layer behind
            else
                L_errors{i} = (weightsCell{i+1}' * L_errors{i+1}(2:end)) .* ReLUd(hiddenOutputsCell{i});
            end
        end
end

% Compute gradients
Inputs = [1;Inputs]; % Add 1 for the bias
for j = 1:nLayers + 1
    if j==nLayers + 1  % handle output layer
        gradientCell{j} = L_errors{j} * hiddenOutputsCell{j-1}';
    elseif j==1  % handle input layer
        gradientCell{j} = L_errors{j}(2:end) * Inputs';
    else  % handle hidden layers
        gradientCell{j} = L_errors{j}(2:end) * hiddenOutputsCell{j-1}'; % the (2:end) is there to ignore the bias in position 1 of the column vector. The bias is not connected with the previous layer
    end
end
end