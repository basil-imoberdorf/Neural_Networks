function [Outputs,hiddenOutputsCell] = feedForwardNN(Inputs,weightsCell,ActFun)
%This function computes the forward propagation of the inputs trought the dense neural network defined by the weightsCell array
%Inputs must be a colum vector containing all inputs of a given trainingsample
%ActFun defines the used activation function for the whole network, Valid strings are: 'tanh','sig','ReLU'
%Returns the output(s) of the neural network
%default activation function is tanh
switch nargin
    case 2
        ActFun = 'tanh';
end

%generate cell for the hidden outputs
nLayers = length(weightsCell) - 1;          % nLayer doesnt account for the input layer nor output layer
hiddenOutputsCell = cell(1,nLayers);        % the -1 accounts for the fact that the final output gets stored in a seperate variable

%add 1 as first element of the input vector for the bias weights
Inputs = [1;Inputs];

switch ActFun %choose activation function
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TANH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case 'tanh'
        % Feed inputs into first layer
        hiddenOutputsCell{1} = [1;tanh(weightsCell{1} * Inputs)];

        for i = 1:nLayers
            if i~=nLayers   %put final output in seperate variable
                hiddenOutputsCell{i+1} = [1;tanh(weightsCell{i+1} * hiddenOutputsCell{i})];  %feed forward into next layer, %apply activation function and %add 1 for the bias weights of the next layer
            else
                Outputs =  tanh(weightsCell{i+1} * hiddenOutputsCell{i}); %feed forward into output layer and %apply activation function on output layer and put those in a seperate variable
                %Outputs =  weightsCell{i+1} * hiddenOutputsCell{i}; % feed forward into output layer without applying any activation function to enable the previous layer to form a weighted sum in the target region
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SIGMOID %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case 'sig'

        % Feed inputs into first layer
        hiddenOutputsCell{1} = [1;sig(weightsCell{1} * Inputs)];

        for i = 1:nLayers
            if i~=nLayers   % put final output in seperate variable
                hiddenOutputsCell{i+1} = [1;sig(weightsCell{i+1} * hiddenOutputsCell{i})]; %feed forward into next layer, %apply activation function and %add 1 for the bias weights of the next layer
            else
                Outputs =  tanh(weightsCell{i+1} * hiddenOutputsCell{i}); %feed forward into output layer and %apply activation function on output layer and put those in a seperate variable
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ReLctified Linear Unit %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case 'ReLU'

        % Feed inputs into first layer
        hiddenOutputsCell{1} = [1;ReLU(weightsCell{1} * Inputs)];
        for i = 1:nLayers
            if i~=nLayers   %put final output in seperate variable
                hiddenOutputsCell{i+1} = [1;ReLU(weightsCell{i+1} * hiddenOutputsCell{i})]; %feed forward into next layer, %apply activation function and %add 1 for the bias weights of the next layer
            else
                Outputs =  ReLU(weightsCell{i+1} * hiddenOutputsCell{i});  %feed forward into output layer and %apply activation function on output layer and put those in a seperate variable
            end
        end
end
end