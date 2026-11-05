function weightsCell = genWeightsNN(designVector)
%returns a cell array containig the weight matrices for the given designVector. It's length is length(desginVector)-1
%designVector must be a one by m Vector. The length defines the ammount of desired layers and the values at index i corresponds to the ammount of neurons in  layer i.
%The first entry of the vector must be the ammount of inputs to the network
%The last entry of the vector must be the ammount of desired outputs  of the network
%Example for the designVector: [nInputs,nHidden1,nHidden2,...,nHiddeni-1,nOutputs]
%i.e the neural network with designVector = [2,2,1] is sufficient to learn the task of XOR
weightsCell = cell(1,length(designVector) - 1);  % the -1 accounts for the fact that there are only weights between layers. So on no weights for the last layer --> the output layer has no weights
for i = 1:length(designVector)-1
    weightsCell{i} = randn(designVector(i+1),designVector(i)+1);  %the +1 is to add a weight for the bias
end
end