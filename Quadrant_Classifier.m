clear
close all

%create dataset and labels
numSamples = 1300;                  %best values so far: 1300
range = 12;                         %best values so far: 12

x = randn(1,numSamples) + randi(2*range,1,numSamples) - range;
y = randn(1,numSamples) + randi(2*range,1,numSamples) - range;
true_output = zeros(1,numSamples);

for j = 1:numSamples
    if x(j)>-5 && 5>x(j) && y(j)>-5 && 5>y(j)
        true_output(:,j) =  1;
    else
        true_output(:,j) = -1;
    end
end

%design dense neural network
designVector = [2,16,16,1];         %best values so far: [2,26,26,1]

%parameters for the algorithm
%define the ammount of training samples in the dataset
lern_rate = 0.055;                  %best values so far: 55
epochs = 400;                       %best values so far: 400
minibatch_size = 1;                 %best values so far: 2

%set up presets for algorithm
Inputs = [x;y];
weightsCell = genWeightsNN(designVector);           %generate random weights
averageLoss = zeros(1,epochs);
epochalLoss = 0;
gradientSumCell = cell(1,length(weightsCell));
previous_gradientSumCell = weightsCell;

%plot the decision surface
res = 30;
xx = linspace(-range,range,res);
yy = linspace(-range,range,res);
[X,~] = meshgrid(xx);
[~,Y] = meshgrid(yy);
Ot = zeros(res);
Oj = Ot;

figure(1)
srf = surf(Y,X,Ot);
axis([-range, range, -range ,range , -1, 1])
colormap('cool');

for e = 1:epochs
    for k = 1:numSamples
        % functions that coordinate the neural network
        [Outputs,hiddenOutputsCell] = feedForwardNN(Inputs(:,k),weightsCell);
        [gradientCell,loss] = backpropNN(Inputs(:,k),Outputs,hiddenOutputsCell,weightsCell,true_output(k));
        %sum up loss during whole dataset even while we update while going through
        epochalLoss = epochalLoss + loss;

        %sum up weights
        for i = 1:length(designVector) - 1
            if k == 1
                gradientSumCell{i} = gradientCell{i};
            else
                gradientSumCell{i} = gradientSumCell{i} + gradientCell{i};
            end
        end

        %update weights minibatch style
        if mod(k,minibatch_size) == 0
            for i = 1:length(designVector) - 1
                weightsCell{i} = weightsCell{i} - lern_rate.*((gradientCell{i} + previous_gradientSumCell{i}.*0.9)/minibatch_size);
                %store gradients for usage in the momentum term in the next update
                previous_gradientSumCell = gradientSumCell;
                %reset summation variable
                gradientSumCell{i} = 0;
            end
        end
    end
    averageLoss(e) = epochalLoss/numSamples;
    epochalLoss = 0;
    %refine lern_rate over epochs
    lern_rate = lern_rate * 0.999;

    for u = 1:res
        for j = 1:res
            % 3D-Figure
            Inpuut = [Y(u,j);X(u,j)];
            prediction = feedForwardNN(Inpuut,weightsCell);
            Ot(u,j) = prediction(1); %for true decision surface
            %Oj(u,j) = prediction(2); %for false decision surface
        end
    end
    
    srf.XData = X;
    srf.YData = Y;
    srf.ZData = Ot;

    view([10+e*0.8,45]);
    drawnow limitrate
end