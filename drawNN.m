function [storedPlot,weightsColorMap,storedScatter] = drawNN(designVector,Inputs,Outputs,hiddenOutputsCell,weightsCell)
%Returns everything necessary for drwing the network in the axes of choice.
% weightsColorMap contains the colors for hthe connections. Use colororder() in your script to apply the colors
% storedPlot(ax) & storedScatter(ax) are functions to plot the network in the desired ax
%
%Positive Inputs, Ouputs, Activations and weighted connections are colored green. Negative are colored red
%The intensity of the color corresponds to the abs of the value. Values near zero are faint


%presets
designVector(1:end-1) = designVector(1:end-1) + 1; %for the bias add one neuron except at the end
nLayers = length(designVector);
nNeurons = sum(designVector);
maxNeuron = max(designVector);
Inputs = [1;Inputs];     %add the 1 for the bias neuron in the input layer
idefix = 0;
changeIndicator = 0;
nConnections = 0;

xxx = zeros(1,nNeurons);
yyy = zeros(1,nNeurons);

%generate right ammount of nConnections assuming each layer except for the putput has 1 bias neuron
for h = 1:nLayers-1
    if h+1 == nLayers
        nConnections = nConnections + designVector(h)*(designVector(h+1));
    else
        nConnections = nConnections + designVector(h)*(designVector(h+1)-1);
    end
end
weightsColorMap = zeros(nConnections,3);
totalXX = zeros(2,nConnections);
totalYY = zeros(2,nConnections);

%place neurons
for k = 1:nLayers
    if k==1
        xxx(1:designVector(k)) = ones(1,designVector(k)).*k;
        yyy(1:designVector(k)) = (1:designVector(k)) + (maxNeuron - designVector(k))/2;
    else
        xxx(sum(designVector(1:k-1))+1:sum(designVector(1:k))) = ones(1,designVector(k)).*k;
        yyy(sum(designVector(1:k-1))+1:sum(designVector(1:k))) = (1:designVector(k)) + (maxNeuron - designVector(k))/2;
    end
end
%to draw the last layer fully connected (there is no bias neuron in the output) we need to add artificially one more because it stopps 1 before the end so its good fix!
designVector(end) = designVector(end) + 1;
for i = 1:nLayers-1
    if designVector(i+1)==1
        changeIndicator = 1;
        designVector(i+1) = 2;       %make the alogrithm work even when only one output is there. Dont forget to reset
    end
    for j = 1:designVector(i+1)-1
        %generate x axis values for connective lines
        xx = zeros(2,designVector(i));
        xx(1,:) = nonzeros(xxx==i) .* i;
        xx(2,:) = xx(1,:) + 1;

        %generate y axis values for connective lines
        yy = zeros(2,designVector(i));
        if i==1
            yy(1,:) = yyy(1:designVector(i));
            yy(2,:) = yyy(sum(designVector(1:i)) + j);
        else
            yy(1,:) = yyy(sum(designVector(1:i-1))+1:sum(designVector(1:i)));
            yy(2,:) = yyy(sum(designVector(1:i)) + j);
        end

        %do everything to color these connections according to their sign
        %maxWeight = max(max(abs(weightsCell{i})));
        maxWeight = 1;
        if size(weightsCell{i},1) == 1
            inversMatrix = weightsCell{i};
        else
            inversMatrix = flip(weightsCell{i});
        end
        negLogic = flip(inversMatrix(j,:)')<=0;
        posLogic = ~negLogic;
        posShit = 1./(1 + exp(-3.*posLogic.*flip(abs(inversMatrix(j,:))'/maxWeight))) - 0.5;
        negShit = 1./(1 + exp(-3.*negLogic.*flip(abs(inversMatrix(j,:))'/maxWeight))) - 0.5;
        weightsColor = [negShit + negLogic.*(1-negShit)/2,posShit + posLogic.*(1-posShit)/2,zeros(designVector(i),1)];
        inder = length(nonzeros(weightsColorMap));

        %collect all connections in two new variables
        weightsColorMap(inder + 1:inder + length(inversMatrix(j,:)),:) = weightsColor;  %make colormap becaus oder wise the curren weightsColor changes all previous lines, so it has to be added at the end
        totalXX(:,inder + 1:inder + length(xx)) = xx;
        totalYY(:,inder + 1:inder + length(xx)) = yy;
    end
end
designVector(end) = designVector(end) - 1; %reset the add 1 to the end
%export products of this function
storedPlot = @(ax) plot(ax,totalXX,totalYY,'LineWidth',3);

if changeIndicator==1
    designVector(end) = designVector(end) - 1; %reset the add if there was only one output neuron
end

%color neurons
c = zeros(nNeurons,3);
for i = 1:nNeurons
    if i <= designVector(1)   %color input layer neurons
        maxInput = max(Inputs);
        minInput = min(Inputs);
        idefox = designVector(xxx(i)) - idefix; %because we color from top to bottom but the values are stored from top to bottom we have to adjoust the index here
        if idefox == 1
            c(i,:) = [0,1,1];
        else
            c(i,:) = [(Inputs(idefox)<=0)*((Inputs(idefox)/minInput) + 0.5),(Inputs(idefox)>0)*((Inputs(idefox)/maxInput) + 0.5),0];
        end
        idefix = idefix + 1;
        if xxx(i) ~= xxx(i+1)
            idefix = 0; %reset index variable to acces the elements in the hidden output cell
        end
    elseif i <= sum(designVector(1:end-1)) %color hidden layer neurons
        maxcurrentHidden = max(hiddenOutputsCell{xxx(i)-1});
        minCurrentHidden = min(hiddenOutputsCell{xxx(i)-1});
        idefox = designVector(xxx(i)) - idefix; %because we color from top to bottom but the values are stored from top to bottom we have to adjoust the index here
        if idefox == 1
            c(i,:) = [0,1,1];
        else
            c(i,:) = [(hiddenOutputsCell{xxx(i)-1}(idefox)<=0)*((hiddenOutputsCell{xxx(i)-1}(idefox)/minCurrentHidden) + 0.5),(hiddenOutputsCell{xxx(i)-1}(idefox)>0)*((hiddenOutputsCell{xxx(i)-1}(idefox)/maxcurrentHidden) + 0.5),0];
        end
        idefix = idefix + 1;
        if xxx(i) ~= xxx(i+1)
            idefix = 0; %reset index variable to acces the elements in the hidden output cell
        end
    else
        maxOutputs = max(Outputs);
        minOutputs = min(Outputs);
        idefox = designVector(xxx(i)) - idefix; %because we color from top to bottom but the values are stored from top to bottom we have to adjoust the index here
        c(i,:) = [(Outputs(idefox)<=0)*((Outputs(idefox)/minOutputs) + 0.5),(Outputs(idefox)>0)*((Outputs(idefox)/maxOutputs) + 0.5),0];
        idefix = idefix + 1; %no reset of idefix at the end necessary
    end
end
%export product of this function
storedScatter = @(ax) scatter(ax,xxx',yyy',10000/nNeurons,c,'filled','MarkerEdgeColor','w','LineWidth',2);
end