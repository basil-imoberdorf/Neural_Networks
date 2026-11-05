clear
close all
%delete(findall(0));

set(groot,'defaultAxesXTick',[],'defaultAxesXTickMode','manual')
set(groot,'defaultAxesYTick',[],'defaultAxesYTickMode','manual')

len = 2400;
hei = 1300;
div = 20;

%set up for 16:9 144Hz monitor
fig1 = uifigure('WindowStyle','alwaysontop');
fig1.Position = [(2560-len)/2, (1440-hei)/2 - 50, len, hei];
pause(2) %give it time to open the fig
fig1.Color = [0.6 0.6 0.6];

ax = uiaxes(fig1);
ax.Position = [(2*hei/div + ((1-3/div)*hei*.5)), (hei/div), (len - ((1-3/div)*hei*.5) - (3*hei/div)), (hei - (2*hei/div))];
ay = uiaxes(fig1);
ay.Position = [(hei/div), (2*hei/div + ((1-3/div)*hei*.5)), ((1-3/div)*hei*.5), ((1-3/div)*hei*.5)];
az = uiaxes(fig1);
az.Position = [(hei/div), (hei/div), ((1-3/div)*hei*.5), ((1-3/div)*hei*.5)];

%design dense neural network
designVector = [2,2,1];

%parameters for the algorithm
%define the ammount of training samples in the dataset
numSamples = 5000;
lern_rate = 0.1;
m = 0.5;  % how much of the previous gradient is used for the gradient descent step. the current gradient is scaled by 1-m to avoid distorting the gradient
epochs = 20;
minibatch_size = 16;

%create dataset and according labels
x = randi(2,1,numSamples)-1;
y = randi(2,1,numSamples)-1;
Targets = double(xor(x,y));

%set zeros to -1 for tanh usage
x(x==0) = -1;
y(y==0) = -1;
Targets(Targets==0) = -1;

%set up presets for algorithm
Inputs = [x;y];
weightsCell = genWeightsNN(designVector);   % generate random weights
averageLoss = zeros(1,epochs);
epochalLoss = 0;
previous_gradientCell = weightsCell;

for e = 1:epochs
    for k = 1:numSamples
        % functions that coordinate the neural network
        [Outputs,hiddenOutputsCell] = feedForwardNN(Inputs(:,k),weightsCell,'tanh');
        [gradientCell,loss] = backpropNN(Inputs(:,k),Outputs,hiddenOutputsCell,weightsCell,Targets(k));
        %sum up loss during whole dataset even while we update while going trogh
        epochalLoss = epochalLoss + loss;
        %update weights
        for i = 1:length(designVector) - 1
            weightsCell{i} = weightsCell{i} - lern_rate.*((1-m)*gradientCell{i} + m*previous_gradientCell{i});
        end
        %store gradients for usage in the momentum term in the next
        previous_gradientCell = gradientCell;
    end
    averageLoss(e) = epochalLoss/numSamples;
    epochalLoss = 0;

    %plot the decision surface
    res = 25;
    xx = linspace(-1,1,res);
    yy = linspace(1,-1,res);
    [X,~] = meshgrid(xx);
    [~,Y] = meshgrid(yy);
    Ot = zeros(res);
    for u = 1:res
        for j = 1:res
            % 3D-Figure
            Inpuut = [Y(u,j);X(u,j)];
            Ot(u,j) = feedForwardNN(Inpuut,weightsCell);
        end
    end
    surf(ay,Y,X,Ot);
    ay.ZLim = [-1 1];
    ay.DataAspectRatio = [1 1 1];
    ay.Box = 'on';
    ay.Color = [.2 .2 .2];
    colormap(ay, 'cool');
    ay.View = [10+e*0.8,45];
    ay.Projection = 'perspective' ;
    ay.Title.String = 'Decision Surface';

    %draw neural network
    [storedPlot,weightsColorMap,storedScatter] = drawNN(designVector,Inputs(:,randi(numSamples)),Outputs,hiddenOutputsCell,weightsCell);
    storedPlot(ax);
    colororder(ax, weightsColorMap);
    hold(ax, 'on');
    storedScatter(ax);
    ax.XLim = [0.5 length(designVector)+0.5];
    ax.YLim = [0 max(designVector)+2];
    ax.Color = 'k';

    %plot evolution of loss
    plot(az,1:e,nonzeros(averageLoss),'LineWidth',2,'Color','r')
    az.YLim = [0 max(averageLoss)];
    az.XLim = [0 e];
    az.YTickMode = 'auto';
    az.XTickMode = 'auto';
    az.YLabel.String = 'Loss';
    az.XLabel.String = 'Epochs';
    az.Title.String = 'Loss Trend';
    az.Color = 'k';
    pause(10e-10)
end

%restet groot properties
reset(groot)