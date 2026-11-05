clear
close all

% fig1 = figure('Position',[200,200,800,800]);
fig2 = figure('Position',[500,700,1200,500]);
fig3 = figure('Position',[500,100,1200,500]);
% ax = axes(fig1); hold(ax, 'on')
ay = axes(fig2); hold(ay, 'on')
az = axes(fig3); hold(ay, 'on')

% design dense neural network
designVector = [12,18,1];

% parameters for the algorithm
% define the ammount of training samples in the dataset
num_observations = 40;
lern_rate = 0.08;
epochs = 200;

% Synthetic Data to train the Neural Network on
r = 1000;
m = 100;
x = linspace(-1,1,num_observations);
z = linspace(-1,1,num_observations);
[X,Z] = meshgrid(x,z);
Y = 0.5*cos(2*pi*X*1) + 0.5*cos(2*pi*Z*1);
x = X(:)';
z = Z(:)';
y = Y(:)';
y_pred = zeros(num_observations,num_observations);
Targets = y;

% Y = zeros(m,r);
% for i = 1:m
%     Y(i,:) = 30*sin(2*pi*x) + randn(1,r);
% end
% Targets = Y;


% Set up presets for algorithm
Inputs = [x;z;x.^2;z.^2;x.^3;z.^3;x.^4;z.^4;x.^5;z.^5;x.^6;z.^6];
weightsCell = genWeightsNN(designVector);    % generate random weights
averageLoss = zeros(1,epochs);
epochalLoss = 0;
previous_gradientCell = weightsCell;

% Initialize plots
plt2 = scatter3(ay,X,Z,Y,50,[1,0.6,0],'filled');
plt1 = surf(ay,X,Z,y_pred);
ay.Box = 'on';
ay.XLabel.String = 'x';
ay.YLabel.String = 'y';
ay.ZLabel.String = 'z';
% ay.YLim = [-1.1, 1.1];
ay.View = [20,60];

% x1 = linspace(-1,1,10*num_observations);
% z1 = linspace(-1,1,10*num_observations);
% [X1,Z1] = meshgrid(x,z);
% x = X1(:)';
% z = Z1(:)';
% Inputs = [x;z];

runner = 0;
for e = 1:epochs
    for k = 1:num_observations^2
        % functions that coordinate the neural network
        [Outputs,hiddenOutputsCell] = feedForwardNN(Inputs(:,k),weightsCell,'tanh');
        [gradientCell,loss] = backpropNN(Inputs(:,k),Outputs,hiddenOutputsCell,weightsCell,Targets(k),'tanh');
        %sum up loss during whole dataset even while we update while going trogh
        epochalLoss = epochalLoss + loss;
        %update weights
        for i = 1:length(designVector) - 1
            weightsCell{i} = weightsCell{i} - lern_rate.*(gradientCell{i} + previous_gradientCell{i}.*0.0009); % with momentum
        end
        %store gradients for usage in the momentum term in the next
        previous_gradientCell = gradientCell;
    end
    averageLoss(e) = epochalLoss/num_observations;
    epochalLoss = 0;
    
    if mod(e,1)==0
        % Plot the networks output at the current training status
        for i = 1:num_observations
            for j = 1:num_observations
                runner = runner + 1;
                y_pred(j,i) = feedForwardNN(Inputs(:,runner),weightsCell);
            end
        end
        runner = 0;
        plt1.ZData = y_pred;
        pause(0)
    end
end

% % Draw neural network
% [storedPlot,weightsColorMap,storedScatter] = drawNN(designVector,Inputs(:,1),Outputs,hiddenOutputsCell,weightsCell);
% storedPlot(ax);
% colororder(ax, weightsColorMap);
% hold(ax, 'on');
% storedScatter(ax);
% ax.XLim = [0.5 length(designVector)+0.5];
% ax.YLim = [0 max(designVector)+2];
% ax.Color = 'k';
% pause(10e-10)

% Plot evolution of loss
plot(az,1:e,nonzeros(averageLoss),'LineWidth',2,'Color','r')
az.YLim = [0,max(averageLoss)];
az.YLabel.String = 'Loss';
az.XLabel.String = 'Epochs';
az.Title.String = 'Loss Trend';
az.Color = 'k';