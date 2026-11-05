if exist('uifig1','var')
    delete(uifig1)
end
clear 
close all

%set up diagonistc windows
res = 30;
uifig1 = uifigure('WindowStyle','alwaysontop');
uifig1.Position = [600 50 1250 1250];
uifig1.Color = [0.6 0.6 0.6];

set(0,'DefaultFigureVisible','off');
uiax1 = uiaxes(uifig1);
uiax1.Box = 'on';
uiax1.Color = 'k';
cmp = colormap('cool');
set(0,'DefaultFigureVisible','on');
uiax1.Colormap = cmp;
uiax1.BoxStyle = 'full';
uiax1.PlotBoxAspectRatio = [1 1 1];
uiax1.ZLim = [-1 1];
ylabel(uiax1,'Input 2','FontSize',20)
xlabel(uiax1,'Input 1','FontSize',20)
zlabel(uiax1,'Output','FontSize',20)
uiax1.InnerPosition = [32.5 625+32.5 560 560];

uiax2 = uiaxes(uifig1);
uiax2.Box = 'on';
uiax2.YTick = [];
uiax2.XTick = [];
uiax2.XLim = [0.5 res+0.5];
uiax2.YLim = [0.5 res+0.5];
uiax2.InnerPosition = [32.5 32.5 560 560];

uiax3 = uiaxes(uifig1);
uiax3.Color = 'k';
ylabel(uiax3,'Cost','FontSize',20)
xlabel(uiax3,'Epochs','FontSize',20)
uiax3.Box = 'on';
uiax3.InnerPosition = [625+32.5 625+32.5 560 560];

uiax4 = uiaxes(uifig1);
uiax4.Box = 'on';
uiax4.Color = 'k';
uiax4.YTick = [];
uiax4.XTick = [];
uiax4.XLim = [0.5 3+0.5];
uiax4.YLim = [0.5 3+0.5];
uiax4.InnerPosition = [625+32.5 32.5 560 560];

%parameters for the algorithm
%define the ammount of training samples in the dataset
numSamples = 15;
lern_rate = 0.15;
epochs = 20;

%create dataset and according labels
x = randi(2,1,numSamples)-1;
y = randi(2,1,numSamples)-1;
true_output = double(xor(x,y));

%set zeros to -1 for tanh usage
x(x==0) = -1;
y(y==0) = -1;
true_output(true_output==0) = -1;

%set up random weights
%generate 3 weigths per neuron: two for each input/activation and one for the bias
weightsHid = randn(2,3);
weightsOut = randn(1,3);

%set up presets for algorithm
inputs = ones(3,numSamples);        %the constant input 1 for the hidden layer biases is added here
inputs([1,2],:) = [x;y];
outputHid = ones(3,numSamples);     %the constant input 1 for the output layer bias is added here
averageLoss = zeros(1,epochs);
epochalLoss = 0;
previous_deltaWeightsOut = 0;
previous_deltaWeightsHid = 0;
%presets for diagnostics
im = uint8(zeros(res,res,3));
x = linspace(-1,1,res);
y = linspace(1,-1,res);
[X,~] = meshgrid(x);
[~,Y] = meshgrid(y);
Ot = zeros(res);
nNodesInLayerK = [3,3,1];
nNeurons = sum(nNodesInLayerK);
xx = zeros(1,nNeurons);
yy = zeros(1,nNeurons);

for e = 1:epochs
    for i = 1:numSamples
        %feedforward into hidden layer
        weightedSumHid = weightsHid * inputs(:,i);
        outputHid([1,2],i) = tanh(weightedSumHid); %put the outputs of the two hidden neurons to the vector containing the bias 1 to be able to calculate the next layer
        %feedforward into output layer
        output = tanh(weightsOut * outputHid(:,i));

        %calculate loss with square loss
        loss = 1/2 * (true_output(i) - output)^2;    %note: the factor 1/2 is there to simplify the loss functions derivative to -(target-guess). In the end by going in the negative diretion of the gradient the expression turns back to positve: deltaWeight = -lern_rate(-(target-guess)*bli*bla*blub) = lern_rate*(target - guess)*bli*bla*blub
        epochalLoss = epochalLoss + loss;

        %calculate errors for each layer
        output_error  = (true_output(:,i) - output) .* tanhd(output);
        hidden_errors = weightsOut' * output_error;

        %calculate weights correction values
        deltaWeightsOut = output_error * outputHid(:,i)';
        deltaWeightsHid = hidden_errors(1:2) .* tanhd(outputHid(1:2,i)) * inputs(:,i)';

        %update weights
        weightsHid = weightsHid + lern_rate * (deltaWeightsHid + previous_deltaWeightsHid*0.9);
        weightsOut = weightsOut + lern_rate * (deltaWeightsOut + previous_deltaWeightsOut*0.9);

        %store current weight change to be used the next momentum term in the update equation
%                 previous_deltaWeightsOut = deltaWeightsOut;
%                 previous_deltaWeightsHid = deltaWeightsHid;
        %drawNeuralNet(inputs(:,i),outputHid(:,i),output,weightsHid,weightsOut)
        %         hold on
        %         pause(10e-10)

    end
    for u = 1:res
        for j = 1:res
            % 2D-Image
            Input = [y(u);x(j);1];
            output = tanh(weightsOut*[tanh(weightsHid*Input);1]);
            im(u,j,:) = [(output+1)/2,(output+1)/2,(output+1)/2] .* 255;

            % 3D-Figure
            Input = [Y(u,j);X(u,j);1];
            Ot(u,j) = tanh(weightsOut*[tanh(weightsHid*Input);1]);
        end
    end
    %plot 3D digonstics
    surf(uiax1,X,Y,Ot);
    uiax1.View = [10+e*0.3, 45];
    pause(10e-10)
    %plot 2D digonstics
    image(uiax2,im);
    uiax2.View = [0 90];
    pause(10e-3)
    %plot loss function digonstics
    averageLoss(e) = epochalLoss/numSamples;
    epochalLoss = 0;
    plot(uiax3,1:e,nonzeros(averageLoss),'LineWidth',2,'Color','r')
    uiax3.YLim = [0,max(averageLoss)];
    uiax3.XLim = [0,e];

    %plot network
    %Draws the current neural network. Green if +positive values. Red if 0 or negative
    t = 0;
    %place neurons
    for k = 1:length(nNodesInLayerK)
        for i = 1:nNeurons
            currentLen = length(nonzeros(xx)) + 1;
            if currentLen <= sum(nNodesInLayerK(1:k))
                t = t + 1;
                xx(currentLen) = k;
                yy(currentLen) = t;
            else
                break
            end
        end
        t = 0;
    end
    if nNodesInLayerK(end) == 1
        yy(end) = nNodesInLayerK(end-1)/2;
    end
    maxHid = max(max(abs(weightsHid)));
    maxOut = max(abs(weightsOut));
    for k = 1:length(nNodesInLayerK)-1
        %last layer
        if k == length(nNodesInLayerK)-1
            for g = 1:nNodesInLayerK(end-1)
                p = [k k+1];
                o = [g yy(end)];
                shit = abs(weightsOut(g))/maxOut;
                shieet = (1-shit)/2;
                c = [(weightsOut(g)<=0)*(shit + shieet),(weightsOut(g)>0)*(shit + shieet),0];
                plot(uiax4,p,o,'color',c,'LineWidth',(shit + shieet)*7)
            end
            %first layer
        else
            for g = 1:nNodesInLayerK(k)-1
                for t = 0:nNodesInLayerK(k)-1
                    if g+t <= max(nNodesInLayerK)
                        if t==2 || (g==2 && t==1)
                        else
                            %from bottom to top
                            p = [k k+1];
                            o = [g g+t];
                            if g<2
                                shit = abs(weightsHid(2-t,end))/maxHid;
                                shieet = (1-shit)/2;
                                c = [(weightsHid(2-t,end)<=0)*(shit + shieet),(weightsHid(2-t,end)>0)*(shit + shieet),0];
                                plot(uiax4,p,o,'color',c,'LineWidth',(shit + shieet)*7)
                                hold(uiax4, 'on')
                            else
                                shit = abs(weightsHid(1,2))/maxHid;
                                shieet = (1-shit)/2;
                                c = [(weightsHid(1,2)<=0)*(shit + shieet),(weightsHid(1,2)>0)*(shit + shieet),0];
                                plot(uiax4,p,o,'color',c,'LineWidth',(shit + shieet)*7)
                            end
                        end
                        if t~=0
                            %from top to bottom
                            w = [k k+1];
                            v = [nNodesInLayerK(k)-g+1 nNodesInLayerK(k)-t-g+1];
                            if t<2
                                shit = abs(weightsHid(t,1))/maxHid;
                                shieet = (1-shit)/2;
                                c = [(weightsHid(t,1)<=0)*(shit + shieet),(weightsHid(t,1)>0)*(shit + shieet),0];
                                plot(uiax4,w,v,'color',c,'LineWidth',(shit + shieet)*7)
                            else
                                shit = abs(weightsHid(2,2))/maxHid;
                                shieet = (1-shit)/2;
                                c = [(weightsHid(2,2)<=0)*(shit + shieet),(weightsHid(2,2)>0)*(shit + shieet),0];
                                plot(uiax4,w,v,'color',c,'LineWidth',(shit + shieet)*7)
                            end
                        end
                    end
                end
            end
        end
    end
    %color neurons
    c = zeros(nNeurons,3);
    rander = randi(numSamples);
    for i = 1:nNeurons
        if i <= 3
            c(i,:) = [(inputs(i,rander)<=0)*((inputs(i,rander)/-1) + 0.5),(inputs(i,rander)>0)*((inputs(i,rander)) + 0.5),0];
        elseif i<=6 && i>3
            c(i,:) = [(outputHid(i-3,rander)<=0)*((outputHid(i-3,rander)/-1) + 0.5),(outputHid(i-3,rander)>0)*((outputHid(i-3,rander)) + 0.5),0];
        else
            c(i,:) = [(output<=0)*((output/-1) + 0.5),(output>0)*((output) + 0.5),0];
        end
    end
    %plot neurons
    scatter(uiax4,xx,yy,1000,c,'filled','MarkerEdgeColor','w','LineWidth',2);
    pause(10e-10)
end

%test neural network
trueInput1  = [1;-1;1];
shoudBeOne1 = tanh(weightsOut*[tanh(weightsHid*trueInput1);1])

trueInput2  = [-1;1;1];
shoudBeOne2 = tanh(weightsOut*[tanh(weightsHid*trueInput2);1])

falseInput1  = [1;1;1];
shoudBeMinusOne1 = tanh(weightsOut*[tanh(weightsHid*falseInput1);1])

falseInput2  = [-1;-1;1];
shoudBeMinusOne2 = tanh(weightsOut*[tanh(weightsHid*falseInput2);1])
