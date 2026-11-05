function handle = drawNeuralNet(inputValues,hiddenValues,outputValues,weightsHid,weightsOut)
%Draws the current neural network. Green if +positive values. Red if 0 or negative
clf
nNodesInLayerK = [3,3,1];
nNeurons = sum(nNodesInLayerK);
nConnections = 0;
x = zeros(1,nNeurons);
y = zeros(1,nNeurons);
t = 0;

%generate right ammount of nConnections assuming each layer except for the putput has 1 bias neuron
for h = 1:length(nNodesInLayerK)-1
    if h+1 == length(nNodesInLayerK)
        nConnections = nConnections + nNodesInLayerK(h)*(nNodesInLayerK(h+1));
    else
        nConnections = nConnections + nNodesInLayerK(h)*(nNodesInLayerK(h+1)-1);
    end
end

%place neurons
for k = 1:length(nNodesInLayerK)
    for i = 1:nNeurons
        currentLen = length(nonzeros(x)) + 1;
        if currentLen <= sum(nNodesInLayerK(1:k))
            t = t + 1;
            x(currentLen) = k;
            y(currentLen) = t;
        else
            break
        end
    end
    t = 0;
end
if nNodesInLayerK(end) == 1
    y(end) = nNodesInLayerK(end-1)/2;
end

maxHid = max(max(abs(weightsHid)));
maxOut = max(abs(weightsOut));

for k = 1:length(nNodesInLayerK)-1
    %last layer 
    if k == length(nNodesInLayerK)-1
        for g = 1:nNodesInLayerK(end-1)
            p = [k k+1];
            o = [g y(end)];
            shit = abs(weightsOut(g))/maxOut;
            shieet = (1-shit)/2;
            c = [(weightsOut(g)<=0)*(shit + shieet),(weightsOut(g)>0)*(shit + shieet),0];
            plot(p,o,'color',c,'LineWidth',(shit + shieet)*7)
            %hold on
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
                            plot(p,o,'color',c,'LineWidth',(shit + shieet)*7)
                            hold on
                        else
                            shit = abs(weightsHid(1,2))/maxHid;
                            shieet = (1-shit)/2;
                            c = [(weightsHid(1,2)<=0)*(shit + shieet),(weightsHid(1,2)>0)*(shit + shieet),0];
                            plot(p,o,'color',c,'LineWidth',(shit + shieet)*7)
                            %hold on
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
                            plot(w,v,'color',c,'LineWidth',(shit + shieet)*7)
                            %hold on
                        else
                            shit = abs(weightsHid(2,2))/maxHid;
                            shieet = (1-shit)/2;
                            c = [(weightsHid(2,2)<=0)*(shit + shieet),(weightsHid(2,2)>0)*(shit + shieet),0];
                            plot(w,v,'color',c,'LineWidth',(shit + shieet)*7)
                            %hold on
                        end
                    end
                end
            end
        end
    end
end
%color neurons
c = zeros(nNeurons,3);
for i = 1:nNeurons
    if i <= 3
        c(i,:) = [(inputValues(i)<=0)*((inputValues(i)/-1) + 0.5),(inputValues(i)>0)*((inputValues(i)) + 0.5),0];
    elseif i<=6 && i>3
        c(i,:) = [(hiddenValues(i-3)<=0)*((hiddenValues(i-3)/-1) + 0.5),(hiddenValues(i-3)>0)*((hiddenValues(i-3)) + 0.5),0];
    else
        c(i,:) = [(outputValues<=0)*((outputValues/-1) + 0.5),(outputValues>0)*((outputValues) + 0.5),0];
    end
end
%plot neurons
handle = scatter(x,y,1000,c,'filled','MarkerEdgeColor','w','LineWidth',2);
%hold off
axis([0.5 max(nNodesInLayerK+1)-0.5 0.5 max(nNodesInLayerK+1)-0.5])
set(gca,'XTick',[])
set(gca,'YTick',[])
set(gca,'Box','on')
set(gca,'color','k')
