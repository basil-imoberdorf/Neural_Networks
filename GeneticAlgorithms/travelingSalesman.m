clear
close all

sz = get(groot,'ScreenSize');
fig = figure('Position',[sz(3)*0.45,sz(4)*0.02,sz(4)*0.73,sz(4)*0.73]);
ax = axes(fig,'Position',[0.07,0.07,0.9,0.9]); 

fig1 = figure('Position',[sz(3)*0.12,sz(4)*0.4,sz(3)*0.3,sz(4)*0.3]);
ay = axes(fig1,'Position',[0.08,0.1,0.90,0.85]); ay.XLabel.String = 'Generations [-]'; ay.YLabel.String = 'Distance [m]';


%set up genetic algorithm
pop_size = 100; %must be an even number since in each cross over step there are two offsprings generated
generations = 10000;
mutation_rate = 0.9;

%place at random cites within at 10km2 field
nCities = 30; %number of cities the salesman has to visit
City = randperm(10000,2*nCities)';
City = [City(1:nCities),City(nCities+1:end)]; %random nonrepeating coordinates

%presets
Pop = zeros(nCities,2,pop_size);
Loss = zeros(pop_size,1);
AverageLoss = zeros(generations,1);
MinLoss = zeros(generations,1);
StDevHigh = zeros(generations,1);
StDevLow = zeros(generations,1);
lowestLoss = Inf;
First_OffSpring = zeros(nCities,2);
Second_OffSpring = zeros(nCities,2);

%create initial population
for i = 1:pop_size
    t = randperm(nCities);
    Pop(:,:,i) = City(t,:);
end

%plot presets
hold(ax,'on')
hold(ay,'on')

plt2 = plot(ax,[City(:,1);City(1,1)],[City(:,2);City(1,2)],'c-','LineWidth',6); %overall best path
plt1 = plot(ax,[City(:,1);City(1,1)],[City(:,2);City(1,2)],'y-','LineWidth',2); %current best path

plt3 = plot(ay,0,0,'r-','LineWidth',2); %generation best distance, min loss
plt4 = plot(ay,0,0,'b-'); %generation average distance, average loss

plt5 = plot(ay,0,0,'Color',[0.6,0,1]); %upper standard deviation
plt6 = plot(ay,0,0,'Color',[0.6,0,1]); %lower standard deviation

sct = scatter(ax,City(:,1),City(:,2),500,'magenta','filled','MarkerEdgeColor','w');
ax.Color = 'k';
ax.XLim = [-1000,10000]; ay.XGrid = 'on';
ax.YLim = [-1000,10000]; ay.YGrid = 'on';
ax.XLabel.String = 'Map Width [m]'; ax.YLabel.String = 'Map Height [m]';

%do the genetics
for k = 1:generations
    pause(0)

    % FITNESS FUNCTION: calcualte distance traveled = loss
    for i = 1:pop_size
        Loss(i) = calcDist(Pop(:,:,i));
    end
    % mutation_rate  = mutation_rate*0.9999;

    if k==round(generations/4)
        mutation_rate = mutation_rate/2;
    elseif k==round(generations/2)
        % mutation_rate = mutation_rate/2;
    elseif k==round(generations/1.5)
        % mutation_rate = mutation_rate/2;
    end

    %choose fittest two parents
    [~,index] = mink(Loss,2);
    Parents = Pop(:,:,index);

    %show currents generation results
    drawRoute(plt1,Parents(:,:,1)) %draw current best
    
    if Loss(index(1)) < lowestLoss
        lowestLoss = Loss(index(1));
        plt2.XData = Parents(:,1,1); %plt2.XData = [Parents(:,1,1);Parents(1,1,1)];
        plt2.YData = Parents(:,2,1); %plt2.YData = [Parents(:,2,1);Parents(1,2,1)];
    end

    %plot diagnostics
    CurrenMean = mean(Loss);
    CurentStDev = std(Loss);
    AverageLoss(k) = CurrenMean;
    StDevHigh(k) = CurrenMean + CurentStDev;
    StDevLow(k) = CurrenMean - CurentStDev;
    MinLoss(k) = Loss(index(1));

    if mod(k,5)==0
        plt3.XData = 1:k;
        plt3.YData = MinLoss(1:k);
        plt4.XData = 1:k;
        plt4.YData = AverageLoss(1:k);

        plt5.XData = 1:k;
        plt5.YData = StDevHigh(1:k);
        plt6.XData = 1:k;
        plt6.YData = StDevLow(1:k);
    end

    %generate new popualtion and applying cross over and mutation
    for i = 1:pop_size/2


        % CROSSOVER
        cut_hi = randi([2,nCities]); %upper cut index
        cut_low = randi(cut_hi-1); %lower cut index
        num_remaining_cities = nCities - (cut_hi - cut_low + 1); %amount of cities do add in 

        % FirstSnipped = Parents(cut_low:cut_hi,:,1); 
        % Inter_Comp = [FirstSnipped;zeros(num_remaining_cities,2)];
        % [~, indices] = ismember(Parents(:,:,2), Inter_Comp, 'rows');
        % Pop(:,:,i) = [FirstSnipped;Parents(indices==0,:,2)];

        % % MUTATION
        % if mutation_rate<rand
        %     t = randperm(nCities,2);
        %     Pop(:,:,i) = swapCity(Parents(:,:,1),t); %swap two random cities
        % end

        First_OffSpring(cut_low:cut_hi,:) = Parents(cut_low:cut_hi,:,2); %first offspring gets the snipped from the second parent
        Second_OffSpring(cut_low:cut_hi,:) = Parents(cut_low:cut_hi,:,1); %second offspring gets the snipped from the first parent

        %fill remaining nubers
        Cities_From_First_Parent_Not_In_First_Offspring = Parents(:,:,1).*~ismember(Parents(:,:,1),First_OffSpring(cut_low:cut_hi,:));
        First_OffSpring([1:cut_low-1,cut_hi+1:nCities],:) = Cities_From_First_Parent_Not_In_First_Offspring([1:cut_low-1,cut_hi+1:nCities],:); %past the cityies wich are not yet in the snipped from the other parent in the offspring
        First_OffSpring(First_OffSpring==0) = nonzeros(Parents(:,:,1).*~ismember(Parents(:,:,1),First_OffSpring));

        Cities_From_Second_Parent_Not_In_Second_Offspring = Parents(:,:,2).*~ismember(Parents(:,:,2),Second_OffSpring(cut_low:cut_hi,:));
        Second_OffSpring([1:cut_low-1,cut_hi+1:nCities],:) = Cities_From_Second_Parent_Not_In_Second_Offspring([1:cut_low-1,cut_hi+1:nCities],:); %past the cityies wich are not yet in the snipped from the other parent in the offspring
        Second_OffSpring(Second_OffSpring==0) = nonzeros(Parents(:,:,2).*~ismember(Parents(:,:,2),Second_OffSpring));

        Pop(:,:,i*2-1) = First_OffSpring;
        Pop(:,:,i*2) = Second_OffSpring;


        % MUTATION
        if mutation_rate>rand
            t = randperm(nCities,2);
            Pop(:,:,i*2-1) = swapCity(First_OffSpring,t);
            Pop(:,:,i*2) = swapCity(Second_OffSpring,t); %swap two random cities

            t = randperm(nCities,2);
            Pop(:,:,i*2-1) = swapCity(First_OffSpring,t);
            Pop(:,:,i*2) = swapCity(Second_OffSpring,t); %swap two random cities

            if mutation_rate/2>rand
                t = randperm(nCities,2);
                Pop(:,:,i*2-1) = swapCity(First_OffSpring,t);
                Pop(:,:,i*2) = swapCity(Second_OffSpring,t); %swap two random cities

                t = randperm(nCities,2);
                Pop(:,:,i*2-1) = swapCity(First_OffSpring,t);
                Pop(:,:,i*2) = swapCity(Second_OffSpring,t); %swap two random cities
            end
        end
    end
end
mutation_rate
MinLoss(k)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %functions for closed sales route
% function Distance = calcDist(Pop)
% Distance = sum(sqrt(diff(Pop(:,1)).^2 + diff(Pop(:,1)).^2));
% Distance = Distance + sqrt((Pop(end,1)-Pop(1,1))^2 + (Pop(end,2)-Pop(1,2))^2);
% end
% 
% function drawRoute(plt,City)
% plt.XData = [City(:,1);City(1,1)];
% plt.YData = [City(:,2);City(1,2)];
% end
% 
% function Pop = swapCity(Pop,t)
% InterSave = Pop(t(1),:);
% Pop(t(1),:) = Pop(t(2),:);
% Pop(t(2),:) = InterSave;
% end

%functions for open ended sales route
function Distance = calcDist(Pop)
Distance = sum(sqrt(diff(Pop(:,1)).^2 + diff(Pop(:,1)).^2));
end

function drawRoute(plt,City)
plt.XData = [City(:,1)];
plt.YData = [City(:,2)];
end

function Pop = swapCity(Pop,t)
InterSave = Pop(t(1),:);
Pop(t(1),:) = Pop(t(2),:);
Pop(t(2),:) = InterSave;
end

