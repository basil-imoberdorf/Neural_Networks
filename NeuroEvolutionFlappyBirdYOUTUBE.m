clear
close all

%%%%%%%%%%%%%%%%%%%%%%%%%% create canvases %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig3 = figure;
fig3.Position = [200, 200, 2000, 1000];

ax = axes(fig3);
ax.Position = [0.4 0 .6 1];
ax.YLim = [0 10];

%prepare plot for the dignostics
ay = axes(fig3);
%ay.Position = [.1 .1 .8 .6];
ay.Position = [.05 .075 .3 .45];
polty = plot(ay, 1, 1, 1, 1,'LineWidth',2);
farb = ['r';'b'];
colororder(ay,farb)
lgd = legend(ay, ['max';'avg'],'Location','NW','Interpreter','latex');
lgd.FontSize = 14;
lgd.Box = 'off';
xlabel(ay, 'Generations','FontSize',20,'Interpreter','latex')
ylabel(ay, 'Fitness','FontSize',20,'Interpreter','latex')
ay.XLim = [0 1];
az = axes(fig3,'XTick',[],'YTick',[],'Box','on');
az.Position = [.05 .6 .3 .35];
papyrus = text(.05, .9,'\underline{Exploring Phase}','Clipping', 'on','hittest', 'off','HorizontalAlignment','left','FontSize',22,'Parent',az,'Interpreter','latex');
tex = text(.05, .7,'','Clipping', 'on','hittest', 'off','HorizontalAlignment','left','FontSize',18,'Parent',az,'VerticalAlignment','baseline');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%set up algorithm parameters
designVector = [3,3,1]; %theis defines the nerual network -> [Inputs,1.LayerHiddenNodes,Outpus]
population = 1000;
generations = 100;
mutation_rate = 0.1;
pipeWidth = 3;   %defines the gap size between de pipes
simspeed = 1;
levelLength = 200; % defines the ammount of pillars generated
gravity_acc = 16.81;
time_step = 0.05;
bgr = 1; %defines the distance from the boarder in witch no pipe gaps are allowed
pild = 6;
flapSpeed = 5; %defines how much yspeed the bird gains wehn flapping
birdPos = zeros(population,2);
birdSpeed = zeros(population,2);
fitness = zeros(population,1);
pillarInfo = zeros(levelLength,2); %contains all necessary info about the pillars/pipes
t = ones(population,1); %index to scan the pillarInfo for each bird
maxDeviation = 10-bgr-pipeWidth/2;
avgFitness = zeros(generations+1,1); %for diagonstics
maxFitness = zeros(generations+1,1);
avgFitness(1) = 0.01;
maxFitness(1) = 0.01;
farbe = linspace(0,1,population);
axlim = 0;  %lower initial axes limit
secounter  = 0; %counter to use in the while loop
breakout = 0; %logical value to end the trainig and proceed to use the fittest bird
highscore = 0;
a = -1;

%create population of networks
weightsPop = cell(population,length(designVector)-1);
for i = 1:population
    weightsPop(i,:) = genWeightsNN(designVector);
    birdPos(i,:)   = [3+randn()/3,7+randn()/3];
    birdSpeed(i,:) = [3,0];
end

handle = scatter(ax, birdPos(:,1),birdPos(:,2),1000,farbe,'filled','Marker','>','MarkerEdgeColor','k');
colormap(ax, 'autumn')
ax.Color = [0.4 0.4 1];
ax.YTick = [];
ax.XTick = [];
axis(ax, [0 12 0 10])

%make clouds
for i = 1:(pild*levelLength)/10
    x = randi([8,9]);
    rectangle(ax, 'Position',[(i-1)*10+rand*2 x+rand randi(2)/2+rand (10-x)/3],'FaceColor',[0.9 1 1],'EdgeColor','k','Curvature',[rand,rand]);
end
rectangle(ax, 'Position',[5+pild*levelLength+rand*2 2.5 5 5],'FaceColor',[1 1 0.2],'EdgeColor','k','Curvature',[.9,.9]); %sun at the end

%make pillars with increasing difficulty
for i = 1:10
    x = randi([bgr,round(10-bgr-pipeWidth)]);
    rectangle(ax, 'Position',[i*pild 0 2 x],'FaceColor',[0.4 1 0.4],'EdgeColor','k');
    rectangle(ax, 'Position',[i*pild x+pipeWidth 2 10-pipeWidth-x],'FaceColor',[0.4 1 0.4],'EdgeColor','k');
    pillarInfo(i,:) = [x,i*pild]; %contains [minPillar1, xPillarStartPos; ...] info defines together with pipeWidth all vertices of the pipes/pillars-rectangles
end
pipeWidth = 2.5;
bgr = 1;
for i = 11:20
    x = randi([bgr,round(10-bgr-pipeWidth)]);
    rectangle(ax, 'Position',[i*pild 0 2 x],'FaceColor',[0.4 1 0.4],'EdgeColor','k');
    rectangle(ax, 'Position',[i*pild x+pipeWidth 2 10-pipeWidth-x],'FaceColor',[0.4 1 0.4],'EdgeColor','k');
    pillarInfo(i,:) = [x,i*pild]; %contains [minPillar1, xPillarStartPos; ...] info defines together with pipeWidth all vertices of the pipes/pillars-rectangles
end
pipeWidth = 2;
bgr = 1;
for i = 21:30
    x = randi([bgr,round(10-bgr-pipeWidth)]);
    rectangle(ax, 'Position',[i*pild 0 2 x],'FaceColor',[0.4 1 0.4],'EdgeColor','k');
    rectangle(ax, 'Position',[i*pild x+pipeWidth 2 10-pipeWidth-x],'FaceColor',[0.4 1 0.4],'EdgeColor','k');
    pillarInfo(i,:) = [x,i*pild]; %contains [minPillar1, xPillarStartPos; ...] info defines together with pipeWidth all vertices of the pipes/pillars-rectangles
end
pipeWidth = 1.75;
bgr = 2;
for i = 31:40
    x = randi([bgr,round(10-bgr-pipeWidth)]);
    rectangle(ax, 'Position',[i*pild 0 2 x],'FaceColor',[0.4 1 0.4],'EdgeColor','k');
    rectangle(ax, 'Position',[i*pild x+pipeWidth 2 10-pipeWidth-x],'FaceColor',[0.4 1 0.4],'EdgeColor','k');
    pillarInfo(i,:) = [x,i*pild]; %contains [minPillar1, xPillarStartPos; ...] info defines together with pipeWidth all vertices of the pipes/pillars-rectangles
end
pipeWidth = 1.5;
bgr = 2;
for i = 41:levelLength
    x = randi([bgr,round(10-bgr-pipeWidth)]);
    rectangle(ax, 'Position',[i*pild 0 2 x],'FaceColor',[0.4 1 0.4],'EdgeColor','k');
    rectangle(ax, 'Position',[i*pild x+pipeWidth 2 10-pipeWidth-x],'FaceColor',[0.4 1 0.4],'EdgeColor','k');
    pillarInfo(i,:) = [x,i*pild]; %contains [minPillar1, xPillarStartPos; ...] info defines together with pipeWidth all vertices of the pipes/pillars-rectangles
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Into the animation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for e = 1:generations
    %mutation_rate = mutation_rate * 0.9
    fitness = zeros(population,1); %reset fitness container for current generation

    while max(birdPos(:,1))>0 && breakout==0
        tic
        %adjoust pipeWidth for correct collisin detection
        [~,idefix] = max(birdPos(:,1));
        if a==-1
            a = 0;
            pipeWidth = 3;
            maxDeviation = 10-bgr-pipeWidth/2;
        elseif t(idefix)==11 && a==0
            a = 1;
            pipeWidth = 2.5;
            maxDeviation = 10-bgr-pipeWidth/2;
        elseif  t(idefix)==21 && a==1
            a = 2;
            pipeWidth = 2;
            maxDeviation = 10-bgr-pipeWidth/2;
        elseif  t(idefix)==31 && a==2
            a = 3;
            pipeWidth = 1.75;
            maxDeviation = 10-bgr-pipeWidth/2;
        elseif  t(idefix)==41 && a==3
            a = 4;
            pipeWidth = 1.5;
            bgr = 2;
            maxDeviation = 10-bgr-pipeWidth/2;
        end
        for i = 1:population
            if  birdPos(i,1)>0 %only if xpos is positive the bird can enter the calculation  %only if it has not collided d.h xpos is positive, the bird can enter the calculation, this saves time and is really cool shieet cause it prevents the collided birds from getting moved after collision so they remain outside of calculation
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Decision making time, flap or no flap thats the question %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %Input = [birdPos(i,2)/1;(birdSpeed(i,2)/max(max(abs(birdSpeed))));(pillarInfo(t(i),2)-birdPos(i,1))/10;pillarInfo(t(i),1)/1;(pillarInfo(t(i),1)+pipeWidth)/1];
                Input = [birdPos(i,2)-5;pillarInfo(t(i),1)-.16+pipeWidth-5;pillarInfo(t(i),1)-.16-5]; %inputs are not normlized like so: [yPositionBird; ySpeedBird; xDistanceToPipe; yPipeLowLim; yPipeTopLim]
                [Outputs,hiddenOutputsCell] = feedForwardNN(Input,weightsPop(i,:));
                if Outputs>0 && birdSpeed(i,1)>0
                    birdSpeed(i,2) = flapSpeed; %flap if NN outputs 1
                end

                %check for colision with bootom and top
                if (birdPos(i,2)>=9.84 || birdPos(i,2)<=0.16)
                    %fitness(i) = birdPos(i,1);
                    birdPos(i,:) = [-10,7]; %make bird unable to acces the functions by making his xposition values negative,remove the bird from the sky by force if he collided with something
                end

                %check for colision with rectangles
                if birdPos(i,1)>=pillarInfo(t(i),2)-.16 && birdPos(i,1)<=pillarInfo(t(i),2)+2
                    if birdPos(i,2)<=pillarInfo(t(i),1)+.16 || birdPos(i,2)>=pillarInfo(t(i),1)+pipeWidth-.16
                        %fitness(i) = birdPos(i,1); %distance based fitness
                        birdPos(i,:) = [-10,7]; %remove the bird from the sky by force if he collided with a pillar
                    end
                elseif birdPos(i,1)>pillarInfo(t(i),2)+2  %the 2 is the width of the pillar, after surpassing the current pillar the next one gets loaded via incrementing t(of that bird)
                    t(i) = t(i) + 1;
                    %check termination criterion
                    if t(i)>levelLength
                        t(i) = levelLength; %now idea why this is necessary but without this the reset after reaching the end does not work
                        birdPos(i,:) = [-10,7];
                        %breakout = 1; %if one bird makes it to the end leave the loop
                    end
                end

                %apply constant horizontal speed
                birdPos(i,1) = birdPos(i,1) + birdSpeed(i,1)*time_step;

                %let gravity act
                birdSpeed(i,2) = birdSpeed(i,2) - gravity_acc*time_step;
                birdPos(i,2) = birdPos(i,2) + birdSpeed(i,2)*time_step;

                %fitness according to distance to optimal height
                heightDeviation = abs(pillarInfo(t(i),1)+pipeWidth/2 - birdPos(i,2)); %measure deviation from ideal height
                fitness(i) =  fitness(i) + maxDeviation-heightDeviation; %rewards the bird for positioning himself at the height of the gap between the upcomming pipes

                %write generation data
                livingBirds = sum(birdPos(:,1)>0);
                score = max(t);
                %update Highscore
                if score>highscore
                    highscore = score;
                end
                tex.String = sprintf('Airborne Birds:       %i\n\nScore:                     %i\n\nHighscore:              %i\n\nMutation Rate:        %.1f%%',livingBirds,score,highscore,100*mutation_rate);
            end
        end

        %update animation frame
        secounter = secounter + 1; %update only if simspeed matches secounter. this allows to steer the speed of the simulation since most time is wasted for drwaing the figure
        axlim = axlim + 3*time_step;
        if mod(secounter,simspeed)==0
            %update bird position scatter points
            handle.XData(:) = birdPos(:,1)';
            handle.YData(:) = birdPos(:,2)';

            %move axis to keep the bird at the same xPosition
            ax.XLim = [axlim axlim+12];
            pause(0)
        end
        toc
    end

    %kick in booster after few generations
    if e==3
        simspeed = 10;
        papyrus.String = '\underline{Boost Evolution}';
        population = 50;
        birdPos = zeros(population,2);
        birdSpeed = zeros(population,2);
        delete(handle)
        farbe = linspace(0,1,population);
        handle = scatter(ax, birdPos(:,1),birdPos(:,2),1000,farbe,'filled','Marker','>','MarkerEdgeColor','k');
        colormap(ax, 'autumn')
        ax.Color = [0.4 0.4 1];
        ax.YTick = [];
        ax.XTick = [];
        %remake clouds
        for i = 1:(pild*levelLength)/10
            x = randi([8,9]);
            rectangle(ax, 'Position',[(i-1)*10+rand*2 x+rand randi(2)/2+rand (10-x)/3],'FaceColor',[0.9 1 1],'EdgeColor','k','Curvature',[rand,rand]);
        end
        rectangle(ax, 'Position',[5+pild*levelLength+rand*2 2.5 5 5],'FaceColor',[1 1 0.2],'EdgeColor','k','Curvature',[.9,.9]); %sun at the end
        t = ones(population,1);
        ax.XLim = [0 12];
    end
    %enlarge population when end is reached and reduce mutation rate
    if max(t)==levelLength
        axlim = axlim + 1;
        ax.XLim = [axlim axlim+12];
        pause(.5)
        mutation_rate = 0.001;
        population = 200;
        birdPos = zeros(population,2);
        birdSpeed = zeros(population,2);
        delete(handle)
        farbe = linspace(0,1,population);
        handle = scatter(ax, birdPos(:,1),birdPos(:,2),1000,farbe,'filled','Marker','>','MarkerEdgeColor','k');
        colormap(ax, 'autumn')
        ax.Color = [0.4 0.4 1];
        ax.YTick = [];
        ax.XTick = [];
        %remake clouds
        for i = 1:(pild*levelLength)/10
            x = randi([8,9]);
            rectangle(ax, 'Position',[(i-1)*10+rand*2 x+rand randi(2)/2+rand (10-x)/3],'FaceColor',[0.9 1 1],'EdgeColor','k','Curvature',[rand,rand]);
        end
        rectangle(ax, 'Position',[5+pild*levelLength+rand*2 2.5 5 5],'FaceColor',[1 1 0.2],'EdgeColor','k','Curvature',[.9,.9]); %sun at the end
        t = ones(population,1);
        simspeed = 1;
        papyrus.String = '\underline{Bird Swarm Test}';
    end

    %breed next generation and add mutations
    [value,index] = max(fitness); %find fittest bird
    for k = 1:population
        weightsPop(k,:) = weightsPop(index,:); %paste fittest bird as template
        for i = 1:size(weightsPop,2)
            %apply mutations
            logic = rand(size(weightsPop{k,i},1),size(weightsPop{k,i},2))<mutation_rate; %brand new weights if mutation hits
            weightsPop{k,i} = weightsPop{k,i}.*~logic +  randn(size(weightsPop{k,i},1),size(weightsPop{k,i},2)).*logic;
            %             logic = rand(size(weightsPop{k,i},1),size(weightsPop{k,i},2))<mutation_rate; %adjoust weights slightly if mutation hits
            %             weightsPop{k,i}(logic) = weightsPop{k,i}(logic) + randn()/10;
        end
    end

    %plot fitness evolution
    avgFitness(e+1) = mean(fitness); %for diagonstics
    maxFitness(e+1) = value;
    polty(1).XData = 0:e;
    polty(2).XData = 0:e;
    polty(1).YData = nonzeros(maxFitness(1:e+1));
    polty(2).YData = nonzeros(avgFitness(1:e+1));
    ay.XLim = [0 e];
    ay.YLim = [0 max(maxFitness)];
    pause(0)

    %check for termination criterion and take the best bird DNA
    if breakout==1 || e==generations
        [value,indexus] = max(fitness);
        bestBirdGenes = weightsPop(indexus,:);
        break
    end

    %reset values for next run
    rend = randn(population,1)/3;
    birdPos(:,1) = 3 + rend;
    birdPos(:,2) = 7 + rend;
    handle.XData(:) = 3 + rend;
    handle.YData(:) = 7 + rend;
    birdSpeed(:,1) = 3;
    birdSpeed(:,2) = 0;
    t(:) = 1;
    axlim = 0;
    ax.XLim = [axlim axlim+12];
    ax.YLim = [0 10];
    a=-1;

    %make new pillars
    rect=(findall(ax,'Type', 'Rectangle'));
    if length(rect)>=2*levelLength
        delete(rect(1:2*levelLength))
    end

    %     for i = 1:levelLength
    %         x = randi([bgr,round(10-bgr-pipeWidth)]);
    %         rectangle(ax, 'Position',[i*pild 0 2 x(1)],'FaceColor',[0.4 1 0.4],'EdgeColor','k');
    %         rectangle(ax, 'Position',[i*pild x(1)+pipeWidth 2 10-pipeWidth-x(1)],'FaceColor',[0.4 1 0.4],'EdgeColor','k');
    %         pillarInfo(i,:) = [x(1),i*pild]; %contains [minPillar1, xPillarStartPos; ...] info defines together with pipeWidth all vertices of the pipes/pillars-rectangles
    %     end
    pipeWidth = 3;
    bgr = 1;
    for i = 1:10
        x = randi([bgr,round(10-bgr-pipeWidth)]);
        rectangle(ax, 'Position',[i*pild 0 2 x],'FaceColor',[0.4 1 0.4],'EdgeColor','k');
        rectangle(ax, 'Position',[i*pild x+pipeWidth 2 10-pipeWidth-x],'FaceColor',[0.4 1 0.4],'EdgeColor','k');
        pillarInfo(i,:) = [x,i*pild]; %contains [minPillar1, xPillarStartPos; ...] info defines together with pipeWidth all vertices of the pipes/pillars-rectangles
    end
    pipeWidth = 2.5;
    bgr = 1;
    for i = 11:20
        x = randi([bgr,round(10-bgr-pipeWidth)]);
        rectangle(ax, 'Position',[i*pild 0 2 x],'FaceColor',[0.4 1 0.4],'EdgeColor','k');
        rectangle(ax, 'Position',[i*pild x+pipeWidth 2 10-pipeWidth-x],'FaceColor',[0.4 1 0.4],'EdgeColor','k');
        pillarInfo(i,:) = [x,i*pild]; %contains [minPillar1, xPillarStartPos; ...] info defines together with pipeWidth all vertices of the pipes/pillars-rectangles
    end
    pipeWidth = 2;
    bgr = 1;
    for i = 21:30
        x = randi([bgr,round(10-bgr-pipeWidth)]);
        rectangle(ax, 'Position',[i*pild 0 2 x],'FaceColor',[0.4 1 0.4],'EdgeColor','k');
        rectangle(ax, 'Position',[i*pild x+pipeWidth 2 10-pipeWidth-x],'FaceColor',[0.4 1 0.4],'EdgeColor','k');
        pillarInfo(i,:) = [x,i*pild]; %contains [minPillar1, xPillarStartPos; ...] info defines together with pipeWidth all vertices of the pipes/pillars-rectangles
    end
    pipeWidth = 1.75;
    bgr = 1;
    for i = 31:40
        x = randi([bgr,round(10-bgr-pipeWidth)]);
        rectangle(ax, 'Position',[i*pild 0 2 x],'FaceColor',[0.4 1 0.4],'EdgeColor','k');
        rectangle(ax, 'Position',[i*pild x+pipeWidth 2 10-pipeWidth-x],'FaceColor',[0.4 1 0.4],'EdgeColor','k');
        pillarInfo(i,:) = [x,i*pild]; %contains [minPillar1, xPillarStartPos; ...] info defines together with pipeWidth all vertices of the pipes/pillars-rectangles
    end
    pipeWidth = 1.5;
    bgr = 2;
    for i = 41:levelLength
        x = randi([bgr,round(10-bgr-pipeWidth)]);
        rectangle(ax, 'Position',[i*pild 0 2 x],'FaceColor',[0.4 1 0.4],'EdgeColor','k');
        rectangle(ax, 'Position',[i*pild x+pipeWidth 2 10-pipeWidth-x],'FaceColor',[0.4 1 0.4],'EdgeColor','k');
        pillarInfo(i,:) = [x,i*pild]; %contains [minPillar1, xPillarStartPos; ...] info defines together with pipeWidth all vertices of the pipes/pillars-rectangles
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Best Bird Run &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%reset values for run of the best bird
birdPos(indexus,1) = 3;
birdPos(indexus,2) = 7;
birdSpeed(indexus,1) = 3;
birdSpeed(indexus,2) = 0;
handle.XData(indexus) =  3;
handle.YData(indexus) =  7;
t(indexus) = 1;
axlim = 0;
secounter = 0;
simspeed = 1; %2 makes it twice as fast
ax.XLim = [axlim axlim+12];

while birdPos(indexus,1)>0 || t(indexus)<=levelLength
    %let the best bird brain in
    Input = [birdPos(indexus,2)-5;pillarInfo(t(indexus),1)-5]; %inputs are not normlized like so: [yPositionBird; ySpeedBird; xDistanceToPipe; yPipeLowLim; yPipeTopLim]
    [Outputs,~] = feedForwardNN(Input,weightsPop(indexus,:));
    if Outputs>0
        birdSpeed(indexus,2) = flapSpeed; %flap if NN outputs 1
    end

    %check for colision with bootom cand top
    if (birdPos(indexus,2)>=9.85 || birdPos(indexus,2)<=0.15) %only if xpos is positive the bird can enter the calculation
        %fitness(i) = birdPos(i,1);
        birdPos(indexus,:) = [-10,7]; %make bird unable to acces the functions by making his xposition values negative,remove the bird from the sky by force if he collided with something
    end

    %check for colision with rectangles
    if birdPos(indexus,1)>=pillarInfo(t(indexus),2)-.15 && birdPos(indexus,1)<=pillarInfo(t(indexus),2)+2
        if birdPos(indexus,2)<=pillarInfo(t(indexus),1)+.15 || birdPos(indexus,2)>=pillarInfo(t(indexus),1)+pipeWidth-.15
            %fitness(i) = birdPos(i,1); %distance based fitness
            birdPos(indexus,:) = [-10,7]; %remove the bird from the sky by force if he collided with something
        end
    elseif birdPos(indexus,1)>pillarInfo(t(indexus),2)+pipeWidth
        t(indexus) = t(indexus) + 1;
    end

    %apply constant horizontal speed
    birdPos(indexus,1) = birdPos(indexus,1) + birdSpeed(indexus,1)*time_step;

    %let gravity act on the beast bird
    birdSpeed(indexus,2) = birdSpeed(indexus,2) - gravity_acc*time_step;
    birdPos(indexus,2) = birdPos(indexus,2) + birdSpeed(indexus,2)*time_step;

    %plot bird
    secounter = secounter + 1;
    if mod(secounter,simspeed)==0
        handle.XData(indexus) = birdPos(indexus,1);
        handle.YData(indexus) = birdPos(indexus,2);
    end

    %move axis to keep the bird at the same xPosition
    axlim = axlim + 3*time_step;
    ax.XLim = [axlim axlim+12];
    pause(1/60);
end