clear
close all

%import target image
tarim = imread('Pic\MonaLisa.jpg');
tarim = compImag(tarim,10,'avg');
imshow(tarim)
tarim = gpuArray(im2double(tarim).*255);

% % %debug low res image
% res = 50;
% loow = res/10;
% low = 4*res/10;
% hiigh = 9*res/10;
% high = 6*res/10;
% tarim = zeros(res,res,3);
% for p = 1:res
%     for q = 1:res
%         if p>low && p<high && q>loow && q<=hiigh|| q>low && q<high && p>loow && p<=hiigh
%             tarim(p,q,:) = [255,255,255];
%         else
%             %tarim(p,q,:) = [255,7,9];
%         end
%     end
% end

tarim = gpuArray(tarim);
tardim = size(tarim);

fig1 = figure;
fig2 = figure;
fig1.Position = [1000, 100, 700, 700];
fig2.Position = [1750, 300, 600, 600];
ax = axes(fig1);
ay = axes(fig2);

%set up genetic algorithm properties
population = 10;
generations = 5000;
mutation_rate = 0.01;
fitness = zeros(population,3);
parents = zeros(tardim(1),tardim(2),tardim(3),2,'gpuArray');
averageLoss = zeros(generations,1);
tarimMax = ones(tardim(1),tardim(2),3)*255;
maxLoss = sum((tarimMax.^2),'all')/(tardim(1)*tardim(2)); %when MSE is used
%maxLoss = (255*tardim(1)*tardim(2)); %when abs deviation is used
xSlicer = zeros(1,2);
ySlicer = zeros(1,2);
Slicer = zeros(2,2);

%make first generation of images
im_pop = (randi(256,tardim(1),tardim(2),tardim(3),population,'gpuArray')) - 1;

for e = 1:generations
    %check fitness
    for k = 1:population
        for n = 1:3 %for all 3 main colors
            %use either absolute error
            %fitness(k,n) = sum(abs(tarim(:,:,1) - im_pop(:,:,n,k)),'all');
            %or use mean least square
            fitness(k,n) = sum((tarim(:,:,n) - im_pop(:,:,n,k)).^2,'all')/(tardim(1)*tardim(2));
        end
    end

    %pick best parents
    fitnessCopy = fitness;
    for r = 1:size(parents,4)
        for n = 1:3
            [value,index] = min(fitnessCopy(:,n));
            parents(:,:,n,r) = im_pop(:,:,n,index);
            fitnessCopy(fitnessCopy(:,n)==value,n) = max(fitnessCopy(:,n));
        end
    end

    %adaptive mutation rate
    if e==200
        population = 10;
        mutation_rate = 0.00095;
    elseif e>=1000 && e<2000
        population = 15;
        if e==1000
            mutation_rate = mutation_rate*0.9995;
        end
        mutation_rate = mutation_rate*0.9999;
    elseif e>=2000 && e<3000
        population = 20;
        if e==2000
            mutation_rate = 0.001;
        end
        mutation_rate = mutation_rate*0.999;
    elseif e>=3000 && e<4000
        population = 30;
        mutation_rate = mutation_rate*0.9995;
    elseif e>=4000 && e<6000
        population = 100;
        if e==4000
            mutation_rate = 0.0009;
        end
        mutation_rate = mutation_rate*0.999;
    end

    %crossing over by overwriting random sized segements of one parent with the corresponding segments of the other parent.
    for k = 1:population
        %generate random cutting points
        Slicer(1,:) = randi(tardim(2)-1,1,2)+1; %top x and y cutting points

        Slicer(2,1) = randi(Slicer(1,1)-1);     %bottom x cutting point
        Slicer(2,2) = randi(Slicer(1,2)-1);     %bottom y cutting point

        %make the childs
        im_pop(:,:,:,k) = parents(:,:,:,1); %paste fittest parent as template
        im_pop(Slicer(2,2):Slicer(1,2),Slicer(2,1):Slicer(1,1),:,k) = parents(Slicer(2,2):Slicer(1,2),Slicer(2,1):Slicer(1,1),:,2); %replace random sized segments with corresponding parts from the second fittest
    end

    averageLoss(e) = mean(fitness,'all');
    if mod(e,20)==0
        %plot current best parent
        parents = uint8(parents);
        image(ax, parents(:,:,:,1))
        parents = double(parents);

        plot(ay, 1:e,1 - nonzeros(averageLoss)./maxLoss,'LineWidth',2,'Color','r')
        tic
        ax.XLim = [0.5 tardim(2)+0.5];
        ax.YLim = [0.5 tardim(1)+0.5];
        ax.DataAspectRatio = [1,1,1];
        ay.YLim = [.9 1];
        ay.XLim = [0 generations];
        ay.XGrid = 'on';
        ay.YGrid = 'on';
        ay.XMinorGrid = 'on';
        ay.YMinorGrid = 'on';
        toc
        pause(10e-10)
    end

    %add in mutations with heftigschte logische schalter who wirkt über 4Dimensionalem array
    logic4D = rand(tardim(1),tardim(2),3,population,'gpuArray');
    ran = randi([0,255],tardim(1),tardim(2),3,population,'gpuArray');
    im_pop = (logic4D>mutation_rate).*im_pop  +  (logic4D<=mutation_rate).*ran;
end

%plot final result
plot(ay, 1:e,1 - nonzeros(averageLoss(:,1))./maxLoss,'LineWidth',2,'Color','r')
parents = uint8(parents);
image(ax, parents(:,:,:,1))


