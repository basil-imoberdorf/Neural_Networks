clear
close all

%import target image
tarim = imread('Pic\VanGogh.jpg');  % public-domain target
tarim = compImag(tarim,40,'avg');
imshow(tarim)
tarim = im2double(tarim).*255;

% % %debug low res image
% res = 40;
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
%             tarim(p,q,:) = [255,7,9];
%         end
%     end
% end

tardim = size(tarim);

fig1 = figure;
fig2 = figure;
fig1.Position = [1000, 100, 700, 700];
fig2.Position = [1750, 300, 500, 500];
ax = axes(fig1);
ay = axes(fig2);

%set up genetic algorithm properties
population = 30;
generations = 6000;
mutation_rate = 0.009;
fitness = zeros(population,1);
parents = (zeros(tardim(1),tardim(2),tardim(3),2));
averageLoss = zeros(generations,1);
tarimMax = ones(tardim(1),tardim(2),3)*255;
maxLoss = sum((tarimMax.^2),'all')/(tardim(1)*tardim(2)); %when MSE is used
%maxLoss = (3*255*tardim(1)*tardim(2)); %when abs deviation is used

%make first generation of images
im_pop = (randi(256,tardim(1),tardim(2),tardim(3),population)) - 1;

for e = 1:generations
    %check fitness
    for k = 1:population
        %use either absolute error
        %fitness(k) = sum(abs(tarim - im_pop(:,:,:,k)),'all');
        %or use mean least square
        fitness(k) = sum((tarim - im_pop(:,:,:,k)).^2,'all')/(tardim(1)*tardim(2));
    end

    %pick best parents
    fitnessCopy = fitness; %copy to not damage anything
    for r = 1:size(parents,4)
        [value,index] = min(fitnessCopy);
        parents(:,:,:,r) = im_pop(:,:,:,index);
        fitnessCopy(fitnessCopy==value) = max(fitnessCopy);  %take the fittest then the second fittest thend the third fittest and so on
    end

    %adaptive mutation rate
    if e<4000
        mutation_rate = mutation_rate*0.9995;
    else
        population = 400;
        mutation_rate = mutation_rate*0.99995;
    end


    %crossing over by overwriting random sized segements of one parent with the corresponding segments of the other parent.
    for k = 1:population
        %generate random cutting points
        topXSlicer = randi(tardim(2)-1)+1; %the +1 -1 stuff is there so that in case here is 1 below doesnt get to zero because of the -1
        bottomXSlicer = randi(topXSlicer-1);

        topYSlicer = randi(tardim(1)-1)+1;
        bottomYSlicer = randi(topYSlicer-1);

        %make the childs
        im_pop(:,:,:,k) = parents(:,:,:,1); %paste fittest parent as template
        im_pop(bottomYSlicer:topYSlicer,bottomXSlicer:topXSlicer,:,k) = parents(bottomYSlicer:topYSlicer,bottomXSlicer:topXSlicer,:,2); %replace random sized segments with corresponding parts from the second fittest
    end

    averageLoss(e) = mean(fitness);
    if mod(e,20)==0
        %plot current best parent
        im_pop = uint8(im_pop);
        [~,index] = min(fitness);
        image(ax, im_pop(:,:,:,index))
        ax.XLim = [0.5 tardim(2)+0.5];
        ax.YLim = [0.5 tardim(1)+0.5];
        %ay.YLim = [0 1];
        ax.DataAspectRatio = [1,1,1];
        im_pop = double(im_pop);

        plot(ay, 1:e,1 - nonzeros(averageLoss)./maxLoss,'LineWidth',2,'Color','r')
        hold(ay, 'on')
        pause(10e-10)
    end

    %add in mutations
    logic4D = rand(tardim(1),tardim(2),3,population);
    im_pop(logic4D<=mutation_rate) = randi([0,255],sum(logic4D<=mutation_rate,'all'),1);
end

plot(ay, 1:e,1 - nonzeros(averageLoss)./maxLoss,'LineWidth',2,'Color','r')
%ay.YLim = [0 1];
im_pop = uint8(im_pop);
[value,index] = min(fitness);
image(ax, im_pop(:,:,:,index))
ax.XLim = [0.5 tardim(2)+0.5];
ax.YLim = [0.5 tardim(1)+0.5];
ax.DataAspectRatio = [1,1,1];


