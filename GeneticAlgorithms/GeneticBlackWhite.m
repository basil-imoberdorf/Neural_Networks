clear
close all

%import target image
tarim = imread('Pic\MonaLisa.jpg');  % public-domain target
tarim = compImag(tarim,10,'avg');
tarim = rgb2gray(tarim);
imshow(tarim)
tarim = im2double(tarim).*255;

% % %debug low res image
% res = 30;
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
population = 250;
generations = 15000;
mutation_rate = 0.0005;
fitness = zeros(population,1);
parents = (zeros(tardim(1),tardim(2),2));
averageLoss = zeros(population,1);
maxLoss = (255*tardim(1)*tardim(2)); %when abs deviation is used

%make first generation of images
im_pop = (randi(256,tardim(1),tardim(2),population)) - 1;

for e = 1:generations
    %check fitness
    for k = 1:population
        %use either absolute error
        fitness(k) = sum(abs(tarim - im_pop(:,:,k)),'all');
        %or use mean least square
        %fitness(k) = sum((tarim - im_pop(:,:,:,k)).^2,'all')/(tardim(1)*tardim(2));
    end

    %pick best parents
    fitnessCopy = fitness; %copy to not damage anything
    for r = 1:size(parents,3)
        [value,index] = min(fitnessCopy);
        parents(:,:,r) = im_pop(:,:,index);
        fitnessCopy(fitnessCopy==value) = max(fitnessCopy);  %take the fittest then the second fittest thend the third fittest and so on
    end

    averageLoss(e) = mean(fitness)/maxLoss;
    if mod(e,10)==0
        %plot current best parent
        im_pop = uint8(im_pop);
        [~,index] = min(fitness);
        image(ax, im_pop(:,:,index))
        colormap(ax, 'hot')
        ax.XLim = [0.5 tardim(2)+0.5];
        ax.YLim = [0.5 tardim(1)+0.5];
        ax.DataAspectRatio = [1,1,1];
        im_pop = double(im_pop);

        %plot loss function
        plot(ay, 1:e,1 - nonzeros(averageLoss),'LineWidth',2,'Color','r')
        hold(ay, 'on')
        pause(10e-10)
    end

    %crossing over by overwriting random sized segements of one parent with the corresponding segments of the other parent.
    for k = 1:population
        %generate random cutting points
        topXSlicer = randi(tardim(2)-1)+1; %the +1 -1 stuff is there so that in case here is 1 below doesnt get to zero because of the -1
        bottomXSlicer = randi(topXSlicer-1);

        topYSlicer = randi(tardim(1)-1)+1;
        bottomYSlicer = randi(topYSlicer-1);

        %make the childs
        im_pop(:,:,k) = parents(:,:,1); %paste fittest parent as template
        im_pop(bottomYSlicer:topYSlicer,bottomXSlicer:topXSlicer,k) = parents(bottomYSlicer:topYSlicer,bottomXSlicer:topXSlicer,2); %replace random sized segments with corresponding parts from the second fittest
    end

    %add in mutations
    %     for k = 1:population
    %         im_pop(:,:,k)  = (im_pop(:,:,k) + rand().*(tarim - im_pop(:,:,k)));
    %     end

    for k = 1:population
        for i = 1:tardim(1)
            for j = 1:tardim(2)
                if mutation_rate > rand()
                    im_pop(i,j,k) = randi(256) - 1;
                end
            end
        end
    end
end

plot(ay, 1:e,1 - nonzeros(averageLoss),'LineWidth',2,'Color','r')
im_pop = uint8(im_pop);
[value,index] = min(fitness);
image(ax, im_pop(:,:,index))
ax.XLim = [0.5 tardim(2)+0.5];
ax.YLim = [0.5 tardim(1)+0.5];
ax.DataAspectRatio = [1,1,1];



