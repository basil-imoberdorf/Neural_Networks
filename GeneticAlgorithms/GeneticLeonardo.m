clear
close all
tic

fig1 = figure;
fig2 = figure;
fig1.Position = [1200, 100, 1200, 1200];
fig2.Position = [400, 300, 700, 700];
ax = axes(fig1);
ay = axes(fig2);

%import target image
tarim = imread('Pic\MonaFace.JPG');  % public-domain target (Mona Lisa face), similar size to the former image
tarim = compImag(tarim,1,'avg');
imshow(tarim,'border','tight','Parent',ay)
tarim = im2double(tarim).*255;

covMat=cov(tarim(:,:,1)');

imagesc(ax,covMat)
colormap(ax,jet)



% %debug low res image
% res = 60;
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

tardimm = size(tarim);
canvas = randi([0,255],tardimm(1),tardimm(2),tardimm(3)) ;
image(ax, uint8(canvas))
ax.XLim = [0.5 tardimm(2)+0.5];
ax.YLim = [0.5 tardimm(1)+0.5];
ax.Box = 'on';
ax.DataAspectRatio = [1,1,1];

%set up genetic algorithm properties
population = 10;    %initial population
generations = [0,200,1000,2000,3000,4000,5500];
mutation_rate = 0.01;
cubsiz = 47;        %size of marching cubes^^
Record_Video = 1;   %creates a MPEG-4 video of the simulation if set to 1
newHeight = ceil(tardimm(1)/cubsiz);
newLength = ceil(tardimm(2)/cubsiz);
Frame_Collection = struct('cdata', cell(1,2), 'colormap', cell(1,2)); %preallocate the array for the images
Frame_Index = 0;
swat = 200; %modulo operator

%capture first frames for movie
if Record_Video == 1
    Frame_Index = Frame_Index + 1;
    Frame_Collection(Frame_Index) = getframe(ax);
end
for z = 1:length(generations)-1
    for i = 0:newHeight-1
        for j = 0:newLength-1
            %get a cubsiz by cubsiz pixel segment of the whole image
            if j==newLength-1 && i==newHeight-1
                tarimm = tarim(1+i*cubsiz:tardimm(1),1+j*cubsiz:tardimm(2),:); %for the corner
            elseif i==newHeight-1
                tarimm = tarim(1+i*cubsiz:tardimm(1),1+j*cubsiz:j*cubsiz + cubsiz,:); %for the bootom edge
            elseif j==newLength-1
                tarimm = tarim(1+i*cubsiz:i*cubsiz + cubsiz,1+j*cubsiz:tardimm(2),:); %for the right edge
            else
                tarimm = tarim(1+i*cubsiz:i*cubsiz + cubsiz,1+j*cubsiz:j*cubsiz + cubsiz,:);   %for all in the middle
            end
            tardim = size(tarimm);
            fitness = zeros(population,3);
            parents = zeros(tardim(1),tardim(2),tardim(3),2);

            if z==1
                %make first generation of images
                im_pop = randi([0,255],tardim(1),tardim(2),tardim(3),population);
            else
                im_pop = zeros(tardim(1),tardim(2),tardim(3),population);
                for k = 1:population
                    im_pop(:,:,:,k) = canvas(1+i*cubsiz:i*cubsiz + tardim(1),1+j*cubsiz:j*cubsiz + tardim(2),:);% all other generations use their previous best material that is stored on the canvas
                end
            end

            for e = 1+generations(z):generations(z+1)
                %check fitness
                for k = 1:population
                    for n = 1:3 %for all 3 main colors
                        %use either absolute error
                        %fitness(k,n) = sum(abs(tarim(:,:,n) - im_pop(:,:,n,k)),'all');
                        %or use mean least square
                        fitness(k,n) = sum((tarimm(:,:,n) - im_pop(:,:,n,k)).^2,'all')/(tardim(1)*tardim(2));
                    end
                end

                %pick fittest individuals to become parents
                fitnessCopy = fitness;
                for r = 1:size(parents,4)
                    for n = 1:3
                        [value,index] = min(fitnessCopy(:,n));
                        parents(:,:,n,r) = im_pop(:,:,n,index);
                        fitnessCopy(fitnessCopy(:,n)==value,n) = max(fitnessCopy(:,n));
                    end
                end

                %do crossing over by overwriting random sized segements of one parent with the corresponding segments of the other parent.
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

                %                 if e==201
                %                     swat = 500;
                %                 end
                if mod(e,100)==0  %plot only every nth picture
                    %plot current best parent
                    if j==newLength-1 && i==newHeight-1
                        canvas(1+i*cubsiz:tardimm(1),1+j*cubsiz:tardimm(2),:) = parents(:,:,:,1);       %if it on the bottom right corner
                    elseif i==newHeight-1
                        canvas(1+i*cubsiz:tardimm(1),1+j*cubsiz:j*cubsiz + cubsiz,:) = parents(:,:,:,1);     %if it on the bottom edge
                    elseif j==newLength-1
                        canvas(1+i*cubsiz:i*cubsiz + cubsiz,1+j*cubsiz:tardimm(2),:) = parents(:,:,:,1);        %if it on the right edge
                    else
                        canvas(1+i*cubsiz:i*cubsiz + cubsiz,1+j*cubsiz:j*cubsiz + cubsiz,:) = parents(:,:,:,1);   %for all in the middle
                    end

                    image(ax, uint8(canvas));
                    %imshow(uint8(canvas),'Parent',ax)
                    ax.XLim = [0.5 tardimm(2)+0.5];
                    ax.YLim = [0.5 tardimm(1)+0.5];
                    ax.DataAspectRatio = [1,1,1];
                    ax.XTick = [];
                    ax.YTick = [];
                    ax.Box = 'on';
                    pause(10e-10)
                    %capture frames for movie
                    if Record_Video == 1
                        Frame_Index = Frame_Index + 1;
                        Frame_Collection(Frame_Index) = getframe(ax);
                    end
                end

                %add in mutations with heftigschte logische schalter who wirkt über 4Dimensionalem array
                logic4D = rand(tardim(1),tardim(2),3,population);
                im_pop(logic4D<=mutation_rate) = randi([0,255],sum(logic4D<=mutation_rate,'all'),1);

                %adaptive mutation manipulation
                if e==200
                    population = 10;
                    mutation_rate = 0.00095;
                elseif e>=1000 && e<2000
                    if e==1000
                        population = 15;
                        mutation_rate = 0.00085;
                    end
                    mutation_rate = mutation_rate*0.9999;
                elseif e>=2000 && e<3000
                    if e==2000
                        population = 20;
                        mutation_rate = 0.0006;
                    end
                    mutation_rate = mutation_rate*0.9998;
                elseif e>=3000 && e<4000
                    if e==3000
                        population = 30;
                        mutation_rate = 0.00069;
                    end
                    mutation_rate = mutation_rate*0.99964;
                elseif e==4000
                    population = 100;
                    mutation_rate = 0.00079;
                elseif e>=4500 && e<5500
                    mutation_rate = mutation_rate*0.9995;
                elseif e>=5500
                    if e==5500
                        population = 200;
                        mutation_rate = 0.00055;
                    end
                    mutation_rate = mutation_rate*0.9994;
                end
            end
        end
    end
end
Frame_Index;
%plot final result
rendertime = toc
image(ax, uint8(canvas))
ax.XLim = [0.5 tardimm(2)+0.5];
ax.YLim = [0.5 tardimm(1)+0.5];
ax.XTick = [];
ax.YTick = [];
ax.Box = 'on';
ax.DataAspectRatio = [1,1,1];

%make mp4 movie
if Record_Video == 1
    makeMovie('LeonardosWork\MonaFace_Leonardo.mp4',Frame_Collection,15)
end