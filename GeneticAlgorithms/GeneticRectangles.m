clear
close all

fig1 = figure;
fig2 = figure;
fig3 = figure;
fig2.Position = [1400,300,500,500];
fig3.Position = [900,300,500,500];
ay = axes(fig2);
az = axes(fig3);

%import target image
tarim = imread('Pic\PietMondrian.JPG');  % public-domain target, Mondrian suits the rectangle primitives
tarim = compImag(tarim,1,'avg');
imshow(tarim,'Border','tight','Parent',ay)

image(ay ,tarim)
tarim = im2double(tarim).*255;
tardim = size(tarim);
fig1.Position = [400,300,tardim(2),tardim(1)];
ax = axes(fig1,"XTick",[],"YTick",[],"Box",'on','Position',[0 0 1 1],'DataAspectRatio',[1,1,1],'XLim',[0 tardim(2)],'YLim',[0 tardim(1)],'Color',[.5 .5 .5],'YColor',[.5 .5 .5],'XColor',[.5 .5 .5]);

population = 100;
strokes = 1;
generations = 2000;
Record_Video = 0;   %creates a MPEG-4 video of the simulation if set to 1
Capture_Ratio = 1; %after how many frames one gets saved for video
xmax = tardim(2);
ymax = tardim(1);
fitness = zeros(population,1);
im_pop = ones(tardim(1),tardim(2),3,population).*128;
parents = ones(tardim(1),tardim(2),3,2).*128; %preallocate two parents
previous_minFitness = 10e10;
patchLim  = max(tardim)/3;
cdata = zeros(tardim(1),tardim(2),3);
calima = imshow(uint8(im_pop(:,:,:,1)),'Parent',ax);
resima = imshow(uint8(parents(:,:,:,1)),'Border','tight','Parent',az);
Frame_Collection = struct('cdata', cell(1,generations/Capture_Ratio), 'colormap', cell(1,generations/Capture_Ratio)); %preallocate the array for the images
Frame_Index = 0;
tot  = 0;



%capture first frames for movie
if Record_Video == 1
    Frame_Index = Frame_Index + 1;
    Frame_Collection(Frame_Index) = getframe(az);
end

%generate first population
for k = 1:population
    %delet previous strokes
    X = findall(ax,'Type', 'Rectangle');
    Y = findall(ax,'Type', 'Text');
    delete(X)
    delete(Y)

    %draw rectangles
    ranrec = rand(strokes,9);

    %     t = rand();
    %     blwi = [t,t,t];

    for i = 1:strokes
        %rectangle(ax, 'Position',[ranrec(i,1).*xmax ranrec(i,2).*ymax .2*ranrec(i,8).*(xmax-ranrec(i,1).*xmax) .2*ranrec(i,9).*(ymax-ranrec(i,2).*ymax)],'Curvature',ranrec(i,[3,4]),'FaceColor',ranrec(i,(5:7)),'EdgeColor',ranrec(i,(5:7))); %makes cool distorted statistics visible on canvas
        rectangle(ax, 'Position',[ranrec(i,1).*xmax ranrec(i,2).*ymax ranrec(i,8).*patchLim ranrec(i,9).*patchLim],'Curvature',ranrec(i,(3:4)),'FaceColor',ranrec(i,5:7),'EdgeColor',ranrec(i,5:7));
    end
    [cdata,~] = getframe(ax);
    im_pop(:,:,:,k) = double(cdata);
end

tic
for e = 1:generations
    for k = 1:population
        %fitness(k) = sum((tarim - im_pop(:,:,:,k)).^2,'all'); %/(tardim(1)*tardim(2))
        fitness(k) = sum(abs(tarim - im_pop(:,:,:,k)),'all');
    end

    %pick fittest individuals to become parents
    fitnessCopy = fitness;
    for r = 1:size(parents,4)
        [value,index] = min(fitnessCopy);
        %check if there was an improvement, if so make new parents
        if previous_minFitness < value
            parents(:,:,:,r) = im_pop(:,:,:,index);
        else
            parents(:,:,:,r) = im_pop(:,:,:,index);
            fitnessCopy(fitnessCopy==value) = max(fitnessCopy(:));
        end
    end

    %this will be used to only make new parents if one of the new idnividuals is fitter than the previous top individual
    [value,index] = min(fitness);
    previous_minFitness = value;
    previous_minIndex = index;

    %adaptive stuff

    %     patchLim = max(tardim)/4*exp(-0.0004*e);

    %     if e==50
    %         population = 25
    %         patchLim = 150
    %     elseif e==1000
    %         patchLim = 100
    %         population = 20
    %     elseif e==2000
    %         patchLim = 75
    %         population = 20
    %     elseif e==3000
    %         patchLim = 50
    %         population = 30
    %     elseif e==4000
    %         patchLim = 40
    %         population = 35
    %     elseif e==5000
    %         patchLim = 20
    %         population = 40
    %     elseif e==6000
    %         patchLim = 15
    %         population = 45
    %     elseif e==8000
    %         patchLim = 10
    %         population = 50
    %     elseif e==10000
    %         patchLim = 5
    %         population = 55
    %     end

    if e==50
        population = 20
        patchLim = 30
    elseif e==1000
        patchLim = 60
        population = 10
        %     elseif e==2000
        %         patchLim = 20
        %         population = 30
        %     elseif e==3000
        %         patchLim = 10
        %         population = 50
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

    %plot best parent
    resima.CData = uint8(im_pop(:,:,:,k));

    for k = 1:population
        %         xlo = randi(xmax);
        %         ylo = randi(ymax);
        %
        %         xhi = randi(xlo);
        %         yhi = randi(xlo);
        %
        %         for u = 1:3
        %             im_pop(yhi:ylo,xhi:xlo,u,k) = ones(ylo-yhi+1,xlo-xhi+1).*randi([0,255]);
        %         end

        %calima.CData = uint8(im_pop(:,:,:,k));

        %delet previous strokes
        X = findall(ax,'Type', 'Rectangle');
        delete(X)

        %draw rectangles
        ranrec = rand(strokes,9);

        %bring indivudal onto an axes
        calima.CData = uint8(im_pop(:,:,:,k));

        %         t = rand();
        %         blwi = [t,t,t];

        for i = 1:strokes
            %   rectangle(ax, 'Position',[ranrec(i,1).*xmax ranrec(i,2).*ymax .2*ranrec(i,8).*(xmax-ranrec(i,1).*xmax) .2*ranrec(i,9).*(ymax-ranrec(i,2).*ymax)],'Curvature',ranrec(i,[3,4]),'FaceColor',ranrec(i,(5:7)),'EdgeColor',ranrec(i,(5:7))); %makes cool distorted statistics visible on canvas
            rectangle(ax, 'Position',[ranrec(i,1).*xmax ranrec(i,2).*ymax ranrec(i,8).*patchLim ranrec(i,9).*patchLim],'Curvature',ranrec(i,(3:4)),'FaceColor',ranrec(i,5:7),'EdgeColor',ranrec(i,5:7));
        end

        %get the array with the rgb values from the axes in the figure
        [cdata,cmp] = getframe(fig1);
        im_pop(:,:,:,k) = double(cdata);
    end

    %capture first frames for movie
    if Record_Video == 1 && mod(e,Capture_Ratio) == 0
        Frame_Index = Frame_Index + 1;
        Frame_Collection(Frame_Index) = getframe(az);
    end
end

Render_Time_sec = toc
Render_Time_min = Render_Time_sec/60
Render_Time_h = Render_Time_min/60

%make mp4 movie
if Record_Video == 1
    makeMovie('LeonardosWork\PietMondrian_Rectangles.mp4',Frame_Collection,120)
end

