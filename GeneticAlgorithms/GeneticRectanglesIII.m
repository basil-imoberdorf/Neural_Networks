clear
close all

fig2 = figure; fig3 = figure;
fig2.Position = [1400,300,500,500];
fig3.Position = [900,300,500,500];
ay = axes(fig2); az = axes(fig3);

%import target image
tarim = imread('Pic\MonaLisa.jpg');  % public-domain target, similar size to the former image
tarim = compImag(tarim,2,'avg');
imshow(tarim,'Border','tight','Parent',ay)

tardim = size(tarim);
%fig1.Position = [400,300,tardim(2),tardim(1)];
%ax = axes(fig1,"XTick",[],"YTick",[],"Box",'on','Position',[0 0 1 1],'DataAspectRatio',[1,1,1],'XLim',[0 tardim(2)],'YLim',[0 tardim(1)],'Color',[.5 .5 .5],'YColor',[.5 .5 .5],'XColor',[.5 .5 .5]);

%paramters
population = 30;
strokes = 1;
generations = 20000;
Record_Video = 0;   %creates a MPEG-4 video of the simulation if set to 1
Capture_Ratio = 5; %after how many frames one gets saved for video
xmax = tardim(2);
ymax = tardim(1);
fitness = zeros(population,1);
im_pop = uint8(ones(tardim(1),tardim(2),3,population).*128);
parents = uint8(ones(tardim(1),tardim(2),3,2).*128); %preallocate two parents
previous_minFitness = 10e10;
patchLim = (tardim(2)/3)*exp(-(5/generations)*0) + 10;
%calima = imshow(uint8(im_pop(:,:,:,1)),'Parent',ax);
resima = imshow(uint8(parents(:,:,:,1)),'Border','tight','Parent',az);
Frame_Collection = struct('cdata', cell(1,generations/Capture_Ratio), 'colormap', cell(1,generations/Capture_Ratio)); %preallocate the array for the images
Frame_Index = 1;
patch_scaler = ones(population,1);

%capture first frames for movie
if Record_Video == 1
    Frame_Collection(1).cdata = az.Children.CData;
end

% %generate first population
% for k = 1:population
% 
%     %random number container for rectangle creation
%     ranrec = rand(strokes,4);
% 
%     %hand made rectangles
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %hand made rectangles
%     xLow = ceil(ranrec(1,1)*xmax);
%     if (xLow + patchLim) > tardim(2)
%         xHigh = xmax;
%     else
%         xHigh = floor(xLow + patchLim*rand);
%     end
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     yLow = ceil(ranrec(1,2)*ymax);
%     if (yLow + patchLim) > tardim(1)
%         yHigh = ymax;
%     else
%         yHigh = floor(yLow + patchLim*rand);
%     end
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%     %draw each color of the rectangle in the loop
%     for i=1:3
%         im_pop(yLow:yHigh,xLow:xHigh,i,k) = ones(1+yHigh-yLow,1+xHigh-xLow).*randi([0,255]);
%     end
% 
% %      calima.CData = im_pop(:,:,:,k);
% %      pause(0)
% 
% end

tic
for e = 1:generations
    for k = 1:population
        %fitness(k) = sum((tarim - double(im_pop(:,:,:,k))).^2,'all')/(tardim(1)*tardim(2));
        total_error = sum((tarim - im_pop(:,:,:,k) + (im_pop(:,:,:,k) - tarim)),'all');
        fitness(k) = total_error./patch_scaler(k);  % account for the fact that small pathes produce lower errors and are then often picked even if the color match is very bad
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
    if e==50
        population = 35
    elseif e==1000
        population = 35
    elseif e==2000
        population = 35
    elseif e==3000
        population = 35
    elseif e==4000
        population = 40
    elseif e==5000
        population = 45
    elseif e==6000
        population = 50
    elseif e==7000
        population = 60
    elseif e==8000
        population = 65
    elseif e==9000
        population = 70
    elseif e==10000
        population = 75
    elseif e==11000
        population = 80
    elseif e==16000
        population = 85
    elseif e==19000
        population = 100
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
    resima.CData = parents(:,:,:,1);
    %pause(0)

    for k = 1:population

        %draw rectangles
        ranrec = rand(strokes,4);
        patchLim1 = (tardim(2)/3)*exp(-(15/generations)*e) + 7;
        patchLim2 = (tardim(1)/3)*exp(-(15/generations)*e) + 7;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %hand made rectangles
        xLow = ceil(ranrec(1,1)*xmax);
        if (xLow + patchLim1) > tardim(2)
            xHigh = xmax;
        else
            xHigh = floor(xLow + patchLim1*ranrec(1,2));
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        yLow = ceil(ranrec(1,3)*ymax);
        if (yLow + patchLim2) > tardim(1)
            yHigh = ymax;
        else
            yHigh = floor(yLow + patchLim2*ranrec(1,4));
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        Ones_Template = ones(1+yHigh-yLow,1+xHigh-xLow);
        patch_scaler(k) = numel(Ones_Template);
        for i=1:3
            im_pop(yLow:yHigh,xLow:xHigh,i,k) = Ones_Template.*randi([0,255]);
        end

       
        if mod(e,10)==1
            calima.CData = im_pop(:,:,:,k);
            pause(0)
        end
    end

    %capture frames for movie
    if Record_Video == 1 && mod(e,Capture_Ratio) == 0
        Frame_Index = Frame_Index + 1;
        Frame_Collection(Frame_Index).cdata = az.Children.CData;
    end
end

Render_Time_sec = toc
Render_Time_min = Render_Time_sec/60
Render_Time_h = Render_Time_min/60

%make mp4 movie
if Record_Video == 1
    makeMovie('LeonardosWork\MonaLisa_RectanglesIII.mp4',Frame_Collection,120)
end


% %make lossless movie
% Movie = VideoWriter('LeonardosWork\EldenHD','Archival');
% Movie.FrameRate = 150;
% 
% %write the movie
% open(Movie)
% writeVideo(Movie,Frame_Collection)
% close(Movie)