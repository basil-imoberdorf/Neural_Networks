close all
testGen = 100;

ima=uint8(randi([0,255],900,600,3));
fig=figure;
ax=axes(fig);
imshow(ima,'Border','tight','Parent',ax)
storage1 = uint8(zeros(900,600,3,testGen));
storage2 = uint8(zeros(900,600,3,testGen));
Frame_Collection = struct('cdata', cell(1,testGen), 'colormap', cell(1,testGen));

tic
for i=1:testGen
    [a,b] = getframe(ax);
    storage1(:,:,:,i) = a;
    Frame_Collection(i) = getframe(ax);
    imshow(uint8(randi([0,255],900,600,3)),'Parent',ax)
end
test_Suite1 = toc



% %write the movie
% Movie = VideoWriter('test1','MPEG-4');
% Movie.FrameRate = 10;
% Movie.Quality = 100;
% 
% open(Movie)
% writeVideo(Movie,Frame_Collection)
% close(Movie)

Frame_Collection = struct('cdata', cell(1,testGen), 'colormap', cell(1,testGen));

tic
for i=1:testGen
     %t=ax.Children.CData;
     %storage1(:,:,:,i) = t;  
     Frame_Collection(i).cdata = ax.Children.CData;
     imshow(uint8(randi([0,255],900,600,3)),'Parent',ax)
end
test_Suite2 = toc
% %write the movie
% Movie = VideoWriter('test2','Archival');
% Movie.FrameRate = 10;
% %Movie.Quality = 100;
% 
% open(Movie)
% writeVideo(Movie,Frame_Collection)
% close(Movie)



%     isequal(a,ax.Children.CData)
%     sum(ax.Children.CData==a,'all')/(size(ima,1)*size(ima,2)*size(ima,3))



