function makeMovie(Filename,Frame_Collection,Frame_Rate)
%create a .mp4 video clip based on the frames contained in the variable "Frame_Collection"

%Filename: input desired filename as string with ending .mp4
%Frame_Collection: Array containing object. Normaly generated in a loop via getframe: Frame_Collection(loop_index) = getframe(gcf)
%FrameRate: optional parameter that defines how many frames are displayed per second. Default is 60 Fps

switch nargin
    case 2
        Frame_Rate = 60;
end

%movie properties
Movie = VideoWriter(Filename,'MPEG-4');
Movie.FrameRate = Frame_Rate;
Movie.Quality = 100;

%write the movie
open(Movie)
writeVideo(Movie,Frame_Collection)
close(Movie)
end