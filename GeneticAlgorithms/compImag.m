function compImag = compImag(rawImag,compFactor,mode)
%Reduces the ammount of pixels by merging rgb values of multiple pixels to a new pixel
%compFactor determines the size of the pool. i.e compFactor = 4 --> the merging raster is now 4 by 4
%compFactor must be a integer
%mode defines the compression style. either take the max of all pixels or take the min or the average. Default is average
%valid modes are: 'avg'
%                 'max'
%                 'min'

switch nargin
    case 2
        mode = 'avg';
end

imdim = size(rawImag);
newHeight = floor(imdim(1)/compFactor);
newLength = floor(imdim(2)/compFactor);
compImag = zeros(newHeight,newLength,3);

switch mode
    case 'avg'
        for i = 0:newHeight-1
            for j = 0:newLength-1
                compImag(i+1,j+1,:) = mean(rawImag(1+i*compFactor:i*compFactor + compFactor,1+j*compFactor:j*compFactor + compFactor,:),[1,2]);
            end
        end
        compImag = uint8(compImag);

    case 'max'
        for i = 0:newHeight-1
            for j = 0:newLength-1
                compImag(i+1,j+1,:) = max(rawImag(1+i*compFactor:i*compFactor + compFactor,1+j*compFactor:j*compFactor + compFactor,:),[],[1,2]);
            end
        end
        compImag = uint8(compImag);

    case 'min'
        for i = 0:newHeight-1
            for j = 0:newLength-1
                compImag(i+1,j+1,:) = min(rawImag(1+i*compFactor:i*compFactor + compFactor,1+j*compFactor:j*compFactor + compFactor,:),[],[1,2]);
            end
        end
        compImag = uint8(compImag);
end