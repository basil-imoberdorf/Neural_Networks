function y = ReLU(x,leak_size)
% Rectified linear unit activation function
%  This Function delivers 0 for negative values. Zero and all positive values go trough as they come in
%  x is the input: Can be scalar or a vector or a matrice
% the  optional value of leak_size defines the slope by which the value is scaled if the value is negative. i. e. ReLU(-3,0.2) returns 0.2*(-3)


switch nargin
    case 1
        leak_size = 0.2;
end

%actual ReLU
y = double((x>=0) .* x) + double((x<0) .* x .*leak_size);
end