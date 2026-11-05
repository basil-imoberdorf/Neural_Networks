function y = ReLUd(x,leak_size)
%First derivative of the ReLU activation function
%Very simple outputs. 0 if argumetn is negative and 1 if argument is zero or positive
%For more information visit help of ReLU

switch nargin
    case 1
        leak_size = 0.2;
end

y = double(x>=0) + double((x<0) .* leak_size);
end