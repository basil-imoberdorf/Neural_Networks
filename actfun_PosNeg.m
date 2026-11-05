function output = actfun_PosNeg(raw_output)
%converts the raw output data from a neuron into a boolean type. Function returns 1 if the weighted sum is
%greater or equal to zero otherwise the function returns a zero
 output = (raw_output>0) + (raw_output<0) * -1;
end