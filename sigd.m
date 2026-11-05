function y = sigd(x)
%first derivative of sigmoid function
y = 1./(1+exp(-x)) .*  (1  -  (1./(1+exp(-x))));
end