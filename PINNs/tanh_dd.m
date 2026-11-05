function y = tanh_dd(x)
% Second derivative of tanh()
y = -2.*tanh(x).*(1 - tanh(x).^2);
end