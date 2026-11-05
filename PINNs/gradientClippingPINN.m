function [W_grad_data, b_grad_data, W_grad_physics, b_grad_physics] = gradientClippingPINN(W_grad_data, b_grad_data, W_grad_physics, b_grad_physics, tresh)
% Hardclip the gradients to a threshold
clip = @(x) min(max(x, -tresh), tresh);

W_grad_data     = cellfun(clip, W_grad_data,     'UniformOutput', false);
b_grad_data     = cellfun(clip, b_grad_data,     'UniformOutput', false);
W_grad_physics  = cellfun(clip, W_grad_physics,  'UniformOutput', false);
b_grad_physics  = cellfun(clip, b_grad_physics,  'UniformOutput', false);

end