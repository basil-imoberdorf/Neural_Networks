function [W_grad_data, b_grad_data] = backpropData(x_data_pred, x_data_true, t_data, W, z, a)
% Computes the gradients w.r.t weights given the state prediction, x_data_pred
% Needs the actual measured x_data_true as well as the corresponding time points, t_data. 
% The function expects furthermore the weights, W, all pre-activations, z, and all activations, a.

% Helper variables
n_data = size(x_data_pred, 2);
n_layers = length(W);

% Pre-allocate gradient cells
W_grad_data = cell(n_layers,1);
b_grad_data = cell(n_layers,1);

% Seeding the adjoint with dLdz[L] = dLdr*drdx since x = z[L] 
delta_data = (x_data_pred - x_data_true)./n_data;

for i = n_layers:-1:1
    
    % Gradients for parameters at layer i 
    if i == 1
        W_grad_data{i} = delta_data * t_data' ;    % a0 = t 
    else
        W_grad_data{i} = delta_data * a{i-1}';
    end
    b_grad_data{i} = sum(delta_data, 2);  % thius is the same as delta_data * ones(n_data,1) but apparently faster (not tested)
    
    % Propagate data adjoint back trough layers (if any)
    if i > 1
        delta_data = (W{i}' * delta_data) .* tanh_d(z{i-1});  % dLdz[i-1] = dz[i]da[i-1]^T * dLdz[i]^T  .* da[i-1]dz[i-1]  Becasue thisis all in Nominator layout --> dLdz[i-1] = dLdz[i] * dz[i]da[i-1] * da[i-1]dz[i-1]
    end
end
end


