%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x, dxdt, z, a, da] = feedForwardPINN(W, b, t)
% W: cell array where W{i} holds the matrix for the linear map from activation a(i-1) to pre-activation, z(i)
% b: cell array where b{i} holds the vector for the biases used in layer i

% Helper variables used in backprop
n_layers = numel(W);
z = cell(n_layers, 1);    % Pre-activations aka weighted sum 
a = cell(n_layers-1, 1);  % Activations 
dz = cell(n_layers-1, 1); % holds dzdt for each layer except output where there is not activation function
da = cell(n_layers-1, 1); % holds dadt for each layer except output where there is not activation function

%%% Forward-Step to compute the state prediciton x and its derivative dxdt
for i = 1:n_layers
    % Helper varaibles to reduce indexing
    W_i = W{i};
    b_i = b{i};

    % Compute pre-activation and derivative wrt inputs 
    if i==1
        z{i} = W_i * t  +  b_i;      % a[0] = t 
        dz{i} = W_i;                 % dz[1]dt = dz[1]da[0]*da[0]dt  since a[0] = t  --> da[0]dt = dt/dt = 1
    else
        z{i} = W_i * a{i-1}  +  b_i;   % State prediciton x forward step 
        dz{i} = W_i * da{i-1};           % dz[i]dt = dz[i]da[i-1]*da[i-1]dt
    end
    
    % Helper variables to reduce indexing
    z_i = z{i};
    dz_i = dz{i};
    
    % Propagate time derivative: tanh_d(z_i) = da[i]dz[i] So overall this becomes da[i]dt
    da{i} = tanh_d(z_i) .* dz_i;       % Actualy its diag(tanh_d(z{i})) * dzdt{i} but since there are no off diagonal elements the element wise multiplication is valid
   
    if i==n_layers % Last layer no activation since we do regression, not classification
        x = z_i;      % State prediction output
        dxdt = dz_i;  % State time derivative output
    else
        a{i} = tanh(z_i); % Activation
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


