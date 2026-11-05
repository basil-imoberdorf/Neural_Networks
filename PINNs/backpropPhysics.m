function [W_grad_physics, b_grad_physics] = backpropPhysics(x_physics, t_physics, W, z, a, s, r, Jf, lambda)

% Helper variables
n_collocation = length(t_physics);
n_layers = numel(W);

% Initialize gradient cells for the contribution from the physics residual
W_grad_physics = cell(n_layers,1);
b_grad_physics = cell(n_layers,1);

% Seed adjoints
delta_a = zeros(size(r));                % dLdx = drdx'*dLdr' = dLdz3' % Transposed because we stay in numerator layout (gradients are then row vectors so we need to transpose to get column vectors. And we want column vectors because they are compatible with the single observation input/outputs being column vectors)
for k = 1:n_collocation
    Jk = Jf(x_physics(:,k), lambda);     % drdx
    delta_a(:,k) = -Jk' * r(:,k);
end


delta_a = delta_a ./n_collocation;      % Divide by number of collocation points to get the correct MSE derivative
delta_s = r./n_collocation;              % dLd(dxdt) = dLdr * drd(dxdt) where dLdr = r./n_collocation 

% Backward loop 
for i = n_layers:-1:1
    % Activation derivatives for this layer
    if i == n_layers  % Output layer: identity
        sig1 = ones(size(z{i}));   % this is necessary if the activation function is not applied on the output layer
        sig2 = zeros(size(z{i}));  % this is necessary if the activation function is not applied on the output layer
    else
        sig1 = tanh_d(z{i});  % first derivative of activation function
        sig2 = tanh_dd(z{i}); % second derivative of activation function
    end
    
    % dz/dt for this layer (needed for sig2 term)
    if i == 1
        dzdt = W{i} * ones(1, n_collocation);          % s0 = 1 = dtdt
    else
        dzdt = W{i} * s{i-1};
    end
    
    % Combined pre‑activation adjoint
    g_z = delta_a .* sig1 + delta_s .* (sig2 .* dzdt);  % dadt
    g_dz = delta_s .* sig1;
    
    % Gradients for parameters at layer i 
    if i == 1
        W_grad_physics{i} = g_z * t_physics' ;    % a0 = t
    else
        W_grad_physics{i} = g_z * a{i-1}';
    end
    % Add the contribution from the direct dependence of s[i] on W[i]
    if i == 1
        W_grad_physics{i} = W_grad_physics{i} + g_dz * ones(n_collocation,1);  % s0 = 1 or ones(length(t),1) for the vectorized input. Basically just sum up the individual columns
    else
        W_grad_physics{i} = W_grad_physics{i} + g_dz * s{i-1}';
    end
    b_grad_physics{i} = sum(g_z, 2);  % this is the same as g_z * ones(n_collocation,1) but apparently faster (not tested)

    % Propagate adjoints to previous layer (if any)
    if i > 1
        delta_a = W{i}' * g_z;
        delta_s = W{i}' * (sig1 .* delta_s);
    end
end
end
