clear
close all

% Define smallest 3 layer network that exhibits full complexity e.g. no backprop shortcuts possible
num_states = 2;
num_neurons_layer = 2;

W1 = randn(num_neurons_layer, 1);
W2 = randn(num_neurons_layer);
W3 = randn(num_states, num_neurons_layer);

b1 = randn(num_neurons_layer, 1);
b2 = randn(num_neurons_layer, 1);
b3 = randn(num_states, 1);

W = {W1,W2,W3};
b = {b1,b2,b3};


%%% Define the vectorized input: 
n_data = 20;
n_collocation = 200;  % number of collocation points to train the physics aka to inform the neural net about the physics
t_data = linspace(0,6,n_data);                % Data points
t_physics = linspace(0,12,n_collocation);     % Collocation points to inform the NN about the ODE aka the physics
t = [t_data, t_physics];                      % Concatenated input to feed them all simultaneously through the newtork

% Create synthetic data of a system
lambda1 = 2; % acceleration term
lambda2 = 0.3; % damping term
lambda  = [lambda1, lambda2];

x0 = [1;1]; % Initial conditions
odefun = @(t,x) myODE(t,x,lambda); % ODE
[~,x_synth] = ode45(odefun, t_data, x0); % solve the ODE
[~,x_truth] = ode45(odefun, t_physics, x0); % solve the ODE

% True labels for supervised learning
x_noisy = x_synth + 0.08.*randn(size(x_synth));
x_meas_data = x_noisy';

% hold on
% scatter(t_data, x_noisy(:,1), 'LineWidth', 1.5, 'DisplayName', 'x_1 - noisy meas.')
% scatter(t_data, x_noisy(:,2), 'LineWidth', 1.5, 'DisplayName', 'x_2 - noisy meas.')
% plot(t_physics, x_truth(:,1), 'DisplayName', 'x_1 - groundtruth', 'Marker', 'x')
% plot(t_physics, x_truth(:,2), 'DisplayName', 'x_2 - groundtruth', 'Marker', 'x')
% legend()
% xlabel('Time  [h]')
% ylabel('Outputs')
% grid on





%%%%%%%%%%%%%%%%% Forward Pass %%%%%%%%%%%%%%%%%
% Compute the forward pass for the data and the physics collocation points seperately to avoid having to merge the input vector and unpack/separate the outputs
[x_data, dxdt_data, z_data, a_data, s_data] = forward_pass(W, b, t_data);
[x_physics, dxdt_physics, z_physics, a_physics, s_physics] = forward_pass(W, b, t_physics);

%%%%%%%%%%%%%%%%% Compute Loss %%%%%%%%%%%%%%%%%
[L_physics, r] = loss_physics_MSE(x_physics, dxdt_physics, lambda);  % Penalize deviations from the ODE by minimizing the residual: dxdt - f(x,t,lambda), where f is the RHS 
L_data = loss_data_MSE(x_data, x_meas_data);  % Penalize deviation from observed data
L = L_physics + L_data;  % Unweighted total loss

%%%%%%%%%%%%%%%%% Backward Pass %%%%%%%%%%%%%%%%%
%%% Backprop through physics loss
% First compute the Jacobian of the ODE RHS wrt states. They are needed for the seeding the adjoint. For example for the x branch: dLdx = drdx'*dLdr' = dLdz[L]' = -Jf' * r
Jf = [0, 1; ...
     -lambda1, -lambda2];  % Since for this system the Jacobian is always constant we can leave it like this
[W_grad_physics, b_grad_physics] = backprop_physics(t_physics, W, z_physics, a_physics, s_physics, r, Jf);

%%% Backprop through data loss
[W_grad_data, b_grad_data] = backprop_data(x_data, x_meas_data, t_data, W, z_data, a_data);

% Sum up the individual contributions from each summand in the loss function
W_grad = cell(3,1);
b_grad = cell(3,1);
for i = 1:3
    W_grad{i} = W_grad_data{i} + W_grad_physics{i};
    b_grad{i} = b_grad_data{i} + b_grad_physics{i};
end


%%%%%%%%%%%%%%%%%%%%%% Check weight gradients %%%%%%%%%%%%%%%%%%%%%%
% --- numerical gradients by central finite differences ---
eps = 1e-6;
W_grad_centralDiff = {zeros(size(W1)),zeros(size(W2)),zeros(size(W3))};
W_copy = W;

for i = 1:numel(W_grad_centralDiff)
    W_curr = W{i};
    for j = 1:numel(W_curr)
        W_pos = W_curr; 
        W_neg = W_curr; 
        
        W_pos(j) = W_curr(j) + eps; 
        W_neg(j) = W_curr(j) - eps; 
        
        % Positiove perturbation
        W_copy{i} = W_pos;
        [x_data, ~] = forward_pass(W_copy, b, t_data);
        [x_physics, dxdt_physics] = forward_pass(W_copy, b, t_physics);
        L_physics_pos = loss_physics_MSE(x_physics, dxdt_physics, lambda);
        L_data_pos = loss_data_MSE(x_data, x_meas_data);
        L_pos = L_physics_pos + L_data_pos;
        
        % Physics loss
        W_copy{i} = W_neg;
        [x_data, dxdt_data] = forward_pass(W_copy, b, t_data);
        [x_physics, dxdt_physics] = forward_pass(W_copy, b, t_physics);
        L_physics_neg = loss_physics_MSE(x_physics, dxdt_physics, lambda);
        L_data_neg = loss_data_MSE(x_data, x_meas_data);
        L_neg = L_physics_neg + L_data_neg;

        W_grad_centralDiff{i}(j) = (L_pos - L_neg)./(2*eps);
    end
    % Reset W_copy
    W_copy = W;
end

%% Compare gradients
for i = 1:numel(W_grad_centralDiff)
    % Get difference
    grad_diff = W_grad_centralDiff{i} - W_grad{i}

    tresh = 1e-8;
    if norm(grad_diff) < tresh
        fprintf('\n✅ PASS ✅: Analytical gradients match central differences down to %.0e \n', tresh);
        fprintf('Backrpop correctly implemented. Norm of gradient difference at layer %d:  -->  %.2e \n', i, norm(grad_diff));
    else
        fprintf('\n❌ FAIL ❌: Analytical gradients differ significantly from numeric approximations!\n');
        fprintf('Check backprop and/or code for numeric differentiation.\n');
    end
end
%%%%%%%%%%%%%%%%%




%% Loss functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [L_physics, r]  = loss_physics_MSE(x_physics, dxdt_physics, lambda)
% Penalize deviations from the ODE by minimizing the residual dxdt - f(x,t,lambda) where f is the RHS
n_physics = size(x_physics, 2);
r = dxdt_physics - [x_physics(2,:); -lambda(1).*x_physics(1,:) - lambda(2).*x_physics(2,:)]; % lets say the ODE is dxdt = lambda*x which leads to the goal of this whole undertaking==> 0 = dxdt - lambda*x
L_physics = 0.5.*(r(:)'*r(:))./n_physics;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function L_data = loss_data_MSE(x_data, x_trueNoisyData)
% Penalize deviation from observed data
n_data = size(x_data, 2);
L_data = 0.5*sum((x_data(:) - x_trueNoisyData(:)).^2)./n_data; % same as 0.5/n_data * (x_data(:) - x_trueNoisyData(:))'*(x_data(:) - x_trueNoisyData(:))
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Backprop for the physics loss term
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [W_grad_physics, b_grad_physics] = backprop_physics(t_physics, W, z, a, s, r, Jf)

n_collocation = length(t_physics);

% Initialize cells for the contribution to the final gradient from the physics residual
W_grad_physics = cell(3,1);
b_grad_physics = cell(3,1);

% Seed adjoints: Basically the chain starting from the Loss down to the network output.
delta_a = cell(3, 1);
delta_s = cell(3, 1);
delta_a{3} = (-Jf' * r)./n_collocation;     % dLdx = drdx'*dLdr' = dLdz3' % Transposed because we stay in numerator layout (gradients are then row vectors so we need to transpose to get column vectors. And we want column vectors because they are compatible with the single observation input/outputs being column vectors) 
delta_s{3} = r./n_collocation;              % dLddxdt = drdx * drdx

% Backward loop throug physics residual
for i = 3:-1:1
    % Activation derivatives for this layer
    if i == 3
        % Output layer: identity
        sig1 = ones(size(z{i}));   % this is necessary if the activation function is not applied on the output layer
        sig2 = zeros(size(z{i}));  % this is necessary if the activation function is not applied on the output layer
    else
        sig1 = tanh_d(z{i});
        sig2 = tanh_dd(z{i});
    end
    
    % dz/dt for layer i (needed for sig2 term)
    if i == 1
        dzdt = W{i} * ones(1, n_collocation);          % s0 = 1 = dtdt
    else
        dzdt = W{i} * s{i-1};
    end
    
    % Combined pre‑activation adjoint
    g_z = delta_a{i} .* sig1 + delta_s{i} .* (sig2 .* dzdt);  % dadt
    g_dz = delta_s{i} .* sig1;
    
    % Gradients
    if i == 1
        W_grad_physics{i} = g_z * t_physics' ;    % a0 = t (scalar)
    else
        W_grad_physics{i} = g_z * a{i-1}';
    end
    % Add the second term (from the direct dependence of s^i on W^i) Note: could be done in one line but this script is for understanding the basics of PINN backprop
    if i == 1
        W_grad_physics{i} = W_grad_physics{i} + g_dz * ones(n_collocation,1);  % s0 = 1 or ones(length(t),1) for the vectorized input. Basically just sum up the individual columns
    else
        W_grad_physics{i} = W_grad_physics{i} + g_dz * s{i-1}';
    end
    % And for the weights its simply g_z (or its sum for vectorized inputs)
    b_grad_physics{i} = g_z * ones(n_collocation,1);  % same as sum(g_z,2)
    
    % Propagate to previous layer (if any)
    if i > 1
        delta_a{i-1} = W{i}' * g_z;
        delta_s{i-1} = W{i}' * (sig1 .* delta_s{i});
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Backprop for the data loss term --> this is classical NN backprop for a MSE loss
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [W_grad_data, b_grad_data] = backprop_data(x_data, y_meas_data, t_data, W, z, a)
% get number of data points
n_data = length(t_data);

% Initialize gradient cells
W_grad_data = cell(3,1);
b_grad_data = cell(3,1);

% Seed adjoint
delta_data = cell(3, 1);
delta_data{3} = (x_data - y_meas_data)./n_data;

for i = 3:-1:1
   
    % Parameter gradients
    if i == 1
        W_grad_data{i} = delta_data{i} * t_data' ;    % a0 = t (scalar)
    else
        W_grad_data{i} = delta_data{i} * a{i-1}';
    end
    b_grad_data{i} = delta_data{i} * ones(n_data,1);
    
    % Propagat data adjoint back trough layers (if any)
    if i > 1
        delta_data{i-1} = (W{i}' * delta_data{i}) .* tanh_d(z{i-1});  % dLdz[i-1] = dz[i]da[i-1]^T * dLdz[i]^T  .* da[i-1]dz[i-1]  Becasue thisis all in Nominator layout --> dLdz[i-1] = dLdz[i] * dz[i]da[i-1] * da[i-1]dz[i-1]
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x, dxdt, z, a, s] = forward_pass(W, b, t)
% W: cell array where W{i} hold the matrix for the linear map from activation, a(i-1) to pre-activation, z(i)
% b: cell array where b{i} hold the vector for the biases

%%% Forward-Step to compute the state prediciton x and its derivative dxdt
for i = 1:3
    % Pre-activation
    if i==1
        z{i} = W{i} * t  +  b{i};       % a{0} = t 
        dzdt{i} = W{i}*1;               % dz[1]dt = dz[1]da[0]*da[0]dt  since a[0] = t  --> da[0]dt = dt/dt = 1
    else
        z{i} = W{i} * a{i-1}  +  b{i};  % State prediciton x forward step 
        dzdt{i} = W{i}*dadt{i-1};       % dz[i]dt = dz[i]da[i-1]*da[i-1]dt
    end

    dadt{i} = tanh_d(z{i}) .* dzdt{i};  % Actualy its diag(tanh_d(z{i})) * dzdt{i} but since dzdt{i} is always a vector the element wise multiplication is valid
   
    if i==3 % Last layer no activation
        x = z{i};
        dxdt = dzdt{i};
    else
        a{i} = tanh(z{i}); % Activation
        s{i} = dadt{i};    % time derivative
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dxdt = myODE(~,x,theta)

dxdt = zeros(length(x),1);
dxdt(1) = x(2);
dxdt(2) = -theta(1)*x(1) -theta(2)*x(2); % acceleration: -theta(1)*x(1). Damping:  -theta(2)*x(2)

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% First derivative of tanh()
function y = tanh_d(x)
    y = 1 - tanh(x).^2;
end
% Second derivative of tanh()
function y = tanh_dd(x)
    y = -2.*tanh(x).*(1 - tanh(x).^2);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

