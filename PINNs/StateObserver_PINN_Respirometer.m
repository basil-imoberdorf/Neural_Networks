clear
close all
rng(42)

%% Define the physics to be learned by the PINN
% State variables
syms X S DOT V real
% Model parameters
syms mu_max Y_xs Y_xo K_s K_o kLa S_in F real
DOT_sat = 100;
% Helper variables
mu = mu_max * S/(S + K_s) * DOT/(DOT + K_s);
% Define RHS: Damped harmonic oscillator
f1 = mu*X - F/V*X;
f2 = F/V*(S_in - S) - mu/Y_xs*X;
f3 = kLa*(DOT_sat - DOT) - mu/Y_xo*X;
f4 = F;

% Vecotrize
x = [X; S; DOT; V];
p = [mu_max; Y_xs; Y_xo; K_s; K_o; kLa; S_in; F];
f = [f1;f2;f3;f4];
% Compute dfdx, the Jacobian needed for seeding the adjoint for backprop
dfdx = jacobian(f,x);
% Create numeric function handles
f_rhs = matlabFunction(f, 'Vars', {x, p}); % Right-hand-side of ODE
Jf = matlabFunction(dfdx, 'Vars', {x, p});

%% Define the vector input for the PINN. The PINN can process multiple inputs at once if the input columns (or in this case only the scalar t is extended, added columns)
n_data    = 8;  % Number of measurements
n_physics = 1000; % Number of collocation points to train on the physics aka to inform the neural network about the physics given by the ODE
% Set input variables for the PINN
start_t = 1;
end_t = 3;
t_data    = linspace(start_t, end_t, n_data);  % Data time points
t_physics = linspace(start_t, end_t, n_physics);  % Physics collocation time points
t_test    = linspace(start_t-2, end_t+2 , 1000);  % To test the prediction of the PINN

t_data_std    = 1; %std(t_data   );
t_physics_std = 1; %std(t_physics);
t_test_std    = 1; %std(t_test   );

% t_data    = t_data   ./ std(t_data   );
% t_physics = t_physics./ std(t_physics);
% t_test    = t_test   ./ std(t_test   );

t_physics_ori = t_physics(1);

% t_physics = [t_physics;sin(t_physics);cos(t_physics)];
% t_data = [t_data;sin(t_data);cos(t_data)];
% t_test = [t_test;sin(t_test);cos(t_test)];


% Set model parameters
mu_max = 0.74;
Y_xs = 0.5;
Y_xo = 0.01;
K_s = 0.015;
K_o = 5;
kLa = 14;
S_in = 100;
F = 0.02;
theta = [mu_max; Y_xs; Y_xo; K_s; K_o; kLa; S_in; F]; % Enter model parameters as COLUMN vector please!!!

% Initial conditions
x0 = [1; 0; DOT_sat/3; 2]; % [X S DOT V]

% Create synthetic training data
odefun = @(t,x) f_rhs(x, theta); % ODE
ode_opts = odeset('RelTol', 1e-13, 'AbsTol', 1e-13); % ensure high quality ground truth
[~, x_synth] = ode45(odefun, t_data(1,:), x0, ode_opts); % solve the ODE to get syn dada
[~, x_physics_true] = ode45(odefun, t_physics(1,:), x0, ode_opts);  % get groundtruth
% Add measurement noise artificially
x_meas = x_synth' + 0.0004.*randn(size(x_synth'));  % true labels for the PINN. Transposed to have observations in the columns. So each row contains the state at a given t_data element
obs_states_idx = [1:4];

%% Define architecture of dense feed forward net by stating input and output dimensions as well as the number of neurons per layer
input_dim = 1;  % for PINNs on ODEs the only input is the scalar time
output_dim = size(x_meas,1);  % one output per state variable
dense_NeuralNet_design = [input_dim, 92, 92, output_dim];  % [Input, Hidden_Layer_1, Hidden_Layer_2, ... , Hidden_Layer_n, Output]

% Training hyperparameters
max_epochs = 40000;
learn_rate = 0.001;
emv_momentum = 0.9; % define what fraction of the previous gradient 2should be used for the current gradient. Only the remaining part, 1-momentum_share, is contributed by the current gradient. Is the beta term here: m_i+1 = beta*m_i + (1-beta)*gradients
emv_sqgrad = 0.999; % Exponential Moving Average coefficient. Is the beta term here: v_i+1 = beta*v_i + (1-beta)*gradients.^2
alpha_data = 0.2; % loss function weight for the data loss
alpha_physics = (1-alpha_data);  % loss function weight for the physics loss

num_hidden_layers = length(dense_NeuralNet_design) - 2;  % -2 to remove input and output layer from consideration
num_forward_steps = num_hidden_layers + 1;  % we need to perform n-1 matrix multiplications + activation function to do one forward pass where n is the number of hidden layers + input and output layer, so n is total layers

% Initialize containers for weights and gradients 
[weights_cell, biases_cell] = initialize_weights_and_biases(dense_NeuralNet_design, 'xavier', 0);  % 
[weights_grad, biases_grad] = initialize_weights_and_biases(dense_NeuralNet_design, 'zeros', 0); % create cell arrays populated with matrices of zeros
gradient_norm = zeros(2, num_forward_steps);  % to store the gradient norm at each layer and for both the data and physics gradietns seperately

% Pre-allocation of helper variables used for the Adam optimizer
m_momentum_weights     = weights_grad;  % Initialize the cells for the Adam optimizer
v_squared_grad_weights = weights_grad;  % Initialize the cells for the Adam optimizer
m_momentum_bias        = biases_grad;   % Initialize the cells for the Adam optimizer
v_squared_grad_bias    = biases_grad;   % Initialize the cells for the Adam optimizer

% Pre-allocate loss arrays
L_data = zeros(max_epochs,1);
L_physics = zeros(max_epochs,1);
L_tot = zeros(max_epochs,1);

% Set-up diagnostic plots
[pred_plt, phys_scat] = setup_output_plot(x_meas, x_physics_true, t_data, t_physics, t_test, obs_states_idx);
pause(1)
L_plt = setup_loss_evolution_canvas();
pause(1)
grad_plt = setup_gradient_norm_plot(num_forward_steps);
pause(1)
% [hist_ax1, hist_ax2] = setup_gradient_histograms();
% pause(1)
% [neuro_handle, line_cell] = setup_NN_canvas(dense_NeuralNet_design);
% pause(2)
% res = 40;
% alpha = linspace(-5, 5, res);
% beta = linspace(-5, 5, res);
% [ax, A, B] = setupLossLandscape2D(weights_cell, biases_cell, t_data, t_physics, x_meas, f_rhs, lambda, alpha_data, alpha_physics, res, alpha, beta);
% pause(1)

plot_interval = 100; % only every plot_interval'th epoch the plots are updated
x_meas = x_meas(obs_states_idx,:); % chose only a subset to be observable, the remaining states must be reconstructed by the PINN


%%% CAUTION: Setup adaptive phyiscs collocation point refinement
Alpha = 4;
t_physics_core = t_physics(2*(1:n_physics/2));
t_physics_core(1) = 0; % ensure that the initial condition is evaluated at the correct position, eg the solver starts at the initial conditon
t_physics_vari = t_physics(2*(1:n_physics/2) - 1);
weights_grad_cell_plot = cell(num_forward_steps, 1);

%% Train the network
tic
for e = 1:max_epochs
    
    % %% CAUTION: Adaptive phyisc collocation point refinement
    % if mod(e, 2000) == 0
    %     % t_candidates = 22*rand(1, n_physics*100);
    %     t_candidates = linspace(0, 12, n_physics*10);
    %     [x_physics, dxdt_physics] = feedForwardPINN(weights_cell, biases_cell, t_candidates);
    %     [~, r_candidate] = lossPhysics(x_physics, dxdt_physics, f_rhs, theta);
    %     % Create and sample from a distribution of the residuals
    %     weights = (vecnorm(r_candidate) + 1e-12).^Alpha;
    %     p = weights / sum(weights);
    %     idx = randsample(length(t_candidates), n_physics - numel(t_physics_core), true, p);
    %     t_physics_vari = t_candidates(idx);
    %     % Recombine and sort with unchangable grid
    %     t_physics = sort([t_physics_core,t_physics_vari]);       
    % end

    %%% Forward pass
    % Compute the forward pass for the data and the physics collocation points seperately to avoid having to merge the input vector and unpack/separate the outputs
    [x_data, ~, z_data, a_data, ~] = feedForwardPINN(weights_cell, biases_cell, t_data);
    t_physics = start_t + sort(abs(start_t-end_t).*(rand(1 , n_physics)-0));  % uncomment for random physisc collocation points each epoch
    % t_physics = t_physics ./ t_physics_std;
    t_physics(1) = t_physics_ori;
    [x_physics, dxdt_physics, z_physics, a_physics, s_physics] = feedForwardPINN(weights_cell, biases_cell, t_physics);

    %%% Compute loss
    L_data(e) = lossData(x_data(obs_states_idx,:), x_meas);
    [L_physics(e), r] = lossPhysics(x_physics, dxdt_physics, f_rhs, theta);
    L_tot(e) = 2*(alpha_physics.*L_physics(e) + alpha_data.*L_data(e)); % Total loss, mal 2 because otherwise total loss is smaller than indicudal loss when lambd are 1 - lambdother

    %%% Compute gradients
    [W_grad_data, b_grad_data] = backpropObsData(x_data, x_meas, t_data, weights_cell, z_data, a_data, obs_states_idx); % backrpop on observable data
    [W_grad_physics, b_grad_physics] = backpropPhysics(x_physics, t_physics, weights_cell, z_physics, a_physics, s_physics, r, Jf, theta);

    % %%% Check gradients via comparison with central difference
    % check_gradients('data', t_data, t_physics, weights_cell, biases_cell, W_grad_data, b_grad_data, x_meas, theta, f_rhs, alpha_data, alpha_physics)
    % check_gradients('physics', t_data, t_physics, weights_cell, biases_cell, W_grad_physics, b_grad_physics, x_meas, theta, f_rhs, alpha_data, alpha_physics)

    % %%% GradientClipping
    % [W_grad_data, b_grad_data, W_grad_physics, b_grad_physics] = gradientClippingPINN(W_grad_data, b_grad_data, W_grad_physics, b_grad_physics, 0.05);

    % %%% Do Adam step to update the weights and biases
    [weights_cell, m_momentum_weights, v_squared_grad_weights] = doAdamStep(learn_rate, emv_momentum, emv_sqgrad, e, weights_cell, W_grad_data, W_grad_physics, num_forward_steps, alpha_data, alpha_physics, m_momentum_weights, v_squared_grad_weights);
    [biases_cell, m_momentum_bias, v_squared_grad_bias] = doAdamStep(learn_rate, emv_momentum, emv_sqgrad, e, biases_cell, b_grad_data, b_grad_physics, num_forward_steps, alpha_data, alpha_physics, m_momentum_bias, v_squared_grad_bias);

    %%% Update diagonistc plots
    if mod(e, plot_interval) == 0
        % Update loss evolution
        L = {L_data, L_physics, L_tot};
        for i = 1:3
            L_plt{i}.XData = 1:e;
            L_plt{i}.YData = L{i}(1:e);
        end

        % Compute current PINN prediction on test input
        x_test = feedForwardPINN(weights_cell, biases_cell, t_test);
        for i = 1:output_dim
            pred_plt{i}.YData = x_test(i,:);

            [~, x_physics_true] = ode45(odefun, t_physics(1,:), x0, ode_opts);  % get groundtruth
            phys_scat{i}.XData = t_physics(1,:);
            phys_scat{i}.YData = x_physics_true(:,i);
        end

        % Update layer norms
        for i = 1:num_forward_steps

            gradient_norm(1,i) = norm(W_grad_data{i})    / norm(weights_cell{i});
            gradient_norm(2,i) = norm(W_grad_physics{i}) / norm(biases_cell{i});

            % gradient_norm(1,i) = norm(W_grad_data{i})   ;
            % gradient_norm(2,i) = norm(W_grad_physics{i});

        end
        for i = 1:2
            grad_plt(i).YData = gradient_norm(i,:);
        end
        
        % % Update gradient distribution plot
        % W_grad_phyiscs_column_array = cellfun(@(x) x(:), W_grad_physics, 'UniformOutput', false);
        % b_grad_physics_column_array = cellfun(@(x) x(:), b_grad_physics, 'UniformOutput', false);
        % W_grad_data_column_array = cellfun(@(x) x(:), W_grad_data, 'UniformOutput', false);
        % b_grad_data_column_array = cellfun(@(x) x(:), b_grad_data, 'UniformOutput', false);
        % flat_W_grad_physics = vertcat(W_grad_phyiscs_column_array{:});
        % flat_b_grad_physics = vertcat(b_grad_physics_column_array{:});
        % flat_W_grad_data = vertcat(W_grad_data_column_array{:});
        % flat_b_grad_data = vertcat(b_grad_data_column_array{:});
        % 
        % 
        % histogram(hist_ax1, flat_b_grad_physics, 'FaceAlpha', 0.5, 'FaceColor', [0,0.5,0]);
        % histogram(hist_ax1, flat_W_grad_physics, 'FaceAlpha', 0.5, 'FaceColor', [0,0.8,0]);
        % histogram(hist_ax2, flat_b_grad_data,    'FaceAlpha', 0.5, 'FaceColor', [0 0 0.5]);
        % histogram(hist_ax2, flat_W_grad_data,    'FaceAlpha', 0.5, 'FaceColor', [0 0 1]);
    
        % Update network visualization
        for i = 1:num_forward_steps
            weights_grad = alpha_data.*W_grad_data{i} + alpha_physics.*W_grad_physics{i};
            biases_grad = alpha_data.*b_grad_data{i} + alpha_physics.*b_grad_physics{i};

            % % Get Adam update
            % eps = 1e-8;  % Helper for numerical stability. To avoid dividing by zero when rescaling the learing rate per parameter
            % m_hat_w = m_momentum_weights_cell{i}./(1 - emv_momentum.^e);
            % v_hat_w = v_squared_gradients_weights_cell{i}./(1 - emv_sqgrad.^e);
            % m_hat_b = m_momentum_bias_cell{i}./(1 - emv_momentum.^e);
            % v_hat_b = v_squared_gradients_bias_cell{i}./(1 - emv_sqgrad.^e);
            % W_update = - learn_rate./(sqrt(v_hat_w) + eps) .* m_hat_w;
            % B_update = - learn_rate./(sqrt(v_hat_b) + eps) .* m_hat_b;
            % weights_grad_cell_plot{i} = [B_update./(biases_cell{i} + 1e-6), W_update./(weights_cell{i} + 1e-6)];

            weights_grad_cell_plot{i} = [biases_grad./(biases_cell{i} + 1e-6), weights_grad./(weights_cell{i} + 1e-6)];
            % weights_grad_cell_plot{i} = [biases_grad, weights_grad];
            % weights_cell_plot{i} = [biases_cell{i}, weights_cell{i}];
        end
        % update_weight_colors(line_cell, weights_grad_cell_plot, num_hidden_layers)

        %%% 2D Loss Update
        % [L_tot_2D, L_data_2D, L_physics_2D] = compute2DLossLandscape(weights_cell, biases_cell, t_data, t_physics, x_meas, f_rhs, theta, alpha_data, alpha_physics, res, alpha, beta);
        % srf.ZData = L_physics_2D;
        % contour(ax1, A, B, L_physics_2D, 50);
        % ax1.ZScale = 'log';
        % ax1.ColorScale = 'log';

        drawnow limitrate
        % pause(0)
    end
end
fprintf('\n\t Training Time: %.2f s \n', toc);
fprintf('\n\t Min total loss: \t %.3e ', min(L_tot));
fprintf('\n\t Min data loss: \t %.3e', min(L_data));
fprintf('\n\t Min physics loss: \t %.3e \n', min(L_physics));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [L_tot, L_data, L_physics] = compute2DLossLandscape(weights_cell, biases_cell, t_data, t_physics, x_meas, f_rhs, lambda, alpha_data, alpha_physics, res, alpha, beta)

n_layers = length(weights_cell);
param_vec = [];
for i = 1:n_layers
    param_vec = [param_vec; weights_cell{i}(:)];
end

rng(42)
n_param = length(param_vec);

% orthogonal perturbation directions
P = pca(randn(n_param));
d1 = P(:, 1);
d2 = P(:, 2);

% % Random perturbation directions
% d1 = randn(n_param, 1);
% d2 = randn(n_param, 1);

% Normalize
d1 = d1./norm(d1);
d2 = d2./norm(d2);

% dot(d1,d2)

% Preallocate output arrays for surf plotting
L_data    = zeros(res);
L_physics = zeros(res);
L_tot     = zeros(res);

for i = 1:res
    for j = 1:res
        % Perturb the flattened parameter vector
        param_vec_perturbed = param_vec + alpha(i).*d1 + beta(j).*d2;

        % Helper variable for reshaping the parameter vector back to the appropriate matrices
        index_tail = 0;

        % Reshape back to weights cells to make the forward pass
        weights_cell_perturbed = weights_cell;
        for k = 1:n_layers
            % Accumulate index
            index_curr = index_tail + numel(weights_cell{k});
            % Get weights from the current layer i
            if k == n_layers % handle the end of the thing
                weights_i_flat = param_vec_perturbed(index_tail + 1:end);
            else
                weights_i_flat = param_vec_perturbed(index_tail + 1:index_curr);
            end
            % Reshape to matrix
            weights_cell_perturbed{k} = reshape(weights_i_flat, size(weights_cell{k}));
            % Increment tailing index
            index_tail = index_curr;
        end

        %%% Forward Pass
        x_data = feedForwardPINN(weights_cell_perturbed, biases_cell, t_data);
        [x_physics, dxdt_physics] = feedForwardPINN(weights_cell_perturbed, biases_cell, t_physics);

        %%% Compute loss
        L_data(i,j) = lossData(x_data, x_meas);
        L_physics(i,j) = lossPhysics(x_physics, dxdt_physics, f_rhs, lambda);
        L_tot(i,j) = 2*(alpha_physics.*L_physics(i,j) + alpha_data.*L_data(i,j)); % Total loss, mal 2 because otherwise total loss is smaller than indicudal loss when lambd are 1 - lambdother
     
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Function for the weights ubdate. Works also for the biases
function [weights_cell, momentum_weights, squared_weights] = doAdamStep(learn_rate, emv_momentum, emv_sqgrad, epoch, weights_cell, W_grad_data, W_grad_physics, num_forward_steps, alpha_data, alpha_physics, momentum_weights, squared_weights)

eps = 1e-8;  % Helper for numerical stability. To avoid dividing by zero when rescaling the learing rate per parameter

% Form the final gradients and do the weight updates
for i = 1:num_forward_steps

    % Complete the chain rule by adding the gradients from all loss function terms
    weights_grad = alpha_data.*W_grad_data{i} + alpha_physics.*W_grad_physics{i};

    % Compute first and second moments to propagate information from previous gradients into the current update
    momentum_weights{i} = emv_momentum.*momentum_weights{i}  +  (1 - emv_momentum).*weights_grad;
    squared_weights{i} = emv_sqgrad.*squared_weights{i}  +  (1 - emv_sqgrad).*weights_grad.^2;

    % Momentum bias correction to mitigate the bias towards the zero vector initialization
    m_hat = momentum_weights{i}./(1 - emv_momentum.^epoch);
    v_hat = squared_weights{i}./(1 - emv_sqgrad.^epoch);

    % Update weights with Adam
    weights_cell{i} = weights_cell{i} - learn_rate./(sqrt(v_hat) + eps) .* m_hat;

end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Gradient Checking to verfiy backpropagation implementation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function check_gradients(lossType, t_data, t_physics, weights_cell, biases_cell, weights_grad_cell, biases_grad_cell, y_meas_data, lambda, f_rhs, alpha_data, alpha_physics)
% Gradient checking for both weights and biases
% weights_cell, biases_cell: current parameters
% weights_grad_cell, biases_grad_cell: gradients from backprop
% t_data: input for data loss
% t_col: input for physics loss
% y_meas_data: measured output for data loss
% lambda: parameters for RHS function (if needed in physics loss)
% f_rhs: funciton handle to the right hand side of the ODE
% alpha_data: loss weight for the data loss term
% alpha_physics: loss weight for the physics loss term

epsilon = 1e-6;
n_layers = numel(weights_cell);

%%% 1. Check weight gradients
numerical_weights_grad = cell(n_layers,1);

for k = 1:n_layers
    W = weights_cell{k};
    numerical_grad = zeros(size(W));

    for i = 1:size(W,1)
        for j = 1:size(W,2)
            % Positive perturbation
            W_pos = W;
            W_pos(i,j) = W_pos(i,j) + epsilon;
            weights_cell_pos = weights_cell;
            weights_cell_pos{k} = W_pos;
            L_pos = localLoss(lossType, weights_cell_pos, biases_cell, t_data, t_physics, y_meas_data, lambda, f_rhs, alpha_data, alpha_physics);

            % Negative perturbation
            W_neg = W;
            W_neg(i,j) = W_neg(i,j) - epsilon;
            weights_cell_neg = weights_cell;
            weights_cell_neg{k} = W_neg;
            L_neg = localLoss(lossType, weights_cell_neg, biases_cell, t_data, t_physics, y_meas_data, lambda, f_rhs, alpha_data, alpha_physics);

            % Central difference
            numerical_grad(i,j) = (L_pos - L_neg) / (2*epsilon);
        end

    end

    numerical_weights_grad{k} = numerical_grad;
    fprintf(sprintf('Max %s weight grad. diff. in layer %d:\n', lossType, k));
    disp(max(numerical_grad - weights_grad_cell{k},[],'all'));
end

%%% 2. Check bias gradients
numerical_biases_grad = cell(n_layers,1);

for k = 1:n_layers
    b = biases_cell{k};
    dims = size(b);
    numerical_grad = zeros(dims);

    for i = 1:dims(1)
        % Positive perturbation
        b_pos = b;
        b_pos(i) = b_pos(i) + epsilon;
        biases_cell_pos = biases_cell;
        biases_cell_pos{k} = b_pos;
        L_pos = localLoss(lossType, weights_cell, biases_cell_pos, t_data, t_physics, y_meas_data, lambda, f_rhs, alpha_data, alpha_physics);

        % Negative perturbation
        b_neg = b;
        b_neg(i) = b_neg(i) - epsilon;
        biases_cell_neg = biases_cell;
        biases_cell_neg{k} = b_neg;
        L_neg = localLoss(lossType, weights_cell, biases_cell_neg, t_data, t_physics, y_meas_data, lambda, f_rhs, alpha_data, alpha_physics);

        % Central difference
        numerical_grad(i) = (L_pos - L_neg) / (2*epsilon);
    end

    numerical_biases_grad{k} = numerical_grad;
    fprintf(sprintf('Max %s bias grad. diff. in layer %d:\n', lossType, k));
    disp(max(numerical_grad - biases_grad_cell{k},[],'all'));
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function L = localLoss(type, weights_cell, biases_cell, t_data, t_physics, y_meas_data, lambda, f_rhs, alpha_data, alpha_physics)
switch type
    case 'data'
        x = feedForwardPINN(weights_cell, biases_cell, t_data);
        L = lossData(x, y_meas_data);

    case 'physics'
        [x, dxdt] = feedForwardPINN(weights_cell, biases_cell, t_physics);
        L = lossPhysics(x, dxdt, f_rhs, lambda);

    case 'both'
        x_d = feedForwardPINN(weights_cell,biases_cell, t_data);
        [x_p, dxdt_p] = feedForwardPINN(weights_cell,biases_cell, t_physics);
        L = alpha_data * lossData(x_d, y_meas_data) ...
            + alpha_physics * lossPhysics(x_p, dxdt_p, f_rhs, lambda);
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%









% Set-up a figure for the neural network visualization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [neuro_handle, line_cell] = setup_NN_canvas(dense_NeuralNet_design)
% The following algorithm creates a figure with scatter points for the neurons as well as the in and outputs for the defined dense neural network.
% The parameter "b" defines the fraction of the the left and right side of the axis with limits [0,1] where the inputs and outputs are aligned. The hidden layers are then evenly spaced between the in and outputs
% The parameter "h" defines the fraction of the the bottom and top side of the axis to be left free. The neurons are then evenly spaced between the in and outputs

sz = get(groot, 'ScreenSize');
fig = figure('Position',[sz(3)*0.51, sz(4)*0.51, sz(3)*0.47, sz(4)*0.38], 'ToolBar', 'none');
% fig = figure('Position',[sz(3)*0.05, sz(4)*0.08, sz(3)*0.9, sz(4)*0.84], 'ToolBar', 'none'); % FullScreen
ax = axes(fig, 'Position',[0,0,1,1]);
hold(ax, 'on');

% The following algorithm creates a figure with scatter points for the neurons as well as the in and outputs for the defined dense neural network.
% The parameter "b" defines the fraction of the the left and right side of the axis with limits [0,1] where the inputs and outputs are aligned. The hidden layers are then evenly spaced between the in and outputs
% The parameter "h" defines the fraction of the the bottom and top side of the axis to be left free. The neurons are then evenly spaced between the in and outputs

num_neurons_per_layer = dense_NeuralNet_design;
num_neurons_per_layer(1:end-1) = dense_NeuralNet_design(1:end-1) + 1; % +1 to acocunt for the bias neuron
num_layers = numel(num_neurons_per_layer);
num_hidden_layers = num_layers - 2; % without input and output layer

%%%%%
% create the coordinates for the scatter points that represent the neurons
b = 0.1; % side padding, must be a value between 0 and <0.5
h = 0.1; % height padding, must be a value between 0 and <0.5

layer_x_spacing = (1 - 2*b)/(num_hidden_layers + 1);

x_coordinate = zeros(sum(num_neurons_per_layer), 1);
y_coordinate = zeros(sum(num_neurons_per_layer), 1);

runner = 0;
for i = 1:num_layers
    for j = 1:num_neurons_per_layer(i)
        runner = runner + 1;
        x_coordinate(runner) = b + layer_x_spacing*(i-1);

        if num_neurons_per_layer(i) == 1
            y_coordinate(runner) = 0.5;
        else
            layer_y_spacing = (1 - 2*h)/(num_neurons_per_layer(i) - 1);
            y_coordinate(runner) = 1 - (h + layer_y_spacing*(j-1));
        end
    end
end

% make color matrix for neurons
index_bias = cumsum(num_neurons_per_layer(1:end-1)); % get the index of the bias neurons to plot them separately. makes coloring of the active neurons later easier
x_coordi_neuro = x_coordinate(setdiff(1:sum(num_neurons_per_layer), index_bias));
x_coordi_bias = x_coordinate(index_bias);
y_coordi_neuro = y_coordinate(setdiff(1:sum(num_neurons_per_layer), index_bias));
y_coordi_bias = y_coordinate(index_bias);

%%%%%
% create the lines that connect the neurons
linwidth = 2;
line_cell = cell(num_layers-1, 1);  % the handle for each line is stored in this nested cell. The top level contains the cells wich contain all lines connecting two layers, so there are num_layers-1 top level cells
for i = 1:num_layers-1
    runner_v2 = 0;
    line_cell{i} = cell(num_neurons_per_layer(i) * dense_NeuralNet_design(i+1), 1); % there are num_neurons_layer_k * (num_neurons_layer_k+1 - 1) layer-to-layer connections

    if i == num_layers-1
        by_pass_bias = 0;  % for the last layer where there is no bias we need to set this to 0 to avoid connecting the last output
    else
        by_pass_bias = 1;
    end

    for j = 1:num_neurons_per_layer(i)
        for k = 1:num_neurons_per_layer(i+1) - by_pass_bias
            runner_v2 = runner_v2 + 1;
            if i == 1
                line_cell{i}{runner_v2} = line(ax, [x_coordinate(j), x_coordinate(index_bias(i) + k)], [y_coordinate(j), y_coordinate(index_bias(i) + k)], 'LineWidth', linwidth);
            else
                line_cell{i}{runner_v2} = line(ax, [x_coordinate(index_bias(i-1) + j), x_coordinate(index_bias(i) + k)], [y_coordinate(index_bias(i-1) + j), y_coordinate(index_bias(i) + k)], 'LineWidth', linwidth);
            end
        end
    end
end
%%%%%

% plot the neurons on the axis
neuro_handle = scatter(ax, x_coordi_neuro, y_coordi_neuro, 400, [0, 0.3, 0.7], 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 1.6);
scatter(ax, x_coordi_bias, y_coordi_bias, 400, [1, 1, 0], 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 1.6);  % since the bias neurons wont change color it is much easier to make a separate scatter object for them.

ax.XLim = [0, 1];
ax.YLim = [0, 1];
ax.Color = 'k';
ax.XTick = [];
ax.YTick = [];

% colormap((slanCM('iceburn')));
% colorbar(ax, 'Location', 'west', 'Ticks', []);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function update_neuron_colors(dense_NeuralNet_design, Inputs, Outputs, hiddenOutputs, neuro_handle, Num_Hidden_Layers, n_inputs, n_outputs)
% function to assign colors to the cooresponding lines and neurons
neuron_state_vector = zeros(sum(dense_NeuralNet_design), 1);

for i = 1:Num_Hidden_Layers + 2
    index_bias = cumsum(dense_NeuralNet_design);
    if i == 1
        neuron_state_vector(1:n_inputs) = Inputs;
    elseif i == Num_Hidden_Layers + 2
        neuron_state_vector(end-n_outputs + 1:end) = Outputs;
    else
        neuron_state_vector(index_bias(i-1)+1:index_bias(i)) = hiddenOutputs{i-1};
    end
end
neuro_handle.CData = twoColorColormap(neuron_state_vector);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function update_weight_colors(line_cell, weights_gradients_cell, Num_Hidden_Layers)
% function to assign colors to the lines/weights, they are colored relative to each other, no grouping by layers. should make differences in the magnitude of the weight across layers visible
num_weights = zeros(numel(line_cell), 1);
for i = 1:numel(line_cell)
    num_weights(i) = numel(line_cell{i});
end
num_weights = cumsum(num_weights);

weights_state_vector = zeros(num_weights(end), 1);

for i = 1:Num_Hidden_Layers + 1
    if i == 1
        weights_state_vector(1:num_weights(i)) = weights_gradients_cell{i}(:);
    else
        weights_state_vector(num_weights(i-1) + 1:num_weights(i)) = weights_gradients_cell{i}(:);
    end
end


% Get colors
weights_color_vector = twoColorColormap(weights_state_vector);




% MIN = min(weights_state_vector);
% MAX = max(weights_state_vector);
% scaled = (weights_state_vector - MIN)./(MAX - MIN);
% 
% 
% 
% 
% % [~, order] = sort(weights_state_vector);
% col = colormap((slanCM('iceburn'))); % 'seaweed'
% n_colors = length(weights_state_vector);
% interp_col = interp1((1:size(col, 1))/size(col, 1), col, scaled);
% weights_color_vector = interp_col;


% [~, order] = sort(weights_state_vector);
% % col = colormap((slanCM('iceburn'))); % 'seaweed'
% col = colormap(plasma); % 'seaweed'
% n_colors = length(weights_state_vector);
% interp_col = interp1(1:size(col, 1), col, linspace(1, size(col, 1), n_colors));
% weights_color_vector = interp_col(order,:);


% Apply each color to the desired line
runner = 0;
for i = 1:numel(line_cell)
    for j = 1:numel(line_cell{i})
        runner = runner + 1;
        line_cell{i}{j}.Color = weights_color_vector(runner, :);
    end
end
%%%%%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function colormap = twoColorColormap(weights_state_vector)

MIN = min(weights_state_vector);
MAX = max(weights_state_vector);

scaled = (weights_state_vector - MIN)./(MAX - MIN);
colormap = [scaled, 0.25 * ones(length(weights_state_vector),1), 1-scaled];

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



















%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function L_plt = setup_loss_evolution_canvas()
% Create screen size independent figure
sz = get(groot, 'ScreenSize');
fig = figure('Position',[sz(3)*0.03, sz(4)*0.51, sz(3)*0.47, sz(4)*0.38], 'Color',' w');
ax = axes(fig);
hold on;    grid on;

display_name = {'L_{data}', 'L_{physics}', 'L_{total}'};
colors = [0,0.8,0; 0,0,1; 1,0,0];

L_plt = cell(3,1);
for i = 1:3
    L_plt{i} = plot(ax, 0, 0, 'Color', colors(i,:), 'DisplayName', display_name{i}, 'LineWidth', 1.5);
end

xlabel('Epoch  [-]', 'FontSize', 15)
ylabel('MSE Loss  [-]', 'FontSize', 15)
legend('FontSize', 15); % slows down the animation...

% Static legend
setappdata(ax,'LegendColorbarManualSpace',1);
setappdata(ax,'LegendColorbarReclaimSpace',1);

ax.YScale = 'log';
% ax.YLim = [1e-4, 1];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [pred_plt, phys_scat] = setup_output_plot(x_meas, x_physics_true, t_data, t_physics, t_test, obs_states_idx)
% Create screen size independent figure
sz = get(groot, 'ScreenSize');
figure('Position',[sz(3)*0.03, sz(4)*0.08, sz(3)*0.47, sz(4)*0.38], 'Color',' w');
% ax = axes(fig);

phys_scat = cell(size(x_meas, 1), 1);
pred_plt  = cell(size(x_meas, 1), 1);

str = {'Biomass  [g/L]', 'Substrate  [g/L]', 'DOT  [%]', 'Volume  [L]'};

tyl = tiledlayout(2,2, 'TileSpacing','compact');
for i = 1:size(x_meas, 1)
    ax = nexttile;
    hold on;    grid on;
    if ismember(i, obs_states_idx) % plot only obervable states
        scatter(ax, t_data(1,:), x_meas(i,:), 'LineWidth', 1.5, 'DisplayName', sprintf('x_{%d} - noisy meas.', i))   % Scatter plot for the measured data
    end
    phys_scat{i} = scatter(ax, t_physics(1,:), x_physics_true(:,i), 'DisplayName', sprintf('x_{%d} - physics colloc.',i), 'Marker', 'x');   % Scatter plot for the measured data
    pred_plt{i} = plot(ax, t_test(1,:), zeros(1,length(t_test(1,:))), 'DisplayName', sprintf('x_{%d} - PINN pred.', i), 'LineWidth', 1.8);
    % legend('FontSize', 11, 'Location','best');
    minVal = min([x_meas(i,:)';x_physics_true(:,i)]);
    maxVal = max([x_meas(i,:)';x_physics_true(:,i)]);
    range = maxVal - minVal;
    ax.YLim = [minVal - range*0.1, maxVal + range*0.1];
    ax.XLim = [min([t_data(:);t_physics(:);t_test(:)]), max([t_data(:);t_physics(:);t_test(:)])];
    ylabel(ax, str{i}, 'FontSize', 11)
end

% Formatting
xlabel(tyl, 'Time  [h]', 'FontSize', 15)
% ylabel(tyl, 'Output', 'FontSize', 15)

setappdata(ax,'LegendColorbarManualSpace',1); % Static legend
setappdata(ax,'LegendColorbarReclaimSpace',1); % Static legend


end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function grad_plt = setup_gradient_norm_plot(num_forward_steps)
% Create screen size independent figure
sz = get(groot, 'ScreenSize');
fig = figure('Position',[sz(3)*0.51, sz(4)*0.08, sz(3)*0.47, sz(4)*0.38], 'Color',' w');
ax = axes(fig);
hold on;    grid on;

% Plot gradient norm of data term
grad_plt = bar(ax, 1:num_forward_steps, zeros(num_forward_steps,2), 'LineWidth', 1.5);
grad_plt(1).FaceColor = [0,0.8,0]; % RGB for data
grad_plt(2).FaceColor = [0 0 1];   % RGB for physics

% Formatting
xlabel('Layer  ', 'FontSize', 15)
ylabel('Gradient norm', 'FontSize', 15)
% legend({'$\|\nabla w^{(\ell)} \mathcal{L}_{data}\|_2$', '$\|\nabla w^{(\ell)} \mathcal{L}_{physics}\|_2$'}, 'FontSize', 15, 'Interpreter', 'latex');
legend({'Data', 'Physics'}, 'FontSize', 15);
% Static legend
setappdata(ax,'LegendColorbarManualSpace',1);
setappdata(ax,'LegendColorbarReclaimSpace',1);
ax.XTick = 1:num_forward_steps;
% ax.YLim = [1e-2,1e4];
% ax.YLim = [1e-5, 1];
ax.YScale = 'log';
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ax, A, B] = setupLossLandscape2D(weights_cell, biases_cell, t_data, t_physics, x_meas, f_rhs, lambda, alpha_data, alpha_physics, res, alpha, beta)

%%% 2D Loss landscape Plot init
sz = get(groot, 'ScreenSize');
fig = figure('Position',[sz(3)*0.51, sz(4)*0.18, sz(3)*0.47, sz(4)*0.7]);
ax = axes(fig);

[A, B] = meshgrid(alpha, beta);
[L_tot_2D, ~, ~] = compute2DLossLandscape(weights_cell, biases_cell, t_data, t_physics, x_meas, f_rhs, lambda, alpha_data, alpha_physics, res, alpha, beta);

% surf_handle = surf(A, B, (L_physics_2D));
% ax1.View = [10,65];
contour(ax, A, B, (L_tot_2D), 50);
ax.ZScale = 'log';
ax.ColorScale = 'log';
colormap(ax, viridis)
title(ax, '2D Loss Landscape Visualization', 'FontSize', 15)
xlabel(ax, 'Alpha')
ylabel(ax, 'Beta')

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [hist_ax1, hist_ax2] = setup_gradient_histograms()
% Create screen size independent figure
sz = get(groot, 'ScreenSize');
figure('Position',[sz(3)*0.51, sz(4)*0.08, sz(3)*0.47, sz(4)*0.38], 'Color',' w');

tiledlayout(1,2, 'TileSpacing', 'compact')

hist_ax1 = nexttile;
% hold on;    grid on;

histogram(zeros(10,1), 'FaceAlpha', 0.5, 'FaceColor', [0,0.8,0]);
histogram(zeros(10,1), 'FaceAlpha', 0.5, 'FaceColor', [0,0.8,0]);
xlabel('Data gradient values [-]', 'FontSize', 15)
ylabel('Number of weights in bin', 'FontSize', 15)

hist_ax2 = nexttile;
% hold on;    grid on;

histogram(zeros(10,1), 'FaceAlpha', 0.5, 'FaceColor', [0 0 1]);
histogram(zeros(10,1), 'FaceAlpha', 0.5, 'FaceColor', [0,0.8,0]);
xlabel('Physicsc gradient values [-]', 'FontSize', 15)
ylabel('Number of weights in bin', 'FontSize', 15)


end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


