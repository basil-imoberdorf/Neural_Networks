clear
close all


% Landscape params
limits = [-3,3 ; -3,3]; % [x_min, x_max; y_min, y_max] % define borders of the plot
kappa = 3; % Condition number = max_eigenvalue/min_eigenvalue
R = 8;
k = 2;
slope = 0.3;

% Presets
Record_Video = 0;  % creates a MPEG-4 video of the simulation if set to 1
Capture_Frame_Ratio = 1;
max_iterations = 200;

% Leran rates
learn_rate_0 = 0.06;
learn_rate_1 = 0.06;
learn_rate_2 = 0.06;
learn_rate_3 = 0.06;

% Vectors for exponetial moving average (EMA) of gradients. Unlike in "heavy-ball" momentum the magnitude cannot growth without bounds. The coefficient ensures that the momentum vector can maximally be as big as the incoming gradient. 
m_0 = zeros(2,1); 
m_1 = zeros(2,1); 
m_2 = zeros(2,1);
m_3 = zeros(2,1);
m_coeff_0 = 0.0;
m_coeff_1 = 0.9;
m_coeff_2 = 0.0;
m_coeff_3 = 0.9;

% Accumulated squared gradient vector for AdaGrad. Each parameter update is scaled by the inverse square root of the corresponding value
v_0 = zeros(2,1);
v_1 = zeros(2,1);
v_2 = zeros(2,1);
v_3 = zeros(2,1);
v_coeff_0 = 0.0;
v_coeff_1 = 0.0;
v_coeff_2 = 0.9;
v_coeff_3 = 0.9;
eps = 1e-6; % to avoid dividing by zero

loss_history_0 = zeros(max_iterations+1, 1);
loss_history_1 = zeros(max_iterations+1, 1);
loss_history_2 = zeros(max_iterations+1, 1);
loss_history_3 = zeros(max_iterations+1, 1);

theta_0 = zeros(2, max_iterations);
theta_1 = zeros(2, max_iterations);
theta_2 = zeros(2, max_iterations);
theta_3 = zeros(2, max_iterations);

% theta_0(:,1) = [0.31; 3.45];
% theta_0(:,1) = [-1.73; 1.50];
% theta_0(:,1) = [-1.76; 2.03];
% theta_0(:,1) = [-0.653; -0.422];
% theta_0(:,1) = [-1.81; 1.34];
% theta_0(:,1) = [-3.52; 3.52];
% theta_1(:,1) = theta_0(:,1); 
% theta_2(:,1) = theta_0(:,1); 
% theta_3(:,1) = theta_0(:,1); 

theta_0(:,1) = [-3; -3];
theta_1(:,1) = [-3; 3]; 
theta_2(:,1) = [3; -3]; 
theta_3(:,1) = [3; 3]; 

Frame_Collection = struct('cdata', cell(1,max_iterations/Capture_Frame_Ratio), 'colormap', cell(1,max_iterations/Capture_Frame_Ratio)); %preallocate the array for the video frames
Frame_Index = 0;

% Set-up loss-landscape plot
[plot_handle_0, plot_handle_1, plot_handle_2, plot_handle_3, ax] = create_canvas(theta_0, kappa, R, k, slope, limits);
[loss_0_plt, loss_1_plt, loss_2_plt, loss_3_plt] = setup_loss_evolution_canvas(1);
pause(0)

if Record_Video == 1
    Filename = 'Momentum_GD_x-2.89_y3.52_1st0.93_2nd0.96_quadartic_NEW_hot';
    Movie = VideoWriter(Filename,'MPEG-4');
    Movie.FrameRate = 30;
    Movie.Quality = 100;
    open(Movie)
end

for i = 1:max_iterations
    
    % Calculate loss
    loss_history_0(i) = feed_forward(theta_0(1, i), theta_0(2, i), kappa, R, k, slope);
    loss_history_1(i) = feed_forward(theta_1(1, i), theta_1(2, i), kappa, R, k, slope);
    loss_history_2(i) = feed_forward(theta_2(1, i), theta_2(2, i), kappa, R, k, slope);
    loss_history_3(i) = feed_forward(theta_3(1, i), theta_3(2, i), kappa, R, k, slope);

    % Calculate gradient
    gradient_0 = compute_gradient(theta_0(1, i), theta_0(2, i), kappa, R, k, slope);
    gradient_1 = compute_gradient(theta_1(1, i), theta_1(2, i), kappa, R, k, slope);
    gradient_2 = compute_gradient(theta_2(1, i), theta_2(2, i), kappa, R, k, slope);
    gradient_3 = compute_gradient(theta_3(1, i), theta_3(2, i), kappa, R, k, slope);
    %check_gradients(1e-5, gradient_0, theta_0(:, i), kappa, R, k, slope)

    % AdaGrad: Compute second moment, the squared gradients. Dont use: v can only grow so the gradients will vanish eventually
    % v_0 = v_0 + gradient_0.^2; 
    % v_1 = v_1 + gradient_1.^2; 
    % v_2 = v_2 + gradient_2.^2;
    % v_3 = v_3 + gradient_3.^2;

    % % RMSProp: Compute second moment but with balance weighting to not accumulate the squared gradients, more like filtering, averaging the squared gradient over time. EMA
    v_0 = v_coeff_0*v_0  +  (1 - v_coeff_0)*gradient_0.^2; 
    v_1 = v_coeff_1*v_1  +  (1 - v_coeff_1)*gradient_1.^2; 
    v_2 = v_coeff_2*v_2  +  (1 - v_coeff_2)*gradient_2.^2;
    v_3 = v_coeff_3*v_3  +  (1 - v_coeff_3)*gradient_3.^2;

    % EMA-Momentum: Compute the Exponetial Moving Average of the gradient. This is formulation, unlike heavy-ball momentum, does not grow unbounded in magnitude, only changes direction momentum
    m_0 = m_coeff_0*m_0  +  (1 - m_coeff_0)*gradient_0;
    m_1 = m_coeff_1*m_1  +  (1 - m_coeff_1)*gradient_1;
    m_2 = m_coeff_2*m_2  +  (1 - m_coeff_2)*gradient_2;
    m_3 = m_coeff_3*m_3  +  (1 - m_coeff_3)*gradient_3;

    % Bias correction for initialization at zero
    v_0_hat = [1; 1]; % v_0 / (1 - v_coeff_0^i);
    v_1_hat = [1; 1]; % v_1 / (1 - v_coeff_1^i);
    v_2_hat = v_2; % / (1 - v_coeff_2^i);
    v_3_hat = v_3 / (1 - v_coeff_3^i);
    m_0_hat = m_0; %  / (1 - m_coeff_0^i);
    m_1_hat = m_1; % / (1 - m_coeff_1^i);  % For EMA Momentum the bias correction is not done
    m_2_hat = m_2; %  / (1 - m_coeff_2^i);
    m_3_hat = m_3 / (1 - m_coeff_3^i);

    % Update Weights
    theta_0(:, i+1) = theta_0(:, i) - learn_rate_0./(eps + sqrt(v_0_hat)) .* m_0_hat;
    theta_1(:, i+1) = theta_1(:, i) - learn_rate_1./(eps + sqrt(v_1_hat)) .* m_1_hat;
    theta_2(:, i+1) = theta_2(:, i) - learn_rate_2./(eps + sqrt(v_2_hat)) .* m_2_hat;
    theta_3(:, i+1) = theta_3(:, i) - learn_rate_3./(eps + sqrt(v_3_hat)) .* m_3_hat;


    % show evolution
    if mod(i, Capture_Frame_Ratio)==0
        plot_handle_0.XData = theta_0(1, 1:i);
        plot_handle_0.YData = theta_0(2, 1:i);
        plot_handle_0.ZData = loss_history_0(1:i);

        plot_handle_1.XData = theta_1(1, 1:i);
        plot_handle_1.YData = theta_1(2, 1:i);
        plot_handle_1.ZData = loss_history_1(1:i);

        plot_handle_2.XData = theta_2(1, 1:i);
        plot_handle_2.YData = theta_2(2, 1:i);
        plot_handle_2.ZData = loss_history_2(1:i);

        plot_handle_3.XData = theta_3(1, 1:i);
        plot_handle_3.YData = theta_3(2, 1:i);
        plot_handle_3.ZData = loss_history_3(1:i);

        % if i <= 190
        %     ax.View = [ 76.2261 + i*0.08, 39.3653 + i*0.07];
        % % ax.View = [82.2261 - i*0.1, 49.3653 - i*0.08];
        % else
        %     ax.View = [ 76.2261 + 190*0.08, 39.3653 + 0.07*190 + (i-190)*0.08];
        % end

        %ax.View = [ -90.3765 + i*0.08, 42.0814 + 10*sin(2*pi*1/360*i)];
        %ax.View = [-17 - i*0.15, 31 + i*0.05];
        % ax.View = [14 + 0.18*i, 14 + 0.05*i];
        % ax.View = [107 - 0.058*i, 44 - 0.02*i];
        % ax.View = [-6 + 0.02*i, 26 + 0.09*i];
        ax.View = [0, 90 - i*0];

        update_loss_evolution(loss_0_plt, loss_1_plt, loss_2_plt, loss_3_plt, i, loss_history_0, loss_history_1, loss_history_2, loss_history_3)
        pause(0)

        %capture frames for movie
        if Record_Video == 1
            % Frame_Index = Frame_Index + 1;
            % Frame_Collection(Frame_Index) = getframe(gcf);
            fname = 'image_tiff2'; % full name of image
            print('-dtiff','-r200',fname)     % save image with '-r200' resolution
            I = imread([fname '.tif']);       % read saved image
            writeVideo(Movie,im2frame(I))

            % Frame_Collection(Frame_Index) = im2frame(I); % convert image to frame
         end
    end

end


% Close video object
if Record_Video == 1
    close(Movie)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%---

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [plot_handle_0, plot_handle_1, plot_handle_2, plot_handle_3, ax] = create_canvas(theta0, kappa, R, k, slope, limits)
% Plots
fig = figure('Position', [910,50,1230,1000], 'Color', 'k', 'ToolBar','auto', 'InvertHardcopy', 'off');  % , 'ToolBar','none'
ax = axes(fig, 'Position', [0,0,1,1]);
hold(ax, 'on')

res = 120;
x = linspace(limits(1,1), limits(1,2), res);
y = linspace(limits(2,1), limits(2,2), res);

[X, Y] = meshgrid(x,y);
Z = feed_forward(X, Y, kappa, R, k, slope);
surf(X,Y,Z, 'EdgeAlpha', 0.05, 'EdgeColor', 'k')
contour(X,Y,Z, 55, 'LineWidth', 1.1, 'EdgeAlpha', 1);
contour3(X,Y,Z, 55, 'LineWidth', 1, 'EdgeAlpha', 0.35, 'EdgeColor', 'k');

colormap(ax, flip(plasma))

plot_handle_0 = scatter3(ax, theta0(1), theta0(2), feed_forward(theta0(1), theta0(2), kappa, R, k, slope), 100, ...
    'filled', 'MarkerEdgeColor', [0, 0, 0.5], 'MarkerFaceColor', 'b', 'LineWidth', 1.2);
plot_handle_1 = scatter3(ax, theta0(1), theta0(2), feed_forward(theta0(1), theta0(2), kappa, R, k, slope), 100, ...
    'filled', 'MarkerEdgeColor', [0, 0.5, 0], 'MarkerFaceColor', 'g', 'LineWidth', 1.2);
plot_handle_2 = scatter3(ax, theta0(1), theta0(2), feed_forward(theta0(1), theta0(2), kappa, R, k, slope), 100, ...
    'filled', 'MarkerEdgeColor', [0.5, 0, 0], 'MarkerFaceColor', 'r', 'LineWidth', 1.2);
plot_handle_3 = scatter3(ax, theta0(1), theta0(2), feed_forward(theta0(1), theta0(2), kappa, R, k, slope), 100, ...
    'filled', 'MarkerEdgeColor', [0.5, 0, 0.5], 'MarkerFaceColor', 'm', 'LineWidth', 1.2);

scatter3(ax, 0, 0, feed_forward(0, 0, kappa, R, k, slope), 100, ...
    'filled', 'Marker', '*', 'MarkerEdgeColor', 'c', 'LineWidth', 1.2);

legend(ax, {'','','','SGD', 'EMA Momentum', 'RMSProp', 'Adam', 'Global Minimum'}, 'TextColor', 'w', 'Color', 'k', 'FontSize', 20, 'Location', 'W')

ax.FontSize = 16;
ax.Color = [0.2,0,0.2];
% ax.ZAxis.Scale = 'log';
ax.XColor = 'k';
ax.YColor = 'k';
ax.ZColor = 'k';
xlabel(ax, '\theta_1', 'Fontsize', 25)
ylabel(ax, '\theta_2', 'Fontsize', 25)
zlabel(ax, 'Loss', 'Fontsize', 20)
% view(ax, -90, 30)
% view(ax, -17, 31)
% view(ax, -65.1395, 8.9850)
% view(ax, 14, 14 )
% view(ax, -57.3765,   42.0814)
% view(ax, -330.1149, 33.5652)
% view(ax, 0, 90)
view(ax, 15.2261, 39.3653) % momentum = 0.965 und 0.922
% view(ax, 82.2261, 49.3653) % momentum = 0.965 und 0.922
grid on

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Forward Pass
function z = feed_forward(x, y, kappa, r, k, slope)
    % 2D Ripples
    intermed = kappa^2*x.^2 + y.^2;
    z = slope * (intermed - k * cos(intermed) .* exp(-(intermed) / r^2)) + 10;
end

% Backpropagation of gradients
function gradient = compute_gradient(x, y, kappa, R, k, slope)
        % Pre-compute a often used term
    timsav = kappa^2*x.^2 + y.^2;

    % Partial derivative with respect to x
    dfdx = (2 * slope * kappa^2 * x * exp(-(timsav) / R^2) .* (R^2 * k * sin(timsav) + k * cos(timsav) + R^2 * exp((timsav) / R^2))) / R^2;

    % 2 * slope * x * exp(-(timsav) / R^2) * (k^2 * kappa * R^2 * sin(k^2 * x^2 + y^2) + kappa * kappa^2 * cos(k^2 * x^2 + y^2) + k^2 * R^2 * exp((kappa^2 * x^2 + y^2) / R^2))

    % Partial derivative with respect to y
    dfdy = (2 * slope* y .* exp(-(timsav) / R^2) .* (R^2 * k * sin(timsav) + k * cos(timsav) + R^2 * exp((timsav) / R^2))) / R^2;

    gradient = [dfdx; dfdy];
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function makeMovie(Filename, Frame_Collection, Frame_Rate)
%create a .mp4 video clip based on the frames contained in the variable "Frame_Collection"

%Filename: input desired filename as string with ending .mp4
%Frame_Collection: Array containing object. Normaly generated in a loop via getframe: Frame_Collection(loop_index) = getframe(gcf)
%FrameRate: optional parameter that defines how many frames are displayed per second. Default is 60 Fps

switch nargin
    case 2
    Frame_Rate = 60;
end

Movie = VideoWriter(Filename,'MPEG-4');
% Movie = VideoWriter(Filename,'Uncompressed AVI');
Movie.FrameRate = Frame_Rate;
Movie.Quality = 100;
open(Movie)
writeVideo(Movie,Frame_Collection)
close(Movie)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% Gradient Checking to verfiy backpropagation implementation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function check_gradients(epsilon, gradient, theta, kappa, R, k, slope)
theta_positive_perturbed = theta;
theta_negative_perturbed = theta;
nummerical_gradient = zeros(2, 1);
% This function computes nummerically the gradients and compares them with the gradients obtained by backpropagation. If they match, the backpropagation algorithm works as intended.
for i = 1:2    
    theta_positive_perturbed(i) = theta(i) + epsilon;
    theta_negative_perturbed(i) = theta(i) - epsilon;

    Loss_positive_perturbed = feed_forward(theta_positive_perturbed(1), theta_positive_perturbed(2), kappa, R, k, slope);
    Loss_negative_perturbed = feed_forward(theta_negative_perturbed(1), theta_negative_perturbed(2), kappa, R, k, slope);

    nummerical_gradient(i) = (Loss_positive_perturbed - Loss_negative_perturbed)./(2*epsilon);
    
    theta_positive_perturbed = theta;
    theta_negative_perturbed = theta;
end

gradient_differences = (nummerical_gradient - gradient)
percent_error = (nummerical_gradient - gradient)./nummerical_gradient * 100
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Loss-Evolution 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [loss_0_plt, loss_1_plt, loss_2_plt, loss_3_plt] = setup_loss_evolution_canvas(iter_plot_ratio)
% Initialize plot for the loss evolution
fig = figure('Position', [50, 700, 840, 490], 'Color', 'k');
ay = axes(fig, 'Position', [0.12, 0.13, 0.82, 0.84]);
hold(ay, 'on')

loss_0_plt = plot(ay, 0, 0, 'Color', 'b', 'LineWidth', 2);
loss_1_plt = plot(ay, 0, 0, 'Color', 'g', 'LineWidth', 2);
loss_2_plt = plot(ay, 0, 0, 'Color', 'r', 'LineWidth', 2);
loss_3_plt = plot(ay, 0, 0, 'Color', 'm', 'LineWidth', 2);

legend(ay, {'SGD', 'EMA Momentum', 'RMSProp', 'Adam'}, 'TextColor', 'w', 'Color', 'k', 'FontSize', 16, 'Location', 'NE')


ay.FontSize = 15;
ay.XLabel.String = 'Number of weight updates';
ay.YLabel.String = 'Loss';
ay.XLabel.FontSize = 16;
ay.YLabel.FontSize = 16;
ay.XColor = 'y';
ay.YColor = 'y';
ay.Color = 'k';

ay.XGrid = 'on';
ay.YGrid = 'on';
ay.YAxis.Scale = 'log';

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function update_loss_evolution(loss_0_plt, loss_1_plt, loss_2_plt, loss_3_plt, e, loss_history_0, loss_history_1, loss_history_2, loss_history_3)
loss_0_plt.XData = 1:e;
loss_0_plt.YData = loss_history_0(1:e);

loss_1_plt.XData = 1:e;
loss_1_plt.YData = loss_history_1(1:e);

loss_2_plt.XData = 1:e;
loss_2_plt.YData = loss_history_2(1:e);

loss_3_plt.XData = 1:e;
loss_3_plt.YData = loss_history_3(1:e);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%