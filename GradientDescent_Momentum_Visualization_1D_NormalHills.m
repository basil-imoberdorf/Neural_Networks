clear
close all


% Landscape params
limits = [0,20 ; -15,15]; % [x_min, x_max; y_min, y_max] % define borders of the plot
kappa = 4; % Condition number = max_eigenvalue/min_eigenvalue
a = 6;
p = 75;
s = 1.4;
slope = 30;

% Presets
Record_Video = 0;  % creates a MPEG-4 video of the simulation if set to 1
Capture_Frame_Ratio = 1;
max_iterations = 280;

learn_rate1 = 0.01;
learn_rate2 = 0.01;
learn_rate3 = 0.01;

momentum_coeff = 0.91;
momentum_vector = [0; 0];

momentum_coeff1 = 0.96;
momentum1_vector1 = [0; 0];

loss_history1 = zeros(max_iterations+1, 1);
loss_history2 = zeros(max_iterations+1, 1);
loss_history3 = zeros(max_iterations+1, 1);

theta = zeros(2, max_iterations);
theta_momentum = zeros(2, max_iterations);
theta_momentum1 = zeros(2, max_iterations);


theta(:,1) = [19; 5];
theta_momentum(:,1) = [19; 0]; % warm-start momentum 
theta_momentum1(:,1) = [19; -5]; % warm-start momentum 

Frame_Collection = struct('cdata', cell(1,max_iterations/Capture_Frame_Ratio), 'colormap', cell(1,max_iterations/Capture_Frame_Ratio)); %preallocate the array for the video frames
Frame_Index = 0;

% Set-up loss-landscape plot
[standard_handle, momentum_handle, momentum1_handle, ax] = create_canvas(theta, a, p, s, slope, limits);
pause(0)

if Record_Video == 1
    Filename = 'Momentum_GD_x17_y0_1st0.91_2nd0.96_DownHill_NEW_markersize';
    Movie = VideoWriter(Filename,'MPEG-4');
    Movie.FrameRate = 30;
    Movie.Quality = 100;
    open(Movie)
end

for i = 1:max_iterations
    
    % Calculate loss
    loss_history1(i) = feed_forward(theta(1, i), theta(2, i), a, p, s, slope);
    loss_history2(i) = feed_forward(theta_momentum(1, i), theta_momentum(2, i), a, p, s, slope);
    loss_history3(i) = feed_forward(theta_momentum1(1, i), theta_momentum1(2, i), a, p, s, slope);

    % Calculate gradient
    gradient = compute_gradient(theta(1, i), theta(2, i), a, p, s, slope);
    gradient_momentum = compute_gradient(theta_momentum(1, i), theta_momentum(2, i), a, p, s, slope);
    gradient_momentum1 = compute_gradient(theta_momentum1(1, i), theta_momentum1(2, i), a, p, s, slope);
    % check_gradients(1e-5, gradient, theta(:, i), a, p, s, slope)

    % Warm-start
    if i == 1
        momentum_vector = gradient;
        momentum_vector1 = gradient;
    end

    % Update weights
    theta(:, i+1) = theta(:, i) - learn_rate1.*gradient;

    momentum_vector = (momentum_coeff.*momentum_vector + (1-momentum_coeff)*gradient_momentum);
    theta_momentum(:, i+1) = theta_momentum(:, i) - learn_rate2 .* momentum_vector;
    
    momentum_vector1 = (momentum_coeff1.*momentum_vector1 + (1-momentum_coeff1)*gradient_momentum1);
    theta_momentum1(:, i+1) = theta_momentum1(:, i) - learn_rate3 .* momentum_vector1;
        
    if loss_history3(i) <= 0
        break
    end

    % show evolution
    if mod(i, Capture_Frame_Ratio)==0
        standard_handle.XData = theta(1, 1:i);
        standard_handle.YData = theta(2, 1:i);
        standard_handle.ZData = 3 + loss_history1(1:i);

        momentum_handle.XData = theta_momentum(1, 1:i);
        momentum_handle.YData = theta_momentum(2, 1:i);
        momentum_handle.ZData = 5 + loss_history2(1:i);

        momentum1_handle.XData = theta_momentum1(1, 1:i);
        momentum1_handle.YData = theta_momentum1(2, 1:i);
        momentum1_handle.ZData = 8 + loss_history3(1:i);

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
        ax.View = [-180 + 0.07*i, 8 + 0.08*i];
        pause(1/24)
       

        %capture frames for movie
        if Record_Video == 1
            % Frame_Index = Frame_Index + 1;
            % Frame_Collection(Frame_Index) = getframe(gcf);

            % f = getframe(gca);
            % img = frame2im(f);
            % Frame_Collection(Frame_Index) = im2frame(img);


            % exportgraphics(gca, 'temp_frame.png', 'Resolution', 300);  % or use gcf
            % img = imread('temp_frame.png');
            % Frame_Collection(Frame_Index) = im2frame(img);


            fname = 'image_tiff2'; % full name of image
            print('-dtiff','-r200',fname)     % save image with '-r200' resolution
            I = imread([fname '.tif']);       % read saved image
            writeVideo(Movie,im2frame(I))

            % Frame_Collection(Frame_Index) = im2frame(I); % convert image to frame
         end
    end

end


%make mp4 movie
if Record_Video == 1

    close(Movie)

    % Filename = 'Momentum_GD_x1.2_y1.17_1st0.922_2nd0.965.avi';
    % makeMovie('Momentum_GD_x0.31_y3.45_1st0.94_2nd0.965_NEW_2',Frame_Collection,30)
    % makeMovie(Filename,Frame_Collection,30)
    % 
    % % convert AVI to MP4
    % pathVideoMP4 = regexprep(Filename,'\.avi','.mp4'); % generate mp4 filename
    % if isunix % for linux
    %     [~,~] = system(sprintf('ffmpeg -i %s -y -an -c:v libx264 -crf 0 -preset slow %s',Filename,pathVideoMP4)); % for this to work, you should have installed ffmpeg and have it available on PATH
    % elseif ispc % for windows
    %     [~,~] = system(sprintf('ffmpeg.exe -i %s -y -an -c:v libx264 -crf 0 -preset slow %s',Filename,pathVideoMP4)); % for this to work, you should have installed ffmpeg and have it available on PATH
    % end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%---

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [standard_handle, momentum_handle, momentum2_handle, ax] = create_canvas(theta0, a, p, s, slope, limits)
% Plots
fig = figure('Position', [910,50,1230,1000], 'Color', 'k', 'ToolBar','auto', 'InvertHardcopy', 'off');  % , 'ToolBar','none'
ax = axes(fig, 'Position', [0,0,1,1]);
hold(ax, 'on')

res = 200;
x = linspace(limits(1,1), limits(1,2), res);
y = linspace(limits(2,1), limits(2,2), res);

[X, Y] = meshgrid(x,y);
Z = feed_forward(X, Y, a, p, s, slope);
surf(X,Y,Z, 'EdgeAlpha', 0.01, 'EdgeColor', 'k')
contour(X,Y,Z, 55, 'LineWidth', 1.1, 'EdgeAlpha', 1);
contour3(X,Y,Z, 55, 'LineWidth', 1, 'EdgeAlpha', 0.35, 'EdgeColor', 'k');

colormap(ax, (magma))

momentum_handle = scatter3(ax, theta0(1), 0, feed_forward(theta0(1), theta0(2), a, p, s, slope), 150, ...
    'filled', 'MarkerEdgeColor', [0, 0.5, 0], 'MarkerFaceColor', 'g', 'LineWidth', 1.4);
momentum2_handle = scatter3(ax, theta0(1), -5, feed_forward(theta0(1), theta0(2), a, p, s, slope), 250, ...
    'filled', 'MarkerEdgeColor', [0.5, 0, 0], 'MarkerFaceColor', 'r', 'LineWidth', 1.6);
standard_handle = scatter3(ax, theta0(1), 5, feed_forward(theta0(1), theta0(2), a, p, s, slope), 80, ...
    'filled', 'MarkerEdgeColor', [0, 0, 0.5], 'MarkerFaceColor', 'b', 'LineWidth', 1.2);
    
% latex_string1 = 'Warm start: $v_0 = \nabla f(w_0) $';
% latex_string2 = 'Cold start: $v_0 = \vec 0$';
% annotation('textbox', [0.02, 0.02, 0.3, 0.2], 'String', {'','', latex_string1, '',latex_string2, ''}, ...
%     'Interpreter', 'latex', 'FitBoxToText', 'on', 'EdgeColor', 'w', 'BackgroundColor', 'k', 'FontSize', 30, 'Color', 'w');

%%%%%%%
latex_string1 = '$ \quad v_i = \beta v_i + (1-\beta) \nabla f(w_i) $';
latex_string2 = '$ \quad w_{i+1} = w_i + \eta v_i $';
annotation('textbox', [0.02, 0.1, 0.3, 0.2], 'String', {'','', latex_string1, '',latex_string2, ''}, ...
    'Interpreter', 'latex', 'FitBoxToText', 'on', 'EdgeColor', 'w', 'BackgroundColor', 'k', 'FontSize', 30, 'Color', 'w');

latex_str1 = '$ \quad \beta_1 = 0 $';
latex_str2 = '$ \quad \beta_2 = 0.9 $';
latex_str3 = '$ \quad \beta_3 = 0.96 $';

% First line
annotation('textbox', [0.02, 0.5, 0.3, 0.05], 'String', '$\quad \beta_1 = 0$', ...
    'Interpreter', 'latex', 'FitBoxToText', 'on', 'EdgeColor', 'w', ...
    'BackgroundColor', 'k', 'FontSize', 30, 'Color', [0.2, 0.4, 1]);

% Second line
annotation('textbox', [0.02, 0.43, 0.3, 0.05], 'String', '$\quad \beta_2 = 0.9$', ...
    'Interpreter', 'latex', 'FitBoxToText', 'on', 'EdgeColor', 'w', ...
    'BackgroundColor', 'k', 'FontSize', 30, 'Color', 'g');

% Third line
annotation('textbox', [0.02, 0.36, 0.3, 0.05], 'String', '$\quad \beta_3 = 0.96$', ...
    'Interpreter', 'latex', 'FitBoxToText', 'on', 'EdgeColor', 'w', ...
    'BackgroundColor', 'k', 'FontSize', 30, 'Color', 'r');
%%%%%%%


ax.FontSize = 16;
ax.Color = [0.2,0,0.2];
% ax.ZAxis.Scale = 'log';
ax.XColor = 'k';
ax.YColor = 'k';
ax.ZColor = 'k';
xlabel(ax, '\theta_1', 'Fontsize', 25)
ylabel(ax, '\theta_2', 'Fontsize', 25)
zlabel(ax, 'Loss', 'Fontsize', 20)
%view(ax, -90, 30)
% view(ax, -17, 31)
% view(ax, -65.1395, 8.9850)
% view(ax, 14, 14 )
% view(ax, -57.3765,   42.0814)
% view(ax, -330.1149, 33.5652)
% view(ax, 0, 90)
view(ax, -180, 8) % momentum = 0.965 und 0.922
% view(ax, 82.2261, 49.3653) % momentum = 0.965 und 0.922
grid on

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Forward Pass
function z = feed_forward(x, y, a, p, s, slope)
    % Rastrigin    
    z = slope*x + p*( 2*exp(-((x-a).^2)./s^2) + exp(-((x-2*a).^2)./s^2) );
end

% Backpropagation of gradients
function gradient = compute_gradient(x, y, a, p, s, slope)

% Partial derivative with respect to x
    dfdx = slope + p*( -4*(x-a)/s^2 * exp(-((x-a).^2)./s^2) - 2*(x-2*a)/s^2 * exp(-((x-2*a).^2)./s^2)) ;

    % Partial derivative with respect to y
    dfdy =  0;

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
function check_gradients(epsilon, gradient, theta, a, p, s, slope)
theta_positive_perturbed = theta;
theta_negative_perturbed = theta;
nummerical_gradient = zeros(2, 1);
% This function computes nummerically the gradients and compares them with the gradients obtained by backpropagation. If they match, the backpropagation algorithm works as intended.
for i = 1:2    
    theta_positive_perturbed(i) = theta(i) + epsilon;
    theta_negative_perturbed(i) = theta(i) - epsilon;

    Loss_positive_perturbed = feed_forward(theta_positive_perturbed(1), theta_positive_perturbed(2), a, p, s, slope);
    Loss_negative_perturbed = feed_forward(theta_negative_perturbed(1), theta_negative_perturbed(2), a, p, s, slope);

    nummerical_gradient(i) = (Loss_positive_perturbed - Loss_negative_perturbed)./(2*epsilon);
    
    theta_positive_perturbed = theta;
    theta_negative_perturbed = theta;
end

gradient_differences = (nummerical_gradient - gradient)
percent_error = (nummerical_gradient - gradient)./nummerical_gradient * 100

end