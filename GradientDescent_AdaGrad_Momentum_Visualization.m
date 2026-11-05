clear
close all

% Peak 1
sig1 = 1.5;
m1 = 0.1;
bx1 = 0.2;
by1 = 3.75;

% Peak 2
sig2 = 1;
m2 = 0.5;
bx2 = -1.93;
by2 = 1.97;

% Peak 3
sig3 = 1.2;
m3 = 0.9;
bx3 = 1.4;
by3 = 1;

% Peak 4
sig4 = 1;
m4 = 0.1;
bx4 = -0.95;
by4 = -1.67;

% Peak 5
sig5 = 0.8;
m5 = 0.4;
bx5 = -0.7;
by5 = -0.5;


% Presets
Record_Video = 0;  % creates a MPEG-4 video of the simulation if set to 1
Capture_Frame_Ratio = 1;
max_iterations = 230;
eps = 1e-8;  % numerical stabilizer, to avoid dividing by zero in AdaGrad

% Leran rates
learn_rate_0 = 0.1;
learn_rate_1 = 0.1;
learn_rate_2 = 0.1;

% Vectors for exponetial moving average (EMA) of gradients. Unlike in "heavy-ball" momentum the magnitude cannot growth without bounds. The coefficient ensures that the momentum vector can maximally be as big as the incoming gradient. 
m_0 = zeros(2,1); 
m_1 = zeros(2,1); 
m_2 = zeros(2,1);
m_coeff_0 = 0.0;
m_coeff_1 = 0.9;
m_coeff_2 = 0.9;

% Accumulated squared gradient vector for AdaGrad. Each parameter update is scaled by the inverse square root of the corresponding value
v_0 = zeros(2,1);
v_1 = zeros(2,1);
v_2 = zeros(2,1);
v_coeff_0 = 0.9;
v_coeff_1 = 0.0;
v_coeff_2 = 0.9;

loss_history_0 = zeros(max_iterations+1, 1);
loss_history_1 = zeros(max_iterations+1, 1);
loss_history_2 = zeros(max_iterations+1, 1);

theta_0 = zeros(2, max_iterations);
theta_1 = zeros(2, max_iterations);
theta_2 = zeros(2, max_iterations);

%theta_0(:,1) = [0.2; 3.55];
theta_0(:,1) = [-1.73; 1.50];
% theta_0(:,1) = [-1.76; 2.03];
% theta_0(:,1) = [-0.653; -0.422];
% theta_0(:,1) = [-1.86; 1.909];
% theta_0(:,1) = [1.29; 1.22];
% theta_0(:,1) = [-0.63; -0.42];
theta_1(:,1) = theta_0(:,1); % warm-start momentum 
theta_2(:,1) = theta_0(:,1); % warm-start momentum 

Frame_Collection = struct('cdata', cell(1,max_iterations/Capture_Frame_Ratio), 'colormap', cell(1,max_iterations/Capture_Frame_Ratio)); %preallocate the array for the video frames
Frame_Index = 0;

% Set-up loss-landscape plot
[standard_handle, momentum_handle, momentum1_handle, ax] = create_canvas(theta_0, sig1, sig2, sig3, sig4, sig5, m1, m2, m3, m4, m5, bx1, bx2, bx3, bx4, bx5, by1, by2, by3, by4, by5);
pause(0)

if Record_Video == 1
    Filename = 'AdaGrad_GD_x-1.86_y1.91_1st0.93_2nd0.97_NEW_ball_width';
    Movie = VideoWriter(Filename,'MPEG-4');
    Movie.FrameRate = 30;
    Movie.Quality = 100;
    open(Movie)
end

for i = 1:max_iterations
    
    % Calculate loss
    loss_history_0(i) = feed_forward(theta_0(1, i), theta_0(2, i), sig1, sig2, sig3, sig4, sig5, m1, m2, m3, m4, m5, bx1, bx2, bx3, bx4, bx5, by1, by2, by3, by4, by5);
    loss_history_1(i) = feed_forward(theta_1(1, i), theta_1(2, i), sig1, sig2, sig3, sig4, sig5, m1, m2, m3, m4, m5, bx1, bx2, bx3, bx4, bx5, by1, by2, by3, by4, by5);
    loss_history_2(i) = feed_forward(theta_2(1, i), theta_2(2, i), sig1, sig2, sig3, sig4, sig5, m1, m2, m3, m4, m5, bx1, bx2, bx3, bx4, bx5, by1, by2, by3, by4, by5);

    % Calculate gradient
    gradient_0 = compute_gradient(theta_0(1, i), theta_0(2, i), sig1, sig2, sig3, sig4, sig5, m1, m2, m3, m4, m5, bx1, bx2, bx3, bx4, bx5, by1, by2, by3, by4, by5);
    gradient_1 = compute_gradient(theta_1(1, i), theta_1(2, i), sig1, sig2, sig3, sig4, sig5, m1, m2, m3, m4, m5, bx1, bx2, bx3, bx4, bx5, by1, by2, by3, by4, by5);
    gradient_2 = compute_gradient(theta_2(1, i), theta_2(2, i), sig1, sig2, sig3, sig4, sig5, m1, m2, m3, m4, m5, bx1, bx2, bx3, bx4, bx5, by1, by2, by3, by4, by5);

    % AdaGrad: Compute second moment, the squared gradients
    % v_0 = v_0 + gradient_0.^2; 
    % v_1 = v_1 + gradient_1.^2; 
    % v_2 = v_2 + gradient_2.^2;

    % % RMSProp: Compute second moment but with balance weighting to not accumulate the squared gradients, more like filtering, averaging the scaling weight
    v_0 = v_coeff_0*v_0  +  (1 - v_coeff_0)*gradient_0.^2; 
    v_1 = v_coeff_1*v_1  +  (1 - v_coeff_1)*gradient_1.^2; 
    v_2 = v_coeff_2*v_2  +  (1 - v_coeff_2)*gradient_2.^2;
    
    % Compute the exponetial moving average of the gradient. This is formulation does not grow in magnitude, only changes direction momentum
    m_0 = m_coeff_0*m_0  +  (1 - m_coeff_0)*gradient_0;
    m_1 = m_coeff_1*m_1  +  (1 - m_coeff_1)*gradient_1;
    m_2 = m_coeff_2*m_2  +  (1 - m_coeff_2)*gradient_2;

    % Bias correction for initialization at zero
    v_0_hat = v_0 / (1 - v_coeff_0^i);
    v_1_hat = [1; 1];%v_1 / (1 - v_coeff_1^i);
    v_2_hat = v_2 / (1 - v_coeff_2^i);
    m_0_hat = m_0 / (1 - m_coeff_0^i);
    m_1_hat = m_1 / (1 - m_coeff_1^i);
    m_2_hat = m_2 / (1 - m_coeff_2^i);
    
    % Update Weights
    theta_0(:, i+1) = theta_0(:, i) - learn_rate_0./(eps + sqrt(v_0_hat)) .* m_0_hat;
    theta_1(:, i+1) = theta_1(:, i) - learn_rate_1./(eps + sqrt(v_1_hat)) .* m_1_hat;
    theta_2(:, i+1) = theta_2(:, i) - learn_rate_2./(eps + sqrt(v_2_hat)) .* m_2_hat;


    % show evolution
    if mod(i, Capture_Frame_Ratio)==0
        standard_handle.XData = theta_0(1, 1:i);
        standard_handle.YData = theta_0(2, 1:i);
        standard_handle.ZData = loss_history_0(1:i);

        momentum_handle.XData = theta_1(1, 1:i);
        momentum_handle.YData = theta_1(2, 1:i);
        momentum_handle.ZData = loss_history_1(1:i);

        momentum1_handle.XData = theta_2(1, 1:i);
        momentum1_handle.YData = theta_2(2, 1:i);
        momentum1_handle.ZData = loss_history_2(1:i);

        % if i <= 190
        %     ax.View = [ 76.2261 + i*0.08, 39.3653 + i*0.07];
        % % ax.View = [82.2261 - i*0.1, 49.3653 - i*0.08];
        % else
        %     ax.View = [ 76.2261 + 190*0.08, 39.3653 + 0.07*190 + (i-190)*0.08];
        % end

        % ax.View = [-110 + i/360*180, 27 + 30*sin(2*pi*1/360*i)];
        ax.View = [-85 - i*0.02, 44 + i*0.18];
        % ax.View = [14 + 0.18*i, 14 + 0.05*i];
        % ax.View = [107 - 0.058*i, 44 - 0.02*i];
        pause(0)

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
function [standard_handle, momentum_handle, momentum2_handle, ax] = create_canvas(theta0, sig1, sig2, sig3, sig4, sig5, m1, m2, m3, m4, m5, bx1, bx2, bx3, bx4, bx5, by1, by2, by3, by4, by5)
% Plots
fig = figure('Position', [910,50,1230,1000], 'Color', 'k', 'ToolBar','auto', 'InvertHardcopy', 'off');  % , 'ToolBar','none'
ax = axes(fig, 'Position', [0,0,1,1]);
hold(ax, 'on')

res = 200;
x = linspace(-3, 2, res);
y = linspace(-4,4, res);

[X, Y] = meshgrid(x,y);
Z = feed_forward(X, Y, sig1, sig2, sig3, sig4, sig5, m1, m2, m3, m4, m5, bx1, bx2, bx3, bx4, bx5, by1, by2, by3, by4, by5);
surf(X,Y,Z, 'EdgeAlpha', 0.1, 'EdgeColor', 'k')
contour(X,Y,Z, 69, 'LineWidth', 1.1, 'EdgeAlpha', 1);
contour3(X,Y,Z, 69, 'LineWidth', 1, 'EdgeAlpha', 0.35, 'EdgeColor', 'k');
% quiver3d(ax, sig1, sig2, sig3, sig4, sig5, m1, m2, m3, m4, m5, bx1, bx2, bx3, bx4, bx5, by1, by2, by3, by4, by5)

% latex_string = '$\quad f(x,y) = \sum_{i=1}^{5} \exp\left(-\frac{(x - a_i)^2 + (y - b_i)^2 + m_i^2}{\sigma_i^2}\right) \quad$';
% annotation('textbox', [0.02, 0.02, 0.3, 0.2], 'String', {'','', latex_string, ''}, ...
%     'Interpreter', 'latex', 'FitBoxToText', 'on', 'EdgeColor', 'w', 'BackgroundColor', 'k', 'FontSize', 35, 'Color', 'w');

colormap(ax, magma)

momentum_handle = scatter3(ax, theta0(1), theta0(2), feed_forward(theta0(1), theta0(2), sig1, sig2, sig3, sig4, sig5, m1, m2, m3, m4, m5, bx1, bx2, bx3, bx4, bx5, by1, by2, by3, by4, by5), 150, ...
    'filled', 'MarkerEdgeColor', [0, 0.5, 0], 'MarkerFaceColor', 'g', 'LineWidth', 1.2);
momentum2_handle = scatter3(ax, theta0(1), theta0(2), feed_forward(theta0(1), theta0(2), sig1, sig2, sig3, sig4, sig5, m1, m2, m3, m4, m5, bx1, bx2, bx3, bx4, bx5, by1, by2, by3, by4, by5), 200, ...
    'filled', 'MarkerEdgeColor', [0.5, 0, 0], 'MarkerFaceColor', 'r', 'LineWidth', 1.2);
standard_handle = scatter3(ax, theta0(1), theta0(2), feed_forward(theta0(1), theta0(2), sig1, sig2, sig3, sig4, sig5, m1, m2, m3, m4, m5, bx1, bx2, bx3, bx4, bx5, by1, by2, by3, by4, by5), 100, ...
    'filled', 'MarkerEdgeColor', [0, 0, 0.5], 'MarkerFaceColor', 'b', 'LineWidth', 1.2);


ax.FontSize = 16;
ax.Color = [0.2,0,0.2];
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
view(ax, 15.2261, 39.3653) % momentum = 0.965 und 0.922
% view(ax, 82.2261, 49.3653) % momentum = 0.965 und 0.922
grid on

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Forward Pass
function z = feed_forward(x, y, sig1, sig2, sig3, sig4, sig5, m1, m2, m3, m4, m5, bx1, bx2, bx3, bx4, bx5, by1, by2, by3, by4, by5)
peak1 =  exp( -(((x - bx1).^2 + (y - by1).^2 + m1.^2)./sig1^2) );
peak2 =  exp( -(((x - bx2).^2 + (y - by2).^2 + m2.^2)./sig2^2) );
peak3 =  exp( -(((x - bx3).^2 + (y - by3).^2 + m3.^2)./sig3^2) );
peak4 = -exp( -(((x - bx4).^2 + (y - by4).^2 + m4.^2)./sig4^2) );
peak5 =  exp( -(((x - bx5).^2 + (y - by5).^2 + m5.^2)./sig5^2) );
z = (peak1 + peak2 + peak3 + peak4 + peak5) + 1;
end

% Forward Pass
function gradient = compute_gradient(x, y, sig1, sig2, sig3, sig4, sig5, m1, m2, m3, m4, m5, bx1, bx2, bx3, bx4, bx5, by1, by2, by3, by4, by5)
peak1 =  exp( -(((x - bx1).^2 + (y - by1).^2 + m1.^2)./sig1^2) );
peak2 =  exp( -(((x - bx2).^2 + (y - by2).^2 + m2.^2)./sig2^2) );
peak3 =  exp( -(((x - bx3).^2 + (y - by3).^2 + m3.^2)./sig3^2) );
peak4 =  exp( -(((x - bx4).^2 + (y - by4).^2 + m4.^2)./sig4^2) );
peak5 =  exp( -(((x - bx5).^2 + (y - by5).^2 + m5.^2)./sig5^2) );

dx1 = -(2*(x - bx1)*peak1)/sig1.^2;
dy1 = -(2*(y - by1)*peak1)/sig1.^2;

dx2 = -(2*(x - bx2)*peak2)/sig2.^2;
dy2 = -(2*(y - by2)*peak2)/sig2.^2;

dx3 = -(2*(x - bx3)*peak3)/sig3.^2;
dy3 = -(2*(y - by3)*peak3)/sig3.^2;

dx4 = (2*(x - bx4)*peak4)/sig4.^2;
dy4 = (2*(y - by4)*peak4)/sig4.^2;

dx5 = -(2*(x - bx5)*peak5)/sig5.^2;
dy5 = -(2*(y - by5)*peak5)/sig5.^2;

dx = dx1 + dx2 + dx3 + dx4 + dx5;
dy = dy1 + dy2 + dy3 + dy4 + dy5;

gradient = [dx; dy];
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



function quiver3d(ax, sig1, sig2, sig3, sig4, sig5, m1, m2, m3, m4, m5, bx1, bx2, bx3, bx4, bx5, by1, by2, by3, by4, by5)

res = 20;

x = linspace(-3.5, 1.5, res);
y = linspace(-4, 4, res);

[X,Y] = meshgrid(x,y);
Z = feed_forward(X, Y, sig1, sig2, sig3, sig4, sig5, m1, m2, m3, m4, m5, bx1, bx2, bx3, bx4, bx5, by1, by2, by3, by4, by5);

x_gradients = zeros(res);
y_gradients = zeros(res);
z_gradients = zeros(res);
for i = 1:res
    for j = 1:res
        gradient = compute_gradient(x(i), y(j), sig1, sig2, sig3, sig4, sig5, m1, m2, m3, m4, m5, bx1, bx2, bx3, bx4, bx5, by1, by2, by3, by4, by5);
        x_gradients(i,j) = gradient(1);
        y_gradients(i,j) = gradient(2);
        z_gradients(i,j) = sqrt(gradient(1)^2 + gradient(2)^2);
    end
end

quiver(ax, X,Y,(-x_gradients./z_gradients)', (-y_gradients./z_gradients)', 0.6, 'Color', 'w')
% quiver3(ax, X,Y,Z,-x_gradients./z_gradients, -y_gradients./z_gradients, -z_gradients, 'Color', 'w')

end
