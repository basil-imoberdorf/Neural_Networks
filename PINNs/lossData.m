function L_data = lossData(x_predicted, x_measured)
% Penalize deviation from observed data
n_data = size(x_predicted, 2);
L_data = 0.5*sum((x_predicted(:) - x_measured(:)).^2)./n_data; % same as 0.5/n_data * (x_predicted(:) - x_measured(:))'*(x_predicted(:) - x_measured(:))
end
