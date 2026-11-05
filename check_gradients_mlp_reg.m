function check_gradients_mlp_reg(Epsilon, Inputs, Targets, weights_cell, hiddenOutputs, gradients_cell, Num_Hidden_Layers, bias_filler, num_output_elements)
% This function computes nummerically the gradients of a multilayer perceptron, MLP, and compares them with the gradients obtained by backpropagation. If they match, the backpropagation algorithm works as intended.
%   Epsoilon:   Defines the magnitude of the weight perturbance. Usually is chosen to be: Epsoilon < 1e-6
%   
% First copy the weight matrices to manipulate them separately

% Difference criterion
% Difference < 1e-6     : Good
% Difference > 1e-3     : There is likely a problem


% First copy the weight matrices to manipulate them separately
numerical_weights_gradients_cell = gradients_cell;

for k = 1:Num_Hidden_Layers + 1 
    weights_cell_positive_perturbed = weights_cell;
    weights_cell_negative_perturbed = weights_cell;
    [dim_x, dim_y] = size(weights_cell{k});
    for i = 1:dim_x
        for j = 1:dim_y
            % Perturb each paramter one at a time and compute the loss
            weights_cell_positive_perturbed{k}(i,j) = weights_cell{k}(i,j) + Epsilon;
            weights_cell_negative_perturbed{k}(i,j) = weights_cell{k}(i,j) - Epsilon;

            [~, ~, Loss_positive_perturbed, ~] = feedForward_reg(Inputs, Targets, weights_cell_positive_perturbed, hiddenOutputs, Num_Hidden_Layers, bias_filler, num_output_elements);
            [~, ~, Loss_negative_perturbed, ~] = feedForward_reg(Inputs, Targets, weights_cell_negative_perturbed, hiddenOutputs, Num_Hidden_Layers, bias_filler, num_output_elements);
            
            % Compute the numerical approximation of the derivative for the corresponding weight
            numerical_weights_gradients_cell{k}(i,j) = (Loss_positive_perturbed - Loss_negative_perturbed)/(2*Epsilon);

            % Reset the weights for the next round to not carry over disturbances from previous tests
            weights_cell_positive_perturbed = weights_cell;
            weights_cell_negative_perturbed = weights_cell;
        end
    end

    gradient_differences = (numerical_weights_gradients_cell{k} - gradients_cell{k})

    numerator = norm(numerical_weights_gradients_cell{k} - gradients_cell{k});
    denominator = norm(numerical_weights_gradients_cell{k}) + norm(gradients_cell{k});
    relative_error = numerator / (denominator + 1e-12);  % Add small eps to avoid 0/0

    fprintf('\t Layer %d relative gradient error: %.2e  \n', k, relative_error);
end
fprintf('\n');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%