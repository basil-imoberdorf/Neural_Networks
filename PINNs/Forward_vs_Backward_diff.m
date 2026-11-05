clear
close all

rng(420)

y_true = [1;2];
x = randn(3,1);

W1 = randn(2,3);
W2 = randn(2);
W3 = randn(2);

z1 = W1*x;
a1 = tanh(z1);

z2 = W2*a1;
a2 = tanh(z2);

z3 = W3*a2;
y = z3;

% Loss
L = 0.5*sum((y - y_true).^2);

% Construct all Jacobians
dLdy = (y - y_true)';  % same as dLdz3
dz3da2 = W3;  % dz3da2
da2dz2_diag = 1 - tanh(z2).^2;
da2dz2 = [da2dz2_diag(1),0;0,da2dz2_diag(2)];

dz2da1 = W2;
da1dz1_diag = 1 - tanh(z1).^2;
da1dz1 = [da1dz1_diag(1),0;0,da1dz1_diag(2)];

dz1dx = W1;

% Forward derivative if evaluated from right to left: Note we use the Jacobian notation not the the column gradient. So dLdy is a row vector here
dLdx_forward = dLdy*(dz3da2*(da2dz2*(dz2da1*(da1dz1*dz1dx)))) % Leads to intermediate matrices along the eval path
dLdx_backward = ((((dLdy*dz3da2)*da2dz2)*dz2da1)*da1dz1)*dz1dx % Leads to only vectors 

% For the backward derivative we can use a different notation that implements the left to right eval of the expression above

dLdx_backprop_grad = dz1dx'*(dz2da1'*((dz3da2'*dLdy') .* da2dz2_diag) .* da1dz1_diag) % Leads to vectors along the eval path

% This is a recursive procedure: Start with the seed dLdy. Then multiply with the transposed weights matrix on the left and do with the result a element wise multiplication with the derivatives of the activations
W = {W1, W2, W3};
z = {z1, z2, z3};
act = {x, a1, a2};

d_layer = cell(3,1);
d_layer{end} = dLdy'; % place the seed

for i = 3:-1:1+1
    
    d_layer{i-1} = W{i}'*d_layer{i} .* (1 - tanh(z{i-1}).^2);

end

dLdx_backprop_grad_loop = W1' * d_layer{1}



% Here we are back using the gradient based notation so we have to transpose and to get the same as before we also need to invert the order of th ematrix multiplication
% Furthermore we dont need the diagonal matrix but a colun vector for the derivative of the activation function. 
% To have the same form as before we would need to transpose also the final result






%%% Numeric Derivative
eps = 1e-6;
dLdx_numdiff = zeros(1,length(x));

for i = 1:length(x)
    x_pos = x;
    x_neg = x;

    x_pos(i) = x(i) + eps;
    x_neg(i) = x(i) - eps;

    L_pos = forwardPass_WithLoss(x_pos, W1, W2, W3, y_true);
    L_neg = forwardPass_WithLoss(x_neg, W1, W2, W3, y_true);

    dLdx_numdiff(i) = (L_pos - L_neg)./(2*eps);
    
end
dLdx_numdiff




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function L = forwardPass_WithLoss(x,W1,W2,W3, y_true)

z1 = W1*x;
a1 = tanh(z1);

z2 = W2*a1;
a2 = tanh(z2);

z3 = W3*a2;
y = z3;

L = 0.5*sum((y - y_true).^2);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






% syms w11_1 w12_1 w21_1 w22_1 x1 x2
% 
% 
% x = [x1;x2];
% W1 = [w11_1, w12_1; w21_1, w22_1];
% y = W1*x;
% 
% 
% jacobian(y,x)












