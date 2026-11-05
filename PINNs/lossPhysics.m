function [L_physics, residual]  = lossPhysics(x_physics, dxdt_physics, f_rhs, lambda)
% Penalize deviations from the ODE by minimizing the residual dxdt - f(x,lambda) where f is the RHS

lambda_rep = repmat(lambda,1,size(x_physics,2));

n_physics = size(x_physics, 2);
residual = dxdt_physics - f_rhs(x_physics,lambda_rep); 
% residual(isnan(residual)) = 0;
L_physics = 0.5.*(residual(:)'*residual(:))./n_physics;
end


