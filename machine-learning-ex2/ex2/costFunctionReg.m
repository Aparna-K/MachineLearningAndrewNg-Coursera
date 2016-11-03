function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
num_params = length(theta);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h_x = sigmoid(X * theta);

term1 = -1 * y .* log(h_x);

term2 = (1 .- y) .* log(1 .- h_x);

individual_cost = term1 - term2;

non_regularized_cost = sum(individual_cost)/m;

regularization_term = sum(theta(2:end) .^ 2)*lambda/(2*m); %theta(2: end) to skip theta0

J = non_regularized_cost + regularization_term;

difference_hyp_actual = h_x - y;
gradient_regularization_term = theta .* lambda ./ m;

gradient_regularization_term(1) = 0; %since we don't add a regularization term for j=0

for i=1:num_params
	grad(i) = sum(difference_hyp_actual .* X(:, i))/m + gradient_regularization_term(i);
end

% =============================================================

end
