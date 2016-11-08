function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

X = [ones(m, 1) X]; % Add bias to input
a2 = sigmoid(X * Theta1');
a3 = sigmoid([ones(m, 1) a2] * Theta2'); % Add bias to hidden layer and compute a3
% h_x = arrayfun(@binaryValue, a3); % find the prediction of hypotheses
h_x=a3;

% Vectorize y 
vectorized_y = arrayfun(@(o) vectorizeOutput(o, num_labels), y, ...
						'UniformOutput', false);

vectorized_y = cell2mat(vectorized_y);
% For loop implementation

% temp1 = zeros(size(y));
% for i = 1:m
% 	temp2 = zeros(num_labels, 1);
% 	for k = 1:num_labels
% 		term1 = -1 * vectorized_y(i, k) * log(h_x(i, k));
% 		term2 = (1 - vectorized_y(i, k)) * log(1 - h_x(i, k));
% 		temp2(k) = term1 - term2;
% 	end
% 	temp1(i) = sum(temp2);
% end

% vectorized implementation

term1 = vectorized_y .* log(h_x) * -1;
term2 = ((1 - vectorized_y) .* log(1 - h_x));
temp1 = term1 - term2;
temp2 = sum(temp1');

J = sum(temp2)/m;

% For-loop implementation

% reg_term1 = 0;
% reg_term2 = 0;

% s_1 = size(Theta1,2);
% s_2 = size(Theta1, 1);

% for i = 2:s_1
% 	for j = 1:s_2
% 		reg_term1 = reg_term1 + Theta1(j,i)^2;
% 	end
% end

% s_1 = size(Theta2, 2);
% s_2 = size(Theta2, 1);

% for i = 2:s_1
% 	for j = 1:s_2
% 		reg_term2 = reg_term2 + Theta2(j, i)^2;
% 	end
% end


% Vectorized implementation

reg_term1 = sum(sum(Theta1(:, 2:end) .^ 2)); % no regularization on the bias parameter
reg_term2 = sum(sum(Theta2(:, 2:end) .^ 2));

regularization_term = (lambda/(2*m)) * (reg_term1 + reg_term2);

J = J + regularization_term;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

delta_2 = zeros(m, size(Theta1, 1));
delta_3 = zeros(m, size(Theta2, 1));

D_1 = zeros(size(Theta1));
D_2 = zeros(size(Theta2));

%  Implementation with a for loop

% for i = 1:m
% 	a_1 = X(i, :);
% 	a_2 = sigmoid(a_1 * Theta1');
% 	a_2 = [ones(1, 1) a_2]; % add a bias
% 	a_3 = sigmoid(a_2 * Theta2');

% 	delta_3(i, :) = a_3 - vectorized_y(i, :);

% 	sigmoid_gradient_a_2 = sigmoidGradient(a_1 * Theta1'); % g'(z) = g(z) * [1 - g(z)]

% 	delta_2(i, :) = (delta_3(i, :) * Theta2(:, 2:end)) .* sigmoid_gradient_a_2; %a_2_no_bias .* (1 - a_2_no_bias);

% 	D_1 = D_1 + (a_1' * delta_2(i, :))';
% 	D_2 = D_2 + (a_2' * delta_3(i, :))';
% end

% Vectoized Implementation of back prop

delta_3 = a3 - vectorized_y;
sigmoid_gradient_a_2 = sigmoidGradient(X * Theta1');
delta_2 = delta_3 * Theta2(:, 2:end) .* sigmoid_gradient_a_2;

D_1 = (X' * delta_2)';
D_2 = ([ones(m, 1) a2]' * delta_3)';

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:, 1) = (1/m) * D_1(:, 1);
Theta1_grad(:, 2:end) = (1/m)*(D_1(:, 2:end) + (lambda * Theta1(:, 2:end)));

Theta2_grad(:, 1) = (1/m) * D_2(:, 1);
Theta2_grad(:, 2:end) = (1/m)*(D_2(:, 2:end) + (lambda * Theta2(:, 2:end)));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
