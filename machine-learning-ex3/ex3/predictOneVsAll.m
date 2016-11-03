function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% Find the value of x1*theta1 + x2*theta2+...+xn*thetan
% for all m inputs
% the size of product_theta_input is 'num_labels x m'
product_theta_input = (all_theta * X');

% find the linear regression values that are greater than 0, so that g(z) > 0.5
% z = x1*theta1 + x2*theta2 + ... + xn*thetan > 0  => g(z) > 0.5
% if multiple values are greater than 0, then take the maximum value
% This is because, greater the z value greater g(z) value => p(y =1) is greater

[max_vals, max_indices] = max(product_theta_input(:, 1:end));

p = max_indices';

% =========================================================================


end
