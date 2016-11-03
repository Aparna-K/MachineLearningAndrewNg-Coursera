function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

%  Add a bias unit
X = [ones(m, 1) X];
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% values of hidden layer
a2 = sigmoid(X * Theta1');
%  add a bias unit to hidden layer
a2 = [ones(m, 1) a2];

% values of the output layer
a3 = a2 * Theta2';


% find the linear regression values that are greater than 0, so that g(z) > 0.5
% z = x1*theta1 + x2*theta2 + ... + xn*thetan > 0  => g(z) > 0.5
% if multiple values are greater than 0, then take the maximum value
% This is because, greater the z value greater g(z) value => p(y =1) is greater

[values, indices] = max(a3'(:, 1:end));

p = indices';

% =========================================================================


end
