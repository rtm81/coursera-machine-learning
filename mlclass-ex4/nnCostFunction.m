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
%
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add ones to the X data matrix
X = [ones(m, 1) X];

Z2 = X*Theta1';
% size(Z2) % 5000 x 25
A2 = sigmoid(Z2);
A2 = [ones(size(A2, 1), 1) A2];

Z3 = A2*Theta2';
A3 = sigmoid(Z3);

% [maxvalue, p] = max(A3, [], 2);

%size(A3) % 5000 x 10
%size(y) % 5000 x 1
hypo = A3;

y_nn = zeros(m, num_labels); % 5000 x 10
for i = 1:m
	y_nn(i, y(i)) = 1;
end

costTemp = (-y_nn .* log(hypo) - (1 - y_nn) .* log(1 - hypo));

% size(Theta1) % 25 x 401
% size(Theta2) % 10 x 26

reg  = (lambda / (2 * m)) * (sum(sum(Theta1(:,2:size(Theta1, 2)).^2)) + sum(sum(Theta2(:,2:size(Theta2, 2)).^2)));
J = sum(sum(costTemp)) / m + reg;

for t = 1:m
	delta_3 = A3(t,:) - y_nn(t,:);
	%size(delta_3) % 1 x 10
	
	Z2temp = Z2(t,:);
	Z2temp = [ones(size(Z2temp, 1), 1) Z2temp];
	delta_2 = delta_3 * Theta2 .* sigmoidGradient(Z2temp);
	%size(delta_2) % 1 x 26
	%delta_2 = delta_2(2:end,:);
	%size(Theta2_grad) % 10 x 26
	Theta2_grad = Theta2_grad .+ delta_3' * A2(t,:);
	
	%size(Theta1_grad) % 25 x 401
	delta_2 = delta_2(:,2:end);
	%size(delta_2) %  1 x 26
	%size(X(t,:)') % 401 x 1
	
	Theta1_grad = Theta1_grad .+ (delta_2' * X(t,:));
end

Theta2_grad = Theta2_grad ./ m;
Theta1_grad = Theta1_grad ./ m;






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
