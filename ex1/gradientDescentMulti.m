function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha


% Initialize some useful values
m = length(y); % number of training examples
n = size(X,2); % number of parameters (+1)
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    %temp_theta1 = theta(1) - (alpha / m)*sum((X*theta - y).*X(:,1));
    %temp_theta2 = theta(2) - (alpha / m)*sum((X*theta - y).*X(:,2));

    %theta(1) = temp_theta1;
    %theta(2) = temp_theta2;
    new_theta = theta;
    for j = 1:n
        sum_m = 0;
        % compute sum
        for i = 1:m
        sum_m += (X(i,:)*theta - y(i))*X(i,j);
        end
        
        new_theta_j = theta(j) - (alpha/m)*sum_m;
        new_theta(j) = new_theta_j;
    end
    theta = new_theta;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
