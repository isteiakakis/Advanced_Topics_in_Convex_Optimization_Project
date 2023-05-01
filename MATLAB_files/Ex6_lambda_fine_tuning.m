clear; close all;

% Problem parameters
% m << n
m = 10;
n = 50;
s = floor(0.99*m/(2*log(n))); % number of non-zero elements of x_s,  m >= 2*s*log(n) << n

% The number of random generated problems to solve for each value of lambda
num_of_iters = 10;

% The values of the penalty parameter lambda
lambda_vals = 10.^(-8:2);

l1_error = zeros(length(lambda_vals), num_of_iters);
l2_error = zeros(length(lambda_vals), num_of_iters);

for k=1:num_of_iters

    % Generate matrix A
    A = randn(m, n);

    % Generate sparse vector x_s
    non_zero_indeces = randperm(n);
    non_zero_indeces = non_zero_indeces(1:s);

    x_s = zeros(n, 1);

    for i=non_zero_indeces
        x_s(i) = rand(1);
    end

    % Compute the vector b
    b = A*x_s;

    %% Question a

    for idx=1:length(lambda_vals)

        lambda = lambda_vals(idx);

        % Solve the problems via CVX

        % First problem

        cvx_begin quiet
            variable x(n)
            minimize( 1/2 * pow_pos(norm(A*x-b, 2), 2) + lambda * norm(x, 1) )
        cvx_end

        x_star1 = cvx_optpnt.x;


        % Second problem

        cvx_begin quiet
            variable x(n)
            minimize( 1/2 * pow_pos(norm(A*x-b, 2), 2) + lambda * pow_pos(norm(x, 2), 2) )
        cvx_end

        x_star2 = cvx_optpnt.x;

        % Store the error
        l1_error(idx, k) = norm(x_s - x_star1);
        l2_error(idx, k) = norm(x_s - x_star2);

    end

end

% Compute the mean values of the errors of the experiments
l1_error_mean = mean(l1_error, 2);
l2_error_mean = mean(l2_error, 2);

% Figure
figs(1) = figure;
loglog(lambda_vals, l1_error_mean, '.', 'MarkerSize', 15);
hold on;
loglog(lambda_vals, l2_error_mean, '.', 'MarkerSize', 15);
plot_setup({'L1 vs L2 regularized problem:', sprintf('Optimal point error per lambda value, $\\mathbf{(m, n)=(%d, %d)}$', m, n)}, ...
           '$\lambda$', '$\| \mathbf{x}^* - \mathbf{x}_s \|_2$', 'L1 regularized problem', 'L2 regularized problem');
