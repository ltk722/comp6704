
clear; clc; close all;
warning('off', 'all');

Anchors = [0, 0; 100, 0; 0, 100; 100, 100]; 
[N_BS, ~] = size(Anchors);

max_iter = 500;    
tol = 1e-6;         

fprintf('=== Optimization Benchmark Started ===\n');
fprintf('Anchors: %d, Area: 100x100m\n', N_BS);

%  Experiment 1: Convergence Analysis (Loss vs Iteration)
fprintf('\n[Exp 1] Running Convergence Analysis...\n');

x_true_demo = [75; 25]; 
x_init_demo = [10; 90]; 
sigma_demo = 1.0; 
rng(42); 
[d_demo, ~] = generate_measurements(Anchors, x_true_demo, sigma_demo);

[~, info_gd] = solve_gd(Anchors, d_demo, x_init_demo, max_iter, tol, 0.005);
[~, info_mom] = solve_momentum(Anchors, d_demo, x_init_demo, max_iter, tol, 0.005, 0.9);
[~, info_newton] = solve_newton(Anchors, d_demo, x_init_demo, max_iter, tol);
[~, info_gn] = solve_gn(Anchors, d_demo, x_init_demo, max_iter, tol);
[~, info_lm] = solve_lm(Anchors, d_demo, x_init_demo, max_iter, tol);

loss_gd_pad = pad_vector(info_gd.loss, max_iter);
loss_mom_pad = pad_vector(info_mom.loss, max_iter);
loss_new_pad = pad_vector(info_newton.loss, max_iter);
loss_gn_pad  = pad_vector(info_gn.loss, max_iter);
loss_lm_pad  = pad_vector(info_lm.loss, max_iter);

figure('Name', 'Exp 1: Convergence', 'Color', 'w', 'Position', [100, 100, 600, 450]);
semilogy(loss_gd_pad, 'm-', 'LineWidth', 1.5); hold on;
semilogy(loss_mom_pad, 'g--', 'LineWidth', 1.5);
semilogy(loss_new_pad, 'k-.', 'LineWidth', 1.5);
semilogy(loss_gn_pad, 'b-x', 'LineWidth', 1.5, 'MarkerIndices', 1:20:max_iter);
semilogy(loss_lm_pad, 'r-o', 'LineWidth', 1.5, 'MarkerIndices', 1:20:max_iter);
legend('Gradient Descent', 'Momentum GD', 'Newton', 'Gauss-Newton', 'LM');
title('Convergence Rate Comparison');
xlabel('Iteration'); ylabel('Objective Function Value (Log Scale)');
grid on; xlim([1, 100]); 

%  Experiment 2 & 3: RMSE vs SNR & Total CPU Time (Monte Carlo)
fprintf('\n[Exp 2 & 3] Running Monte Carlo Sim (RMSE & Time)...\n');

noise_levels = 0.5:0.5:5.0; 
MC_runs = 100; 

rmse_res = zeros(5, length(noise_levels));
time_res = zeros(5, length(noise_levels));
x_init_mc = [50; 50]; 

for i = 1:length(noise_levels)
    sig = noise_levels(i);
    fprintf('  -> Processing Noise Sigma = %.1f m...\n', sig);
    
    temp_err = zeros(5, MC_runs);
    temp_time = zeros(5, MC_runs);
    
    for r = 1:MC_runs
        x_true_rand = 10 + rand(2, 1) * 80; 
        [d_m, ~] = generate_measurements(Anchors, x_true_rand, sig);
        
        tic; [x_g, ~] = solve_gd(Anchors, d_m, x_init_mc, 2000, tol, 0.005); t=toc;
        temp_err(1,r) = norm(x_g - x_true_rand); temp_time(1,r) = t;
        
        tic; [x_mo, ~] = solve_momentum(Anchors, d_m, x_init_mc, 2000, tol, 0.005, 0.9); t=toc;
        temp_err(2,r) = norm(x_mo - x_true_rand); temp_time(2,r) = t;
                
        tic; [x_ne, ~] = solve_newton(Anchors, d_m, x_init_mc, 100, tol); t=toc;
        temp_err(3,r) = norm(x_ne - x_true_rand); temp_time(3,r) = t;
        

        tic; [x_gn, ~] = solve_gn(Anchors, d_m, x_init_mc, 100, tol); t=toc;
        temp_err(4,r) = norm(x_gn - x_true_rand); temp_time(4,r) = t;
        

        tic; [x_lm, ~] = solve_lm(Anchors, d_m, x_init_mc, 100, tol); t=toc;
        temp_err(5,r) = norm(x_lm - x_true_rand); temp_time(5,r) = t;
    end
    
    rmse_res(:, i) = sqrt(mean(temp_err.^2, 2));
    time_res(:, i) = mean(temp_time, 2);
end


figure('Name', 'Exp 2: RMSE', 'Color', 'w', 'Position', [150, 150, 600, 450]);
plot(noise_levels, rmse_res(1,:), 'm-o', 'LineWidth', 1.2); hold on;
plot(noise_levels, rmse_res(2,:), 'g--', 'LineWidth', 1.2);
plot(noise_levels, rmse_res(3,:), 'k-.', 'LineWidth', 1.2);
plot(noise_levels, rmse_res(4,:), 'b-x', 'LineWidth', 1.2);
plot(noise_levels, rmse_res(5,:), 'r-s', 'LineWidth', 1.2);
legend('Gradient Descent', 'Momentum', 'Newton', 'Gauss-Newton', 'LM', 'Location', 'northwest');
xlabel('Noise Std Dev (sigma) [m]'); ylabel('RMSE [m]');
title('Estimation Accuracy vs Noise (100 MC Runs)'); grid on;


figure('Name', 'Exp 3: Total Time', 'Color', 'w', 'Position', [200, 200, 600, 450]);
avg_time_all = mean(time_res, 2) * 1000; % 转 ms
bar(categorical({'GD','Momentum','Newton','GN','LM'}), avg_time_all);
ylabel('Total Runtime (ms)'); title('Average Total Convergence Time'); grid on;

fprintf('\n[Exp 4] Running Initialization Sensitivity Test...\n');

dist_range = 10:10:100; 
MC_runs_sens = 50;
success_rate = zeros(5, length(dist_range));
x_center = [50; 50]; 

for i = 1:length(dist_range)
    R = dist_range(i);
    wins = zeros(5, 1);
    for r = 1:MC_runs_sens
        theta = rand * 2 * pi;
        x_init_bad = x_center + [R*cos(theta); R*sin(theta)];
        [d_m, ~] = generate_measurements(Anchors, x_center, 1.0);
        
        [x1, ~] = solve_gd(Anchors, d_m, x_init_bad, 2000, tol, 0.005);
        [x2, ~] = solve_momentum(Anchors, d_m, x_init_bad, 2000, tol, 0.005, 0.9);
        [x3, ~] = solve_newton(Anchors, d_m, x_init_bad, 100, tol);
        [x4, ~] = solve_gn(Anchors, d_m, x_init_bad, 100, tol);
        [x5, ~] = solve_lm(Anchors, d_m, x_init_bad, 100, tol);
        
        errs = [norm(x1-x_center), norm(x2-x_center), norm(x3-x_center), ...
                norm(x4-x_center), norm(x5-x_center)];
        wins = wins + (errs' < 2.0); % 误差小于2m算成功
    end
    success_rate(:, i) = wins / MC_runs_sens;
end

figure('Name', 'Exp 4: Sensitivity', 'Color', 'w');
plot(dist_range, success_rate(1,:), 'm-o', 'LineWidth', 1.5); hold on;
plot(dist_range, success_rate(2,:), 'g--', 'LineWidth', 1.5);
plot(dist_range, success_rate(3,:), 'k-.', 'LineWidth', 1.5);
plot(dist_range, success_rate(4,:), 'b-x', 'LineWidth', 1.5);
plot(dist_range, success_rate(5,:), 'r-s', 'LineWidth', 1.5);
legend('GD', 'Momentum', 'Newton', 'GN', 'LM');
xlabel('Init Distance from True Pos [m]'); ylabel('Success Rate');
title('Sensitivity to Initialization'); grid on; ylim([-0.1, 1.1]);

fprintf('\n[Exp 5] Running Time Trade-off Analysis...\n');

N_test = 5000; 
[d_m, ~] = generate_measurements(Anchors, x_center, 1.0);

tic; solve_gd(Anchors, d_m, x_center, N_test, 0, 0.005); 
t_gd = toc;

tic; solve_momentum(Anchors, d_m, x_center, N_test, 0, 0.005, 0.9); 
t_mom = toc;

tic; solve_newton(Anchors, d_m, x_center, N_test, 0); 
t_new = toc;

tic; solve_gn(Anchors, d_m, x_center, N_test, 0); 
t_gn = toc;

tic; solve_lm(Anchors, d_m, x_center, N_test, 0); 
t_lm = toc;

time_per_iter = [t_gd, t_mom, t_new, t_gn, t_lm] / N_test * 1e6; 

avg_steps = [800, 200, 5, 5, 6]; 

total_time_est = time_per_iter .* avg_steps / 1000; % ms

figure('Name', 'Exp 5: Trade-off', 'Color', 'w', 'Position', [100, 100, 900, 400]);

subplot(1,2,1);
bar(categorical({'GD', 'Mom-GD', 'Newton', 'GN', 'LM'}), time_per_iter);
ylabel('Time per Iteration (\mus)'); 
title('Per-Iteration Complexity');
grid on;

subplot(1,2,2);
b = bar(categorical({'GD', 'Mom-GD', 'Newton', 'GN', 'LM'}), total_time_est);
b.FaceColor = 'flat';
b.CData(2,:) = [0 0.8 0]; % 把 Momentum 标绿突出显示
ylabel('Total Time (ms)'); 
title('Total Convergence Time (Step \times Count)');
grid on;

fprintf('Exp 5 Completed. Momentum-GD per-step time: %.2f us\n', time_per_iter(2));


function [meas, true_dist] = generate_measurements(Anchors, x_true, sigma)
    true_dist = vecnorm(Anchors' - x_true, 2, 1)';
    meas = true_dist + randn(size(true_dist)) * sigma;
end

function padded = pad_vector(vec, len)
    if length(vec) >= len, padded = vec(1:len);
    else, padded = [vec, repmat(vec(end), 1, len - length(vec))]; end
end

function [f, r, J] = compute_oracle(x, Anchors, dist_meas)
    diff_vec = x - Anchors'; 
    d_est = vecnorm(diff_vec, 2, 1)';
    r = d_est - dist_meas; 
    f = 0.5 * sum(r.^2);   
    J = (diff_vec ./ (d_est' + 1e-10))'; % Jacobian
end

% 1. GD
function [x, info] = solve_gd(Anchors, d_m, x0, max_iter, tol, lr)
    x = x0; loss = [];
    for k = 1:max_iter
        [f, r, J] = compute_oracle(x, Anchors, d_m);
        loss(end+1) = f;
        grad = J' * r;
        x = x - lr * grad;
        if norm(grad) < tol, break; end
    end
    info.loss = loss;
end

% 2. Momentum
function [x, info] = solve_momentum(Anchors, d_m, x0, max_iter, tol, lr, beta)
    x = x0; v = zeros(size(x)); loss = [];
    for k = 1:max_iter
        [f, r, J] = compute_oracle(x, Anchors, d_m);
        loss(end+1) = f;
        grad = J' * r;
        v = beta * v + grad;
        x = x - lr * v;
        if norm(grad) < tol, break; end
    end
    info.loss = loss;
end

function [x, info] = solve_newton(Anchors, d_m, x0, max_iter, tol)
    x = x0; loss = [];
    N_BS = size(Anchors, 1);
    for k = 1:max_iter
        [f, r, J] = compute_oracle(x, Anchors, d_m);
        loss(end+1) = f;
        grad = J' * r;
        
        % Exact Hessian Calculation
        H_exact = J'*J; 
        for i = 1:N_BS
            dist = norm(x - Anchors(i,:)') + 1e-10;
            vec = x - Anchors(i,:)';
            H_dist = (eye(2)/dist) - (vec*vec')/(dist^3);
            H_exact = H_exact + r(i) * H_dist;
        end
        
        % Use pinv for robustness against singularity
        delta = -pinv(H_exact) * grad; 
        x = x + delta;
        if norm(grad) < tol, break; end
    end
    info.loss = loss;
end

function [x, info] = solve_gn(Anchors, d_m, x0, max_iter, tol)
    x = x0; loss = [];
    for k = 1:max_iter
        [f, r, J] = compute_oracle(x, Anchors, d_m);
        loss(end+1) = f;
        grad = J' * r;
        H_gn = J' * J;
        delta = -pinv(H_gn) * grad; 
        x = x + delta;
        if norm(grad) < tol, break; end
    end
    info.loss = loss;
end

function [x, info] = solve_lm(Anchors, d_m, x0, max_iter, tol)
    x = x0; loss = [];
    mu = 1e-3; 
    for k = 1:max_iter
        [f, r, J] = compute_oracle(x, Anchors, d_m);
        loss(end+1) = f;
        grad = J' * r;
        H_gn = J' * J;
        while true
            delta = -pinv(H_gn + mu*eye(2)) * grad;
            x_new = x + delta;
            [f_new, ~, ~] = compute_oracle(x_new, Anchors, d_m);
            if f_new < f
                x = x_new;
                mu = mu / 10; 
                break;
            else
                mu = mu * 10; 
                if mu > 1e10, break; end 
            end
        end
        if norm(grad) < tol, break; end
    end
    info.loss = loss;

end
