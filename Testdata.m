% 参数初始化
clear all;
tau = 1.8;  % Übergangszeit τ
    % 
    % input_data= readmatrix('input_data21.xlsx');
    % i=5;
    % q1_0 = input_data(i,1);  % q1(0) 在 0 到 0.2π 之间
    % q2_0 = input_data(i,2);  % q2(0) 在 0 到 0.2π 之间
    % q1_tau = input_data(i,3); % q1(τ) 在 0 到 2π 之间
    % q2_tau = input_data(i,4); % q2(τ) 在 0 到 2π 之间
    %
    % q1_0 = 0;  % q1(0) 在 0 到 0.2π 之间
    % q2_0 = 0.25 * pi;  % q2(0) 在 0 到 0.2π 之间
    % q1_tau = 0.5 * pi; % q1(τ) 在 0 到 2π 之间
    % q2_tau = 0;
    % 
    q1_0 = 0;  % q1(0) 在 0 到 0.2π 之间
    q2_0 = 0.4 * pi;  % q2(0) 在 0 到 0.2π 之间
    q1_tau = 0.2 * pi; % q1(τ) 在 0 到 2π 之间
    q2_tau = 0.2 * pi;
    % % 
    % q1_0 = 0.1*pi;  % q1(0) 在 0 到 0.2π 之间
    % q2_0 = 0.4 * pi;  % q2(0) 在 0 到 0.2π 之间
    % q1_tau = 0.2 * pi; % q1(τ) 在 0 到 2π 之间
    % q2_tau = 0.2 * pi;
    %
    % x_sample1 = [];
    % % 象限1
    x_sample1 = [q1_0, q2_0, q1_tau, q2_tau];
    % % 象限2
    % x_sample2 = [q1_0, q2_0+pi, q1_tau, q2_tau+pi];
    % % 象限3
    % x_sample3 = [q1_0+pi, q2_0+pi, q1_tau+pi, q2_tau+pi];
    % % 象限3
    % x_sample4 = [q1_0+pi, q2_0, q1_tau+pi, q2_tau];


    % 获取数据
    [x, y, ak_sol, v_vals] = TestBVP(tau,  x_sample1);
   
   
