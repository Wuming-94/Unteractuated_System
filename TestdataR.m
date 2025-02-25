% 参数初始化
clear all;
tau = 1.8;  % Übergangszeit τ
    % 
    % input_data= readmatrix('input_data15.xlsx');
    % output_data= readmatrix('output_data15.xlsx');
    % for i=2680
    % q1_0 = input_data(i,1);  % q1(0) 在 0 到 0.2π 之间
    % q2_0 = input_data(i,2);  % q2(0) 在 0 到 0.2π 之间
    % q1_tau = input_data(i,3); % q1(τ) 在 0 到 2π 之间
    % q2_tau = input_data(i,4); % q2(τ) 在 0 到 2π 之间
    % 
    % a1=output_data(i,1);
    % a2=output_data(i,2);
    % a3=output_data(i,3);
    % a4=output_data(i,4);
    % a5=output_data(i,5);
    % a6=output_data(i,6);
    % end
    % display(output_data(i,:));

    % q1_0 = 0;  % q1(0) 在 0 到 0.2π 之间
    % q2_0 = 0.25 * pi;  % q2(0) 在 0 到 0.2π 之间
    % q1_tau = 0.5 * pi; % q1(τ) 在 0 到 2π 之间
    % q2_tau = 0;
    % 
    % q1_0 = 0;  % q1(0) 在 0 到 0.2π 之间
    % q2_0 = 0.4 * pi;  % q2(0) 在 0 到 0.2π 之间
    % q1_tau = 0.2 * pi; % q1(τ) 在 0 到 2π 之间
    % q2_tau = 0.2 * pi;

    q1_0 = 0.1*pi;  % q1(0) 在 0 到 0.2π 之间
    q2_0 = 0.4 * pi;  % q2(0) 在 0 到 0.2π 之间
    q1_tau = 0.2 * pi; % q1(τ) 在 0 到 2π 之间
    q2_tau = 0.2 * pi;
    % 象限1
    x_sample1 = [q1_0, q2_0, q1_tau, q2_tau];
    % 象限2
    x_sample2 = [q1_0, -q2_0, q1_tau, -q2_tau];
    % 象限3
    x_sample3 = [-q1_0, -q2_0, -q1_tau, -q2_tau];
    % 象限3
    x_sample4 = [-q1_0, q2_0, -q1_tau, q2_tau];
  
    % predicted_output = [a1,a2,a3,a4,a5,a6];
    % predicted_output = [75.23 -324.2  593.8 -596.6661  336.5708
    % -78.5032];%%  1%
   % predicted_outpu%_output);
    % 获取数据
    [x, y, ak_sol, v_vals] = TestBVP(tau,  x_sample2);
    [z] = invbvp(tau, x_sample4,-ak_sol);

    % qs2=[n1, n3];
    % func_robot_new(qs2, 2, 'NN_Robot');
    % qs1=[y(1, :)',y(3, :)'];
    % func_robot_new(qs1, 1 , 'Solver_Robot');

   
   
