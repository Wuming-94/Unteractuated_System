function [x, y, ak_sol, v_vals] = TestBVP(tau, x_sampletest)
    % Parameter initialisation
    N=6;
    ck = linspace(0, tau, N);  % Time points ck evenly distributed in (0, τ)
    q1_0 = x_sampletest(1); 
    q2_0 = x_sampletest(2); 
    q1_tau = x_sampletest(3); 
    q2_tau = x_sampletest(4); 
    % Initial guess for ak
    ak_init = ones(N, 1);  % Initialize ak as an array of ones
    % ak_init =[100;100;100;100;100;100];
    % Define the initial solution for the ODE
    solinit = bvpinit(linspace(0, tau, 10), [q1_0; 0; q2_0; 0], ak_init);  % Initial solution for z and ak

    % Solve the boundary value problem
    sol = bvp4c(@odeFunc, @bcfun, solinit);

    % Extract solution and display results
    x = linspace(0, tau, 100);
    y = deval(sol, x);  % Interpolation of the solution
    ak_sol = sol.parameters;  % Solved values for ak
    v_vals = arrayfun(@(t) guess(t, ak_sol), x);  % Calculate the solved values for v(t)
    
    % Display q1 and q2 as well as their derivatives
    figure('Name', 'solver Output');
    subplot(3, 1, 1);
    plot(x, y(1, :), '--', x, y(3, :));
    legend('z1', 'z3');
    xlabel('Zeit (s)');
    ylabel('θ (rad)');
    title('Gelenkwinkel (t)');
    xlim([0, 2]); 

    subplot(3, 1, 2);
    plot(x, y(2, :), '--', x, y(4, :));
    legend('z2', 'z4');
    xlabel('Zeit (s)');
    ylabel('ω(rad/s)');
    title('Gelenkwinkelgeschwindigkeit (t)');
    xlim([0, 2]); 

    subplot(3, 1, 3);
    plot(x, v_vals, 'LineWidth', 2);
    xlabel('Zeit (s)');
    ylabel('v(t)');
    title('Steuergröße v(t)');
    grid on;
    legend('v(t)');
    xlim([0, 2]); 
    % 
    % qs=[y(1, :)',y(3, :)'];
    % func_robot_new(qs, 1 , 'Solver_Robot');


    %% ODE Function Definition
   function dzdt = odeFunc(t, z, ak)
        z1 = z(1); z2 = z(2); z3 = z(3); z4 = z(4);
        eta = 0.9;

        % Use ak to estimate v(t)
        v = guess(t, ak);  % Estimate v(t) in polynomial form
        
        dz1 = z2;
        dz2 = v;  % v as new input
        dz3 = z4;
        dz4 = -eta * sin(z3) * z2^2 - (1 + eta * cos(z3)) * v;
        
        dzdt = [dz1;
                dz2;
                dz3;
                dz4];
    end

    %% Definition of Boundary Conditions
    function res = bcfun(ya, yb, ak)
        % Boundary conditions include v(0) = 0 and v(tau) = 0
        res = [ya(1) - q1_0;         % q1(0) = q1_0
               ya(2);                % dq1(0) = 0
               ya(3) - q2_0;        % q2(0) = q2_0
               ya(4);                % dq2(0) = 0
               yb(1) - q1_tau;      % q1(τ) = q1_tau
               yb(2);                % dq1(τ) = 0
               yb(3) - q2_tau;      % q2(τ) = q2_tau
               yb(4);                % dq2(τ) = 0
               guess(0, ak);        % v(0) = 0
               guess(tau, ak)];     % v(tau) = 0
    end

    %% Estimation Function v(t) (represented by ak)
    function v = guess(t, ak)
        N = length(ak);
        v = 0;
        ck = linspace(0, tau, N);  % Time points ck evenly distributed

        for k = 1:N
            % Estimate v(t) in polynomial form and ensure v(0) = 0 and v(tau) = 0
            v = v + ak(k) * abs(t - ck(k))^3;  % Polynomial form for estimation
        end
    end
end

