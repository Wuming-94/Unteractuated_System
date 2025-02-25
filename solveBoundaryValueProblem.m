function [ x,y, ak_sol, v_vals] = solveBoundaryValueProblem(tau, q1_0, q2_0, q1_tau, q2_tau)
    % Parameter initialisation
    N=6;
    ck = linspace(0, tau, N);  % Time points ck evenly distributed in (0, τ)

    % Initial guess for ak
    ak_init = ones(N, 1);  % Initialize ak as an array of ones

    % Define the initial solution for the ODE
    solinit = bvpinit(linspace(0, tau, 10), [q1_0; 0; q2_0; 0], ak_init);  % Initial solution for z and ak
   
    % Solve the boundary value problem
    sol = bvp4c(@odeFunc, @bcfun, solinit);

    x = linspace(0, tau, 100);
    y = deval(sol, x);  % Interpolation of the solution
    ak_sol = sol.parameters;
    v_vals = arrayfun(@(t) guess(t, ak_sol), x); 
    %% ODE Function Definition
    function dydx = odeFunc(t, z, ak)
        z1 = z(1); z2 = z(2); z3 = z(3); z4 = z(4);
        eta = 0.9;

        % Use ak to estimate v(t)
        v = guess(t, ak);  % Estimate v(t) in polynomial form
        
        dz1 = z2;
        dz2 = v;  % v as new input
        dz3 = z4;
        dz4 = -eta * sin(z3) * z2^2 - (1 + eta * cos(z3)) * v;
        
        dydx = [dz1;
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
