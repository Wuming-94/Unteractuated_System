function [x, y, v_vals] = invbvp(tau, x_sampletest, ak)
    % 参数初始化
    N = length(ak);  % 参数 ak 的长度
    ck = linspace(0, tau, N);  % 时间点 ck 均匀分布在 (0, τ)
    q1_0 = x_sampletest(1); 
    q2_0 = x_sampletest(2);
    % 定义初始猜测值
    solinit = bvpinit(linspace(0, tau, 10), [q1_0; 0; q2_0; 0]);  % 初始猜测值

    % 求解边界值问题
    sol = bvp4c(@odeFunc, @bcfun, solinit);

    % 提取解并显示结果
    x = linspace(0, tau, 100);
    y = deval(sol, x);  % 对解进行插值
    v_vals = arrayfun(@(t) guess(t, ak), x);  % 计算 v(t) 的值

    % 绘制结果
    figure('Name', 'Inverse Solver Output');
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
    title('Eingabefunktion v(t)');
    grid on;
    legend('v(t)');
    xlim([0, 2]);

    %% ODE 函数定义
    function dzdt = odeFunc(t, z)
        z1 = z(1); z2 = z(2); z3 = z(3); z4 = z(4);
        eta = 0.9;

        % 使用 ak 估计 v(t)
        v = guess(t, ak);  % 估计 v(t)

        dz1 = z2;
        dz2 = v;  % v 作为输入
        dz3 = z4;
        dz4 = -eta * sin(z3) * z2^2 - (1 + eta * cos(z3)) * v;

        dzdt = [dz1;
                dz2;
                dz3;
                dz4];
    end

    %% 边界条件定义
    function res = bcfun(ya, yb)
        % 边界条件
        res = [ya(1) - q1_0;  % q1(0) = q1_0
               ya(2);         % dq1(0) = 0
               ya(3) - q2_0;  % q2(0) = q2_0
               ya(4);         % dq2(0) = 0
               % yb(2);         % dq1(τ) = 0
               % yb(4)
               ];        % dq2(τ) = 0
    end

    %% 估计函数 v(t)（由 ak 表示）
    function v = guess(t, ak)
        v = 0;
        for k = 1:N
            % 估计 v(t) 的多项式形式
            v = v + ak(k) * abs(t - ck(k))^3;  % 多项式形式
        end
    end
end