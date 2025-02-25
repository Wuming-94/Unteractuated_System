% Parameter initialization input_data20
clear all;
tau = 1.8;              % Transition time τ
num_loop = 1000000;    % Number of iterations
j = 0;                  % 用于统计跳过的样本数量
max_delta = 0.25 * pi; % 预设每个关节允许的最大角度变化（示例值，可根据实际需求调整）
min_delta = 1e-5;
input_data = [];
output_data = [];
for i = 1:num_loop
    
% Loop to generate different input and output data
        %%% 选择一个参数配置（例如 Excel 14）
        q1_0 =0;                    % 初始状态 q1(0)
        q2_0 = round(-pi+2*pi*rand(), 4);          % 初始状态 q2(0)
        q1_tau = round(pi *rand(), 4);                   % 目标状态 q1(τ)
        q2_tau = round(-pi+2*pi*rand(), 4);        % 目标状态 q2(τ)
        
        % % 检查角度变化是否超过预设阈值
        % delta_q1 = (q1_tau - q1_0);
        % delta_q2 = abs(q2_tau - q2_0);
        % 
        % flag =(delta_q1 < min_delta);
        % if flag
        %     % disp('flag triggered, sample not adopted, next loop');
        %     j = j + 1;
        %     continue;  % 跳过当前样本
        % end
        
        % 调用求解器获得数据
        [x,y, ak_sol, v_vals] = solveBoundaryValueProblem(tau, q1_0, q2_0, q1_tau, q2_tau);
        
        % 检查ak_sol是否超出范围
        if any(abs(ak_sol) > 2000)
            j = j + 1;
            continue;  % 跳过当前样本
        end

        % 获取最近的警告信息
        [warnMsg, warnId] = lastwarn;
        
        % 检查是否触发了特定的警告
        if contains(warnMsg, '必须使用 2500 个以上的网格点，才能满足容差要求')
            disp('catch warning，next loop');
            lastwarn('');  % 清除警告信息
            j = j + 1;
            continue;    % 跳过当前样本
        elseif contains(warnMsg, '由于存在条件倒数为')
            disp('catch warning，next loop');
            lastwarn('');  % 清除警告信息
            j = j + 1;
            continue;    % 跳过当前样本
        end
        
        % 保存数据：注意 i - j 表示实际保存的样本序号
        input_data(i - j, :) = [q1_0, q2_0, q1_tau, q2_tau];
        output_data(i - j, :) = ak_sol;
        
        % 当样本数量达到预期（例如 1000 个）时结束循环
        if size(input_data, 1) >= 1000
            disp('1000 Samples ,end loop');
            break;
        end
        
    % catch ME
    %     % 如果有错误则跳过当前样本
    %     % disp(['Error encountered: ', ME.message, ', skipping sample.']);
    %     j = j + 1;
    %     continue;
    
end

% 将生成的数据写入 Excel 文件
writematrix(input_data, 'input_data23.xlsx');
writematrix(output_data, 'output_data23.xlsx');
