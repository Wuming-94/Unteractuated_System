% Parameter initialization
clear all;
tau = 1.8;  % Transition time τ

input_data = readmatrix('input_data21.xlsx');
output_data = readmatrix('output_data21.xlsx');

% input_data = readmatrix('data21.xlsx', 'Sheet', 'Sheet1', 'Range', 'A3:C5290');
% output_data = readmatrix('data21.xlsx', 'Sheet', 'Sheet1', 'Range', 'C3:I5290');

% ==== Z-Score 标准化 ====
mu_input = mean(input_data, 1);     % 输入数据均值
sigma_input = std(input_data, 0, 1); % 输入数据标准差
mu_output = mean(output_data, 1);    % 输出数据均值
sigma_output = std(output_data, 0, 1); % 输出数据标准差

input_data_norm = (input_data - mu_input) ./ sigma_input; % 输入标准化
output_data_norm = (output_data - mu_output) ./ sigma_output; % 输出标准化

% Create neural network
hidden_size1 = 32;  % 原为10
hidden_size2 = 16;  % 原为8
hidden_size3 = 8; 
hidden_size4 = 5; 
input_size = size(input_data_norm, 2); % 输入层节点数
output_size = size(output_data_norm, 2); % 输出层节点数

% Initialize weights
rng(42);
W1 = randn(input_size, hidden_size1)* sqrt(2/input_size);
b1 = zeros(1, hidden_size1);
W2 = randn(hidden_size1, hidden_size2)* sqrt(2/hidden_size1);
b2 = zeros(1, hidden_size2);
W3 = randn(hidden_size2, hidden_size3) * sqrt(2/hidden_size2);
b3 = zeros(1, hidden_size3);
W4 = randn(hidden_size3, hidden_size4) * sqrt(2/hidden_size3);  % 原W4改为第四隐藏层
b4 = zeros(1, hidden_size4);
W5 = randn(hidden_size4, output_size) * sqrt(2/hidden_size4);   % 新增输出层权重
b5 = zeros(1, output_size);

% Train network with Mini-batch Gradient Descent
alpha0 = 0.01; % Learning rate
decay_steps = 20;
decay_rate = 0.9;

% ==== L2正则化参数 ==== 【修改点3：添加正则化】
lambda = 0.0295;

epochs = 1000;
batch_size = 32; % 定义批量大小

n_total = size(input_data_norm, 1);
shuffled_idx = randperm(n_total);
train_ratio = 0.7;
n_train = round(n_total * train_ratio);

% 训练数据
train_input = input_data_norm(shuffled_idx(1:n_train), :);
train_output = output_data_norm(shuffled_idx(1:n_train), :);

% 验证数据
val_input = input_data_norm(shuffled_idx(n_train+1:end), :);
val_output = output_data_norm(shuffled_idx(n_train+1:end), :);

% 初始化记录器
train_loss_history = zeros(epochs, 1);
val_loss_history = zeros(epochs, 1);

% 创建实时绘图窗口
figure('Name','Training Progress');
h = axes;
hold on;
xlabel('Epoch');
ylabel('Loss');
title('Training vs Validation Loss');
grid on;

% % 激活函数 (ReLU)
% relu = @(x) max(0, x);
% relu_derivative = @(x) double(x > 0);

relu = @(x) max(0.1*x, x); %Leaky ReLU 
relu_derivative = @(x) (x > 0) + 0.1*(x <= 0);

% relu = @(x) 1 ./ (1 + exp(-x));
% relu_derivative = @(x) x .* (1 - x); % Derivative of the sigmoid function


% Early stopping 参数
patience = 15;
best_val_loss = Inf;
wait_count = 0;
min_delta = 0.005;

% 最佳模型保存
best_model = struct('W1', W1, 'b1', b1, 'W2', W2, 'b2', b2, 'W3', W3, 'b3', b3, 'W4', W4, 'b4', b4, 'W5', W5, 'b5', b5);
for epoch = 1:epochs
     % ==== 动态学习率计算 ====
    alpha = alpha0 * (decay_rate)^(floor(epoch/decay_steps));


    % ==== Mini-batch 训练 ====
    epoch_loss = 0;
    num_batches = ceil(n_train / batch_size);
    shuffled_indices = randperm(n_train);
    % alpha = alpha / (1 + decay_rate * epoch);
    for batch = 1:num_batches
        % 获取当前批次的索引
        start_idx = (batch-1) * batch_size + 1;
        end_idx = min(batch * batch_size, n_train);
        batch_idx = shuffled_indices(start_idx:end_idx);
        
        % 提取批次数据
        % X_batch = train_input(batch_idx, :)+ 0.01*randn(size(train_input(batch_idx, :)));
        X_batch = train_input(batch_idx, :);
        Y_batch = train_output(batch_idx, :);
        
        % 计算批量梯度
        [dW1, db1, dW2, db2, dW3, db3, dW4, db4,dW5, db5] = MiniBatchGradient(X_batch, Y_batch, W1, b1, W2, b2, W3, b3, W4, b4,W5, b5, relu, relu_derivative);
        
        % 更新权重和偏置
        W1 = W1 + alpha * (dW1 - lambda*W1);  % 添加正则化项
        b1 = b1 + alpha * db1;
        W2 = W2 + alpha * (dW2 - lambda*W2);
        b2 = b2 + alpha * db2;
        W3 = W3 + alpha * (dW3 - lambda*W3);
        b3 = b3 + alpha * db3;
        W4 = W4 + alpha * (dW4 - lambda*W4);
        b4 = b4 + alpha * db4;
        W5 = W5 + alpha * (dW5 - lambda*W5);  % 新增
        b5 = b5 + alpha * db5;     
        
        % 计算批次损失
        pred_batch = forward_pass(X_batch, W1, b1, W2, b2, W3, b3, W4, b4,W5, b5, relu);
        epoch_loss = epoch_loss + mean((pred_batch - Y_batch).^2, 'all');
    end
    
    % 记录训练损失
    train_loss_history(epoch) = epoch_loss / num_batches;
    
    % ==== 验证阶段 ====
    val_loss = 0;
    for j = 1:size(val_input, 1)
        x_val = val_input(j, :);
        y_val = val_output(j, :);
        pred_val = forward_pass(x_val, W1, b1, W2, b2, W3, b3, W4, b4,W5, b5, relu);
        val_loss = val_loss + mean((pred_val - y_val).^2);
    end
    current_val_loss = val_loss / size(val_input, 1);
    val_loss_history(epoch) = current_val_loss;
    
    % 更新早停条件
    if current_val_loss < (best_val_loss - min_delta)
        best_val_loss = current_val_loss;
        wait_count = 0;
        best_model.W1 = W1;
        best_model.b1 = b1;
        best_model.W2 = W2;
        best_model.b2 = b2;
        best_model.W3 = W3;
        best_model.b3 = b3;
        best_model.W4 = W4;
        best_model.b4 = b4;
        best_model.W5 = W5;
        best_model.b5 = b5;
        fprintf('>> 保存最佳模型 (loss: %.4f)\n', best_val_loss);
    else
        wait_count = wait_count + 1;
    end
    
    if wait_count >= patience
        fprintf('\n=== 早停于 epoch %d ===\n', epoch);
        break;
    end
    
    % 绘图和打印
    plot(h, 1:epoch, train_loss_history(1:epoch), 'b-', 1:epoch, val_loss_history(1:epoch), 'r-');
    legend(h, 'Training Loss', 'Validation Loss');
    drawnow;
    fprintf('Epoch %d/%d - Train Loss: %.4f - Val Loss: %.4f\n', epoch, epochs, train_loss_history(epoch), val_loss_history(epoch));
end

% 加载最佳模型
W1 = best_model.W1;
b1 = best_model.b1;
W2 = best_model.W2;
b2 = best_model.b2;
W3 = best_model.W3;
b3 = best_model.b3;
W4 = best_model.W4;
b4 = best_model.b4;
W5 = best_model.W5;
b5 = best_model.b5;

% ==== 测试阶段 (含反归一化) ====
% input_data2 = readmatrix('input_data21.xlsx');
% output_data2 = readmatrix('output_data21.xlsx');
% i = 5; % 测试样本索引
% 
% % 标准化测试输入
% x_sample2 = input_data2(i, 1:4);
% x_sample1 = input_data2(i, 1:4);
x_sample2 = [0,0.4*pi, 0.2*pi, 0.2*pi];
x_sample2_norm = (x_sample2 - mu_input) ./ sigma_input;

% 预测并反归一化
predicted_output_norm = forward_pass(x_sample2_norm, W1, b1, W2, b2, W3, b3, W4, b4,W5, b5,relu);
predicted_output = predicted_output_norm .* sigma_output + mu_output;

% 显示结果
disp('预测输出:');
disp(predicted_output);

% 调用 TestBVP 和 T1012
[x, y, ak_sol, v_vals2] = TestBVP(tau, x_sample2);
[z] = invbvp(tau, x_sample2, predicted_output);
% qs2 = [z(1), z(3)];
% func_robot_new(qs2, 2, 'NN_Robot');
% qs1 = [y(1, :)', y(3, :)'];
% func_robot_new(qs1, 1, 'Solver_Robot');

% ==== 新增函数: Mini-batch 梯度计算 ====
function [dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5] = MiniBatchGradient(X_batch, Y_batch, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, activation_fn, activation_derivative)
    batch_size = size(X_batch, 1);
    dW1 = zeros(size(W1)); db1 = zeros(size(b1));
    dW2 = zeros(size(W2)); db2 = zeros(size(b2));
    dW3 = zeros(size(W3)); db3 = zeros(size(b3));
    dW4 = zeros(size(W4)); db4 = zeros(size(b4));
    dW5 = zeros(size(W5)); db5 = zeros(size(b5));
    
    % 累计梯度
    for k = 1:batch_size
        x_sample = X_batch(k, :);
        y_sample = Y_batch(k, :);
        
        % 前向传播
        z1 = x_sample * W1 + b1; a1 = activation_fn(z1);
        z2 = a1 * W2 + b2;       a2 = activation_fn(z2);
        z3 = a2 * W3 + b3;       a3 = activation_fn(z3);
        z4 = a3 * W4 + b4;       a4 = activation_fn(z4);  % 新增第四层
        z5 = a4 * W5 + b5;             
        
        % 反向传播
         output_error = y_sample - z5;
        delta5 = output_error;
        delta4 = (delta5 * W5') .* activation_derivative(a4);  % 第四层梯度
        delta3 = (delta4 * W4') .* activation_derivative(a3);
        delta2 = (delta3 * W3') .* activation_derivative(a2);
        delta1 = (delta2 * W2') .* activation_derivative(a1);


        % 累计梯度
        dW5 = dW5 + a4' * delta5;
        db5 = db5 + delta5;
        dW4 = dW4 + a3' * delta4;
        db4 = db4 + delta4;
        dW3 = dW3 + a2' * delta3;
        db3 = db3 + delta3;
        dW2 = dW2 + a1' * delta2;
        db2 = db2 + delta2;
        dW1 = dW1 + x_sample' * delta1;
        db1 = db1 + delta1;
    end
    
    % 平均梯度
    dW5 = dW5 / batch_size; db5 = db5 / batch_size;
    dW4 = dW4 / batch_size; db4 = db4 / batch_size;
    dW4 = dW4 / batch_size;
    db4 = db4 / batch_size;
    dW3 = dW3 / batch_size;
    db3 = db3 / batch_size;
    dW2 = dW2 / batch_size;
    db2 = db2 / batch_size;
    dW1 = dW1 / batch_size;
    db1 = db1 / batch_size;
end

% ==== 前向传播函数 ====
function output = forward_pass(x_sample, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, activation_fn)
    % 前向传播（含第四隐藏层）
    z1 = x_sample * W1 + b1;
    a1 = activation_fn(z1);
    
    z2 = a1 * W2 + b2;
    a2 = activation_fn(z2);
    
    z3 = a2 * W3 + b3;
    a3 = activation_fn(z3);
    
    % ==== 新增第四隐藏层 ==== 【修改点】
    z4 = a3 * W4 + b4;
    a4 = activation_fn(z4);
    
    z5 = a4 * W5 + b5;  % 输出层
    output = z5;
end

function v = guess(t, ak)
    N = length(ak);
    v = 0;
    ck = linspace(0, tau, N);
    for k = 1:N
        v = v + ak(k) * abs(t - ck(k))^3;
    end
end

