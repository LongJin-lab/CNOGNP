clc;
clear;
close all;

%-----------------------------------f2---------------------------------------------------
func_num=2
% 定义要优化的二维函数 f1
lamda = 0
fun_orginal = @(x) cos(2^(0.8*x(1))+0.1*x(x1))   ;
fun = @(x) cos(2.^(0.8*x(1))+0.1*x(1))+cos(2.^(0.8*x(2))+0.1*x(2))+lamda*((-1*sin(2.^(0.8*x(1))+0.1*x(1))*(log(2)*0.8*2.^(0.8*x(1))+0.1)).^2+(-1*sin(2.^(0.8*x(2))+0.1*x(2))*(log(2)*0.8*2.^(0.8*x(2))+0.1))^2);            
gradient_fun = @(x) [-1*sin(2.^(0.8*x(1))+0.1*x(1))*(log(2)*0.8*2.^(0.8*x(1))+0.1),-1*sin(2.^(0.8*x(2))+0.1*x(2))*(log(2)*0.8*2.^(0.8*x(2))+0.1)];

% 创建一组坐标点以绘制等高线图和3D图
x1 = linspace(-2, 6, 300); % 在 x1 轴上创建均匀分布的点
x2 = linspace(-2, 6, 300); % 在 x2 轴上创建均匀分布的点
[X1, X2] = meshgrid(x1, x2); % 创建网格点矩阵

% 计算函数值
Z = cos(2.^(0.8*X1)+0.1*X1)+cos(2.^(0.8*X2)+0.1*X2)
%-----------------------------------f1---------------------------------------------------

%-----------------------------------f2---------------------------------------------------
% func_num=2
%-----------------------------------f2---------------------------------------------------
% 绘制等高线图
figure;

contour(X1, X2, Z, 30); % 20 表示等高线的数量
set(gca,'FontName','Times New Roman','FontSize',25);
xlabel('$x_1$', 'Interpreter', 'latex','FontSize',30);
ylabel('$x_2$', 'Interpreter', 'latex','FontSize',30);

colorbar; % 添加颜色条
fileName = sprintf('figures/2-f%d-function-figure-contour-map-lamda-%.2f.fig', func_num,lamda);
fileName_eps = sprintf('figures/2-f%d-function-figure-contour-map-lamda-%.2f.eps', func_num,lamda);
print(fileName_eps,'-depsc','-r2000');
savefig(fileName);


% 绘制3D图
figure;
surf(X1, X2, Z);
set(gca,'FontName','Times New Roman','FontSize',25);
xlabel('$x_1$', 'Interpreter', 'latex','FontSize',30);
ylabel('$x_2$', 'Interpreter', 'latex','FontSize',30);
zlabel('$f(x_1,x_2)$', 'Interpreter', 'latex');

% 设置视角
view(3); % 设置为三维视角
shading interp; % 添加平滑着色
fileName = sprintf('figures/2-f%d-function-figure-3d-lamda-%.2f.fig', func_num,lamda);
fileName_eps = sprintf('figures/2-f%d-function-figure-3d-lamda-%.2f.eps', func_num,lamda);
print(fileName_eps,'-depsc','-r1000');
savefig(fileName);














% 设置优化参数
num_particles = 30;
num_dimensions = 2;
max_iterations = 500;
c1 = 1.5;
c2 = 1.5;
w = 0.7;
learning_rate = 0.1;

% 粒子位置的范围限制
position_min = -2;  % 最小位置限制
position_max = 6;   % 最大位置限制
lower_bound = position_min;
upper_bound = position_max;



particles = lower_bound + rand(num_particles, num_dimensions) * (upper_bound - lower_bound);
velocities = zeros(num_particles, num_dimensions);

% 绘制初始粒子位置分布
figure;
colors = jet(num_particles); % 使用 jet 色图生成颜色
scatter(particles(:, 1), particles(:, 2), 50, colors, 'filled');
set(gca,'FontName','Times New Roman','FontSize',25);
xlabel('$x_1$', 'Interpreter', 'latex','FontSize',30);
ylabel('$x_2$', 'Interpreter', 'latex','FontSize',30);
axis([lower_bound upper_bound lower_bound upper_bound]);
box on;
grid on;
fileName_eps = sprintf('figures/2-f%d-initial-particles-location-plane-lamda-%.2f.eps', func_num,lamda);
print(fileName_eps,'-depsc','-r1000');
savefig(fileName);


% % 初始化粒子的位置和速度
% particles = rand(num_particles, num_dimensions);
% velocities = zeros(num_particles, num_dimensions);

% 初始化粒子的个体最佳位置和全局最佳位置
p_best_positions = particles;
g_best_position = particles(1, :);
g_best_value = fun(g_best_position);

% 在循环外初始化一个空矩阵，用于存储每次迭代的粒子位置
particle_positions_history = zeros(max_iterations, num_particles, num_dimensions);


% 开始优化循环
for iter = 1:max_iterations
    % 更新粒子速度和位置
    for i = 1:num_particles
       
        particle_position = particles(i, :);
        particle_value = fun(particle_position);
        particle_gradient = gradient_fun(particle_position);
        
        r1 = rand();
        r2 = rand();
        cognitive_component = c1 * r1 * (p_best_positions(i, :) - particles(i, :));
        social_component = c2 * r2 * (g_best_position - particles(i, :));
        gradient_component = w * (-1) * learning_rate * particle_gradient;
        velocities(i, :) = gradient_component + cognitive_component + social_component;
        particles(i, :) = particles(i, :) + velocities(i, :);
        % 限制粒子位置在指定范围内
        particles(i, :) = max(min(particles(i, :), upper_bound), lower_bound);
        
        % 更新个体最佳位置
        if fun(particles(i, :)) < fun(p_best_positions(i, :))
            p_best_positions(i, :) = particles(i, :);
        end
        
        % 更新全局最佳位置
        if fun(particles(i, :)) < g_best_value
            g_best_position = particles(i, :);
            g_best_value = fun(particles(i, :));
        end
        
        % 记录当前迭代的粒子位置
        record_num=20
        if iter == 1
            particle_positions_history(iter, :, :) = particles;
        end
        if mod(iter, record_num) == 0
            particle_positions_history(iter, :, :) = particles;
        end
    end
end

% 创建一个新的图形窗口，用于绘制粒子位置迭代图
figure;
for i = 1:num_particles
    x_positions = squeeze(particle_positions_history(1, i, 1));
    y_positions = squeeze(particle_positions_history(1, i, 2));
    z_positions = 1; 
    scatter3(x_positions, y_positions, z_positions, 50, colors(i, :));
    hold on;
end


for i = 1:num_particles
    x_positions = squeeze(particle_positions_history(record_num:record_num:end, i, 1));
    y_positions = squeeze(particle_positions_history(record_num:record_num:end, i, 2));
%     z_positions = 1:max_iterations; % 对应的迭代次数
    z_positions = record_num:record_num:max_iterations; % 仅使用每隔 record_num 次迭代的迭代次数
    scatter3(x_positions, y_positions, z_positions, 50, colors(i, :));
    hold on;
end
set(gca,'FontName','Times New Roman','FontSize',25);
xlabel('$x_1$', 'Interpreter', 'latex','FontSize',30);
ylabel('$x_2$', 'Interpreter', 'latex','FontSize',30);
zlabel('Iteration number', 'Interpreter', 'latex','FontSize',30);
grid on;
view(3); % 设置视角为三维
colormap(colors); % 使用相同的颜色映射
fileName = sprintf('figures/2-f%d-Particle-Positions-During-Iterations-lamda-%.2f.fig', func_num,lamda);
fileName_eps = sprintf('figures/2-f%d-Particle-Positions-During-Iterations-lamda-%.2f.eps', func_num,lamda);
print(fileName_eps,'-depsc','-r1000');
savefig(fileName);
hold off;


% 显示最终结果
fprintf('优化完成，最优位置：(%f, %f)，最优值：%f\n', g_best_position(1), g_best_position(2), g_best_value);


% 绘制函数图像
figure; % 创建新的图形窗口
[x, y] = meshgrid(-2:0.1:6, -2:0.1:6);
% z = sin(x.^4) + sin(y.^4);
z = cos(2.^(0.8*x)+0.1*x)+cos(2.^(0.8*y)+0.1*y)
surf(x, y, z, 'FaceAlpha', 0.3);
hold on;
shading interp; % 添加平滑着色

% 绘制最终粒子位置
scatter3(g_best_position(1), g_best_position(2), g_best_value, 100, 'r', 'filled');
set(gca,'FontName','Times New Roman','FontSize',25);
xlabel('$x_1$', 'Interpreter', 'latex','FontSize',30);
ylabel('$x_2$', 'Interpreter', 'latex','FontSize',30);
zlabel('$f(x_1,x_2)$', 'Interpreter', 'latex','FontSize',30);
fileName = sprintf('figures/2-f%d-final-best-particle-location-3d-lamda-%.2f.fig', func_num,lamda);
fileName_eps = sprintf('figures/2-f%d-final-best-particle-location-3d-lamda-%.2f.eps', func_num,lamda);
print(fileName_eps,'-depsc','-r1000');
savefig(fileName);
hold off; % 结束绘图

%绘制g_best在等高线图中的位置
% 创建一个新的图形窗口，绘制函数的等高线图
figure;
contour(x, y, z, 30);
colorbar; % 添加颜色映射条
hold on;

% 绘制最终粒子位置
scatter(g_best_position(1), g_best_position(2), 100, 'r', 'filled');
set(gca,'FontName','Times New Roman','FontSize',25);
xlabel('$x_1$', 'Interpreter', 'latex','FontSize',30);
ylabel('$x_2$', 'Interpreter', 'latex','FontSize',30);
grid on;
hold off;
fileName = sprintf('figures/2-f%d-final-best-particle-location-contour-map-lamda-%.2f.fig', func_num,lamda);
fileName_eps = sprintf('figures/2-f%d-final-best-particle-location-contour-map-lamda-%.2f.eps', func_num,lamda);
print(fileName_eps,'-depsc','-r1000');
savefig(fileName);
% 绘制最终粒子位置分布，平面图
figure;
scatter(particles(:, 1), particles(:, 2), 50, colors, 'filled');
set(gca,'FontName','Times New Roman','FontSize',25);
xlabel('$x_1$', 'Interpreter', 'latex','FontSize',30);
ylabel('$x_2$', 'Interpreter', 'latex','FontSize',30);
axis([lower_bound upper_bound lower_bound upper_bound]);
box on;
grid on;
fileName = sprintf('figures/2-f%d-final-particles-location-plane-lamda-%.2f.fig', func_num,lamda);
fileName_eps = sprintf('figures/2-f%d-final-particles-location-plane-lamda-%.2f.eps', func_num,lamda);
print(fileName_eps,'-depsc','-r1000');
savefig(fileName);
