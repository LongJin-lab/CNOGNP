clc;
clear;
close all;

% 定义一维函数和其导数
% f = @(x) x^2 + 5*sin(5*x);
% df = @(x) 2*x + 25*cos(5*x);

% f = @(x) x^2;
% df = @(x) 2*x ;
% f = @(x) sin(x - 1.5 *cos(x)) - cos(2 * x - 2 *sin(x));
% df = @(x) cos(x - (3*cos(x))/2)*((3*sin(x))/2 + 1) - sin(2*x - 2*sin(x))*(2*cos(x) - 2);

func_num=2
% 定义一维函数和其导数
lamda =0
f = @(x) cos(2^(0.8*x)+0.1*x) + lamda * (-1*sin(2.^(0.8*x)+0.1*x)*(log(2)*0.8*2.^(0.8*x)+0.1))^2;
f_original = @(x) cos(2.^(0.8*x)+0.1*x)   ;
df = @(x) -1*sin(2.^(0.8*x)+0.1*x)*(log(2)*0.8*2.^(0.8*x)+0.1);


% PSO 参数
num_particles = 30;
max_iterations = 500;
c1 = 1.5;
c2 = 1.5;
w = 0.7;
learning_rate = 0.01;

% 粒子位置的范围限制
position_min = -2;  % 最小位置限制
position_max = 6;   % 最大位置限制

% 初始化粒子群
% 初始化粒子群的位置（等间距分布在指定范围内）
lower_bound = position_min;
upper_bound = position_max;
particles.position = linspace(lower_bound, upper_bound, num_particles);
particles.velocity = zeros(1, num_particles);
particles.gd_position = zeros(1, num_particles);
particles.best_position = particles.position;
particles.best_fitness = arrayfun(f, particles.position);
global_best_fitness = min(particles.best_fitness);
global_best_position = particles.position(particles.best_fitness == global_best_fitness);

colors = jet(num_particles); % 使用 jet 色图生成颜色
% 创建图形窗口
figure;
hold on;



% 开始 PSO 迭代
for iteration = 1:max_iterations
    record_num=1
    if iteration == 1
        scatter(particles.position, iteration * ones(1, num_particles), 50, colors, 'filled');
    end
    if mod(iteration, record_num) == 0
        scatter(particles.position, iteration * ones(1, num_particles), 50, colors, 'filled');
    end
    % 绘制每个粒子的位置，并为每个粒子使用不同的颜色
%     scatter(particles.position, iteration * ones(1, num_particles), [], colors, 'filled');

    
    % 显示当前迭代的最佳适应度
    disp(['Iteration ', num2str(iteration), ', Best Fitness: ', num2str(global_best_fitness)]);
    
    for i = 1:num_particles
        % 更新粒子速度和位置(GD_PSO_GNP)
        particles.gd_position(i) = particles.position(i) - learning_rate * df(particles.position(i));

        particles.velocity(i) = w * (particles.gd_position(i) - particles.position(i)) + c1 * rand() * (particles.best_position(i) - particles.position(i)) + c2 * rand() * (global_best_position(1) - particles.position(i));
        particles.position(i) = particles.position(i) + particles.velocity(i);

        particles.position(i) = max(min(particles.position(i), position_max), position_min);
        
        % 计算适应度并更新个体最佳和全局最佳
        current_fitness = f(particles.position(i));
        if current_fitness < particles.best_fitness(i)
            particles.best_fitness(i) = current_fitness;
            particles.best_position(i) = particles.position(i);
        end
        if current_fitness < global_best_fitness
            global_best_fitness = current_fitness;
            global_best_position = particles.position(i);
        end
    end
    
    
end

% 设置图形标题和标签
box on;
ylim([0, 520]); 
yticks([0, 250, 500]);
set(gca,'FontName','Times New Roman','FontSize',25);
xlabel('$x$', 'Interpreter', 'latex','FontSize',30);
ylabel('# Iteration','FontSize',30,'FontName','Times New Roman');
fileName = sprintf('figures/1-f%d-Particle-Positions-During-Iterations-lamda-%.2f.fig', func_num,lamda);
fileName_eps = sprintf('figures/1-f%d-Particle-Positions-During-Iterations-lamda-%.2f.eps', func_num,lamda);
print(fileName_eps,'-depsc','-r1000');
savefig(fileName);

% 创建一维函数的图形
x = linspace(-2, 6, 400);  % 选择适当的 x 范围
y = arrayfun(f_original, x);

% 绘制函数图像
figure;
plot(x, y, 'k', 'LineWidth', 1.5);
hold on;

% 绘制所有粒子在函数上的分布

scatter(particles.position, arrayfun(f_original, particles.position), 70, colors, 'filled');
%scatter(particles.position, arrayfun(f_original, particles.position), 50, 'r', 'filled');
set(gca,'FontName','Times New Roman','FontSize',25);
xlabel('$x$', 'Interpreter', 'latex','FontSize',30);
ylabel('$f(x)$', 'Interpreter', 'latex','FontSize',30);
% savefig('figures/1-final-particles-location.fig');
fileName = sprintf('figures/1-f%d-final-particles-location-lamda-%.2f.fig', func_num, lamda);
fileName_eps = sprintf('figures/1-f%d-final-particles-location-lamda-%.2f.eps', func_num, lamda);
print(fileName_eps,'-depsc','-r1000');% save eps with 600dpi
savefig(fileName);


% 显示最终结果
disp('Optimization completed.');
disp(['Global Best Position: ', num2str(global_best_position)]);
disp(['Global Best Fitness: ', num2str(global_best_fitness)]);
