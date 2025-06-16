clc;
clear;
close all;

%-----------------------------------f2---------------------------------------------------
func_num=2
% ����Ҫ�Ż��Ķ�ά���� f1
lamda = 0
fun_orginal = @(x) cos(2^(0.8*x(1))+0.1*x(x1))   ;
fun = @(x) cos(2.^(0.8*x(1))+0.1*x(1))+cos(2.^(0.8*x(2))+0.1*x(2))+lamda*((-1*sin(2.^(0.8*x(1))+0.1*x(1))*(log(2)*0.8*2.^(0.8*x(1))+0.1)).^2+(-1*sin(2.^(0.8*x(2))+0.1*x(2))*(log(2)*0.8*2.^(0.8*x(2))+0.1))^2);            
gradient_fun = @(x) [-1*sin(2.^(0.8*x(1))+0.1*x(1))*(log(2)*0.8*2.^(0.8*x(1))+0.1),-1*sin(2.^(0.8*x(2))+0.1*x(2))*(log(2)*0.8*2.^(0.8*x(2))+0.1)];

% ����һ��������Ի��Ƶȸ���ͼ��3Dͼ
x1 = linspace(-2, 6, 300); % �� x1 ���ϴ������ȷֲ��ĵ�
x2 = linspace(-2, 6, 300); % �� x2 ���ϴ������ȷֲ��ĵ�
[X1, X2] = meshgrid(x1, x2); % ������������

% ���㺯��ֵ
Z = cos(2.^(0.8*X1)+0.1*X1)+cos(2.^(0.8*X2)+0.1*X2)
%-----------------------------------f1---------------------------------------------------

%-----------------------------------f2---------------------------------------------------
% func_num=2
%-----------------------------------f2---------------------------------------------------
% ���Ƶȸ���ͼ
figure;

contour(X1, X2, Z, 30); % 20 ��ʾ�ȸ��ߵ�����
set(gca,'FontName','Times New Roman','FontSize',25);
xlabel('$x_1$', 'Interpreter', 'latex','FontSize',30);
ylabel('$x_2$', 'Interpreter', 'latex','FontSize',30);

colorbar; % �����ɫ��
fileName = sprintf('figures/2-f%d-function-figure-contour-map-lamda-%.2f.fig', func_num,lamda);
fileName_eps = sprintf('figures/2-f%d-function-figure-contour-map-lamda-%.2f.eps', func_num,lamda);
print(fileName_eps,'-depsc','-r2000');
savefig(fileName);


% ����3Dͼ
figure;
surf(X1, X2, Z);
set(gca,'FontName','Times New Roman','FontSize',25);
xlabel('$x_1$', 'Interpreter', 'latex','FontSize',30);
ylabel('$x_2$', 'Interpreter', 'latex','FontSize',30);
zlabel('$f(x_1,x_2)$', 'Interpreter', 'latex');

% �����ӽ�
view(3); % ����Ϊ��ά�ӽ�
shading interp; % ���ƽ����ɫ
fileName = sprintf('figures/2-f%d-function-figure-3d-lamda-%.2f.fig', func_num,lamda);
fileName_eps = sprintf('figures/2-f%d-function-figure-3d-lamda-%.2f.eps', func_num,lamda);
print(fileName_eps,'-depsc','-r1000');
savefig(fileName);














% �����Ż�����
num_particles = 30;
num_dimensions = 2;
max_iterations = 500;
c1 = 1.5;
c2 = 1.5;
w = 0.7;
learning_rate = 0.1;

% ����λ�õķ�Χ����
position_min = -2;  % ��Сλ������
position_max = 6;   % ���λ������
lower_bound = position_min;
upper_bound = position_max;



particles = lower_bound + rand(num_particles, num_dimensions) * (upper_bound - lower_bound);
velocities = zeros(num_particles, num_dimensions);

% ���Ƴ�ʼ����λ�÷ֲ�
figure;
colors = jet(num_particles); % ʹ�� jet ɫͼ������ɫ
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


% % ��ʼ�����ӵ�λ�ú��ٶ�
% particles = rand(num_particles, num_dimensions);
% velocities = zeros(num_particles, num_dimensions);

% ��ʼ�����ӵĸ������λ�ú�ȫ�����λ��
p_best_positions = particles;
g_best_position = particles(1, :);
g_best_value = fun(g_best_position);

% ��ѭ�����ʼ��һ���վ������ڴ洢ÿ�ε���������λ��
particle_positions_history = zeros(max_iterations, num_particles, num_dimensions);


% ��ʼ�Ż�ѭ��
for iter = 1:max_iterations
    % ���������ٶȺ�λ��
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
        % ��������λ����ָ����Χ��
        particles(i, :) = max(min(particles(i, :), upper_bound), lower_bound);
        
        % ���¸������λ��
        if fun(particles(i, :)) < fun(p_best_positions(i, :))
            p_best_positions(i, :) = particles(i, :);
        end
        
        % ����ȫ�����λ��
        if fun(particles(i, :)) < g_best_value
            g_best_position = particles(i, :);
            g_best_value = fun(particles(i, :));
        end
        
        % ��¼��ǰ����������λ��
        record_num=20
        if iter == 1
            particle_positions_history(iter, :, :) = particles;
        end
        if mod(iter, record_num) == 0
            particle_positions_history(iter, :, :) = particles;
        end
    end
end

% ����һ���µ�ͼ�δ��ڣ����ڻ�������λ�õ���ͼ
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
%     z_positions = 1:max_iterations; % ��Ӧ�ĵ�������
    z_positions = record_num:record_num:max_iterations; % ��ʹ��ÿ�� record_num �ε����ĵ�������
    scatter3(x_positions, y_positions, z_positions, 50, colors(i, :));
    hold on;
end
set(gca,'FontName','Times New Roman','FontSize',25);
xlabel('$x_1$', 'Interpreter', 'latex','FontSize',30);
ylabel('$x_2$', 'Interpreter', 'latex','FontSize',30);
zlabel('Iteration number', 'Interpreter', 'latex','FontSize',30);
grid on;
view(3); % �����ӽ�Ϊ��ά
colormap(colors); % ʹ����ͬ����ɫӳ��
fileName = sprintf('figures/2-f%d-Particle-Positions-During-Iterations-lamda-%.2f.fig', func_num,lamda);
fileName_eps = sprintf('figures/2-f%d-Particle-Positions-During-Iterations-lamda-%.2f.eps', func_num,lamda);
print(fileName_eps,'-depsc','-r1000');
savefig(fileName);
hold off;


% ��ʾ���ս��
fprintf('�Ż���ɣ�����λ�ã�(%f, %f)������ֵ��%f\n', g_best_position(1), g_best_position(2), g_best_value);


% ���ƺ���ͼ��
figure; % �����µ�ͼ�δ���
[x, y] = meshgrid(-2:0.1:6, -2:0.1:6);
% z = sin(x.^4) + sin(y.^4);
z = cos(2.^(0.8*x)+0.1*x)+cos(2.^(0.8*y)+0.1*y)
surf(x, y, z, 'FaceAlpha', 0.3);
hold on;
shading interp; % ���ƽ����ɫ

% ������������λ��
scatter3(g_best_position(1), g_best_position(2), g_best_value, 100, 'r', 'filled');
set(gca,'FontName','Times New Roman','FontSize',25);
xlabel('$x_1$', 'Interpreter', 'latex','FontSize',30);
ylabel('$x_2$', 'Interpreter', 'latex','FontSize',30);
zlabel('$f(x_1,x_2)$', 'Interpreter', 'latex','FontSize',30);
fileName = sprintf('figures/2-f%d-final-best-particle-location-3d-lamda-%.2f.fig', func_num,lamda);
fileName_eps = sprintf('figures/2-f%d-final-best-particle-location-3d-lamda-%.2f.eps', func_num,lamda);
print(fileName_eps,'-depsc','-r1000');
savefig(fileName);
hold off; % ������ͼ

%����g_best�ڵȸ���ͼ�е�λ��
% ����һ���µ�ͼ�δ��ڣ����ƺ����ĵȸ���ͼ
figure;
contour(x, y, z, 30);
colorbar; % �����ɫӳ����
hold on;

% ������������λ��
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
% ������������λ�÷ֲ���ƽ��ͼ
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
