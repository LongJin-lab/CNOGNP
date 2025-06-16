function [output] = sharpness_max(x,r,num)
%SHARPNESS1 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
x_evaluate = linspace(x-r,x+r,num);

global gap_max;
gap_max = 0;
for i = 1:num
    gap_tem = abs(Fitness_Function(x_evaluate(i))-Fitness_Function(x));
    if gap_tem > gap_max
        gap_max = gap_tem;
    end
end
output = gap_max;
end


