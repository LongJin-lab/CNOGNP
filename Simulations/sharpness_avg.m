function [output] = sharpness_avg(x,r,num)
%SHARPNESS1 此处显示有关此函数的摘要
%   此处显示详细说明
x_evaluate = linspace(x-r,x+r,num);

output =0;
for i = 1:num
    gap = abs(Fitness_Function(x_evaluate(i))-Fitness_Function(x));
    output = output +  gap;
end
output=output/num;
end


