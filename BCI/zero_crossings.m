function count = zero_crossings(x)
% x 必须是1位的行向量或者列向量 时域信号
% count为返回的过零率计算

count = 0;
if(length(x) == 1)
    error('ERROR: input signal must have more than one element');
end

if((size(x, 2) ~= 1) && (size(x, 1) ~= 1))
    error('ERROR: Input must be one-dimensional');
end
x = x(:);

num_samples = length(x);
for i=2:num_samples
    
    if((x(i) * x(i-1)) < 0)
        count = count + 1;
    end
    
end
end