function [SRA SRD]=DWT(data,fs,L,level)
[C,L]=wavedec(data,level,'sym7'); %用小波db7对信号进行多尺度分解

% 用分解系数重构
for n=1:level
    SRA(:,n)=wrcoef('a',C,L,'sym5',n);
    SRD(:,n)=wrcoef('d',C,L,'sym5',n);
end