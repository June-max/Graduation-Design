function [SRA SRD]=DWT(data,fs,L,level)
[C,L]=wavedec(data,level,'sym7'); %��С��db7���źŽ��ж�߶ȷֽ�

% �÷ֽ�ϵ���ع�
for n=1:level
    SRA(:,n)=wrcoef('a',C,L,'sym5',n);
    SRD(:,n)=wrcoef('d',C,L,'sym5',n);
end