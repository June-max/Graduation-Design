function [cfs,SRA]=WPT(data,level,wname)
t=wpdec(data,level,wname); %��С��db7���źŽ��ж�߶ȷֽ�
% t = besttree(T);
for n=1:2^level
    SRA(n,:)=wprcoef(t,[level n-1]);
    cfs(n,:)=wpcoef(t,[level n-1]);
end
nodes=0:1:2^level-1;
ord=wpfrqord(nodes');
nodes_ord=nodes(ord);
SRA=SRA(ord,:);
cfs=cfs(ord,:);