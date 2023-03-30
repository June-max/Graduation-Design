function [cfs,SRA]=WPT(data,level,wname)
t=wpdec(data,level,wname); %用小波db7对信号进行多尺度分解
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