%% 
% 多类别CSP计算
% 用法: W = MulticlassCSP(R,N)
% 输入:
% R - Covariance matrices of EEG data given class labels.
%     Dimensions: number of classes x channels x channels
% N - Number of spatial filters to return (e.g., two per class)
%
% 输出:
% W - Spatial filtering matrix
%     Dimensions: Number of spatial filters x EEG channels

function W = MulticlassCSP(R,N);

% 添加 ffdiag到路径
path(path,'ffdiag_pack');

% 参数
Classes = size(R,1); % 类别
Chans = size(R,2); % 通道
Pc = ones(1,Classes)./Classes; 

% 联合对角化协方差矩阵
disp('联合对角化协方差矩阵...');
[V,CD,stat] = ffdiag(shiftdim(R,1),eye(Chans));
V = V';

for n1 = 1:1:Chans,
    w = V(:,n1);
    I(n1) = J_ApproxMI(w,R,Pc);
end
[dummy(n1,:) iMI] = sort(I,'descend');
W = V(:,iMI(1:N))';

disp('Done.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I = J_ApproxMI(w,R,Pc)
M = size(R,1); 
N = size(R,2); 
Rx = zeros(N,N);
for n1 = 1:1:M,
    Rx = Rx + Pc(n1)*reshape(R(n1,:,:),N,N);
end

wv = w'*Rx*w;
w = w./sqrt(wv);
Ig = 1/2*log((w'*Rx*w));
for n1 = 1:1:M,
    Ig = Ig - 1/2*Pc(n1)*log(w'*reshape(R(n1,:,:),N,N)*w);
end
J = 0;
for n1 = 1:1:M,
    J = J + Pc(n1)*(w'*reshape(R(n1,:,:),N,N)*w)^2;
end
J = (J - 1)^2;
J = 9/48*J;
I = Ig - J;
