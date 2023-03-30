function  X=feature_computing(EPO,Fs)
wnd = [0.5 4.5];
wnd = round(Fs*wnd(1))+1 : round(Fs*wnd(2));
% CSP获得空间滤波矩阵S
C_1 = cov(EPO{1});
C_2 = cov(EPO{2});
C_3 = cov(EPO{3});
C_4 = cov(EPO{4});
%整合成三维卷积矩阵
R = zeros(4,22,22);
R(1,:,:)=C_1;
R(2,:,:)=C_2;
R(3,:,:)=C_3;
R(4,:,:)=C_4;
nof=11;
S = MulticlassCSP(R,2*nof);
% 对数-方差特征提取
for k = 1:4
    X{k} = squeeze(log(var(reshape(EPO{k}*S', length(wnd),[],2*nof))));
end
end
