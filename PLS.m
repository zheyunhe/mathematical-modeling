% data:m行n列矩阵,其中n=a+b(a:自变量,b:因变量)
%      m表示共m个观测值(样本)
%      data=自变量+因变量(位置不能改变)
%
% separate:表示自变量的最后一列在data中的列数
function PLS(data,separate)
format long g                       % 长小数的显示方式
%% 数据处理
mu=mean(data);sig=std(data);        % 求均值和标准差
r=corrcoef(data);                   % 求相关系数矩阵
data=zscore(data);                  % 数据标准化
x=data(:,[1:separate]);             % 提出标准化后的自变量数据
y=data(:,[separate+1:end]);         % 提出标准化后的因变量数据
%% 判断提出成分对的个数
[XL1,YL1,XS1,YS1,BETA1,PCTVAR1,MSE1,stats1] =plsregress(x,y);
xw=x\XS1;                                                                                        % 求自变量提出成分的系数,每列对应一个成分,这里xw等于stats.W
yw=y\YS1;                                                                                        % 求因变量提出成分的系数
x_contr=PCTVAR1(1,:);y_contr=PCTVAR1(2,:);                                                       % 自变量和因变量提出成分的贡献率
x_contr_cumsum=cumsum(x_contr);y_contr_cumsum=cumsum(y_contr);                                   % 计算累计贡献率
disp('自变量提出成分的累计贡献率:');disp(x_contr_cumsum');
disp('因变量提出成分的累计贡献率:');disp(y_contr_cumsum');
ncomp=input('请输入选取前几个成分:');                                                             % 选取的主成分对的个数
%% 打印提出的成分对的信息
fprintf('%d对成分分别为:\n',ncomp);
for i=1:ncomp
    fprintf('第%d对成分:\n',i);
    % 打印自变量的主成分
    fprintf('u%d=',i);
    for k=1:size(x,2)	 % 此处为变量x的个数
        fprintf('+(%f*x%d)',xw(k,i),k);
    end
    fprintf('\n');
    % 打印因变量的主成分
    fprintf('v%d=',i);
    for k=1:size(y,2)    % 此处为变量y的个数
        fprintf('+(%f*y%d)',yw(k,i),k);
    end
    fprintf('\n');
end
%% 确定主成分后的回归分析
[XL2,YL2,XS2,YS2,BETA2,PCTVAR2,MSE2,stats2] =plsregress(x,y,ncomp);
n=size(x,2);m=size(y,2);                                                    % n:自变量的个数,m:因变量的个数
beta3(1,:)=mu(n+1:end)-mu(1:n)./sig(1:n)*BETA2([2:end],:).*sig(n+1:end);    % 原始数据回归方程的常数项
beta3([2:n+1],:)=(1./sig(1:n))'*sig(n+1:end).*BETA2([2:end],:);             % 计算原始变量x1,...,xn的系数，每一列是一个回归方程
fprintf('得出回归方程如下:\n')
for i=1:size(y,2)           % 此处为变量y的个数
    fprintf('y%d=%f',i,beta3(1,i));
    for j=1:size(x,2)       % 此处为变量x的个数
        fprintf('+(%f*x%d)',beta3(j+1,i),j);
    end
    fprintf('\n');
end
%% 求预测值
y_forecast = repmat(beta3(1,:),[size(x,1),1])+data(:,[1:n])*beta3([2:end],:);       % 求y1,..,ym的预测值
y_real = data(:,end-size(y_forecast,2)+1:end);                                      % 真实值
%% 画回归系数的直方图
figure;
bar(BETA2','k');
title('回归系数直方图');
%% 贡献率画图
figure;
percent_explained_x = 100 * PCTVAR1(1,:) / sum(PCTVAR1(1,:));
pareto(percent_explained_x);
xlabel('主成分')
ylabel('贡献率(%)')
title('主成分对自变量的贡献率')

figure;
percent_explained_y = 100 * PCTVAR1(2,:) / sum(PCTVAR1(2,:));
pareto(percent_explained_y);
xlabel('主成分')
ylabel('贡献率(%)')
title('主成分对因变量的贡献率')
%% 绘制预测结果和真实值的对比
num = size(x,1);                    % 样本个数
for i =1:size(y,2)
	yr = y_real(:,i);               % 真实值
    yf = y_forecast(:,i);           % 预测值    
    % 计算R方
    R2 = (num*sum(yf.*yr)-sum(yf)*sum(yr))^2/((num*sum((yf).^2)-(sum(yf))^2)*(num*sum((yr).^2)-(sum(yr))^2));
    figure;
    plot(1:num,yr,'b:*',1:num,yf,'r-o')
    legend('真实值','预测值','location','best')
    xlabel('预测样本')
    ylabel('值')
    string = {['第',num2str(i),'个因变量预测结果对比'];['R^2=' num2str(R2)]};
    title(string)
end
%% 检验网络性能
% 剩余标准差/回归系统的拟合标准差
RMSE = sqrt(mean((y_real-y_forecast).^2));
for i=1:1:size(y,2)
    fprintf('第%d个回归方程的RMSE:%f\n',i,RMSE(i));
end
% for i =1:size(y,2)
%     % 回归图
%     figure(5);
%     plotregression(yr,yf,['第',num2str(i),'个回归图']);
%     % 误差直方图
%     e = yr-yf;
%     figure(6);
%     ploterrhist(e,['第',num2str(i),'个误差直方图']);
% end
format              % 恢复到短小数的显示方式
