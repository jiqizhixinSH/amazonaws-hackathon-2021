利用MATLAB来构建出“气候影响的评价指标的选取和等级排序”

1、
%%四个返回函数
function [Entropy,Purity,FMeasure,Accuracy] = Fmeasure(P,C)
% P为人工标记簇
% C为聚类算法计算结果
N = length(C);% 样本总数
p = unique(P);
c = unique(C);
P_size = length(p);% 人工标记的簇的个数
C_size = length(c);% 算法计算的簇的个数
% Pid,Rid：非零数据：第i行非零数据代表的样本属于第i个簇
Pid = double(ones(P_size,1)*P == p'*ones(1,N) );
Cid = double(ones(C_size,1)*C == c'*ones(1,N) );
CP = Cid*Pid';%P和C的交集,C*P
Pj = sum(CP,1);% 行向量，P在C各个簇中的个数
Ci = sum(CP,2);% 列向量，C在P各个簇中的个数

precision = CP./( Ci*ones(1,P_size) );
recall = CP./( ones(C_size,1)*Pj );
F = 2*precision.*recall./(precision+recall);
% 得到一个总的F值
FMeasure = sum( (Pj./sum(Pj)).*max(F) );
Accuracy = sum(max(CP,[],2))/N;
%得到聚类效果 Entropy和Purity
C_i=Ci';
[e1 p1]=EnAndPur(CP ,C_i);
Entropy=e1;
Purity=p1;
end

2、
%%计算熵与纯度
%% 计算聚类效果熵与纯度 输入的矩阵为 CP：算法聚类与实际类别得到的数据的交集
%Ci 算法聚类得到的每个类别的总数
function [Entropy Purity]=EnAndPur(CP,Ci)
%得到行列值
[rn cn]=size(PC);
%% 计算熵
%计算概率 precision
for i=1:rn
    for j=1:cn
     precision(i,j)=PC(i,j)/Ci(1,i);    
    end
end
%计算ei(i,j)
for i=1:rn
    for j=1:cn
     ei(i,j)=precision(i,j)*log2(precision(i,j));    
    end
end
%
%计算ei_sum
for i=1:rn
    ei_sum(i)=-nansum(ei(i,:));
end
%计算mi*ei_sum(i)
for j=1:cn
    mmi(j)=Ci(1,j)*ei_sum(j);
end
%计算entropy
Entropy=nansum(mmi)/nansum(Ci);
%% 计算纯度Purity
%找出最大的一类
for i=1:rn
     pr_max(i)=max(precision(i,:));    
end
%计算类别数量
for j=1:cn
    nni(j)=Ci(1,j)*pr_max(j);
end
Purity=nansum(nni)/nansum(Ci);
end

3、
%%逐步回归
%%XY为对应的气候影响评级的上一步的产出
X=[7,26,6,60;1,29,15,52;11,56,8,20;11,31,8,47;7,52,6,33;11,55,9,22;3,71,17,6;1,31,22,44;2,54,18,22;21,47,4,26;1,40,23,34;11,66,9,12];   %自变量数据
Y=[78.5,74.3,104.3,87.6,95.9,109.2,102.7,72.5,93.1,115.9,83.8,113.3];  %因变量数据
stepwise(X,Y,[1,2,3,4],0.05,0.10)
% in=[1,2,3,4]表示X1、X2、X3、X4均保留在模型中

%4、
%用SPSS最优尺度来分析和删取不同量纲的影响因素 

%第一步：打开主菜单。

%在SPSS数据视图下，在菜单栏中选择【分析】【回归】【最优尺度】选项，调出SPSS分类回归主菜单界面。

%第二步：定义尺度。

%为因变量和所有自变量指定最合适的测度类别。
%首先从左侧的变量栏中选择“满意度”，按箭头按钮方向移入因变量框内，选中底部的“定义尺度”按钮，
%打开相应对话框，因为满意度的3个取值水平是代表着满意程度，含有次序信息，因此选择“有序”单选按钮，
%完成对因变量的最优尺度定义。
%相似的，将3个自变量移入自变量框内，性别定义为名义尺度，年龄定义为有序尺度，月收入定义为有序尺度。

%第三步：其他参数设置

%此时直接点击主菜单下的“确定”按钮，即可执行最优尺度回归过程，其他参数接受SPSS软件的默认设置。
%为了得到更多直观的结果，有必要设置更多参数。本案例主要设置【图】按钮菜单里的参数。

%打开【分类回归：图】按钮菜单，将所有变量移入右侧的转换图框内，要求软件输出原分类变量各取值经最优尺度变换后的数值对应图。

%最后就是解读给出的图表