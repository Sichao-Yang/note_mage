机器学习是目前信息技术中最激动人心的方向之一，其应用已经深入到生活的各个层面且与普通人的日常生活密切相关。本文为清华大学最新出版的《机器学习》教材的学习笔记，书作者是南京大学周志华教授，本书是机器学习理论基础的经典教材。笔记里主要总结了该教材的知识点以及对书中一些关键公式进行推导和解析。

总结部分借鉴笔记作者zyhuster1，公式部分借鉴datawhale的南瓜书

## 绪论

## 1.1 机器学习的定义

正如我们根据过去的经验来判断明天的天气，吃货们希望从购买经验中挑选一个好瓜，那能不能让计算机帮助人类来实现这个呢？机器学习正是这样的一门学科，人的“经验”对应计算机中的“数据”，让计算机来学习这些经验数据，生成一个算法模型，在面对新的情况中，计算机便能作出有效的判断，这便是机器学习。

另一本经典教材的作者Mitchell给出了一个形式化的定义，假设：

 - P：计算机程序在某任务类T上的性能。
 - T：计算机程序希望实现的任务类。
 - E：表示经验，即历史的数据集。

若该计算机程序通过利用经验E在任务T上获得了性能P的改善，则称该程序对E进行了学习。

## 1.2 机器学习的一些基本术语

假设我们收集了一批西瓜的数据，例如：（色泽=青绿;根蒂=蜷缩;敲声=浊响)， (色泽=乌黑;根蒂=稍蜷;敲声=沉闷)， (色泽=浅自;根蒂=硬挺;敲声=清脆)……每对括号内是一个西瓜的记录，定义：	 

 - 所有记录的集合为：数据集。

 - 每一条记录为：一个实例（instance）或样本（sample）。

 - 例如：色泽或敲声，单个的特点为特征（feature）或属性（attribute）。

 - 对于一条记录，如果在坐标轴上表示，每个西瓜都可以用坐标轴中的一个点表示，一个点也是一个向量，例如（青绿，蜷缩，浊响），即每个西瓜为：一个特征向量（feature vector）。

 - 一个样本的特征数为：维数（dimensionality），该西瓜的例子维数为3，当维数非常大时，也就是现在说的“维数灾难”。

	计算机程序学习经验数据生成算法模型的过程中，每一条记录称为一个“训练样本”，同时在训练好模型后，我们希望使用新的样本来测试模型的效果，则每一个新的样本称为一个“测试样本”。定义：	

 - 所有训练样本的集合为：训练集（trainning set），[特殊]。

 - 所有测试样本的集合为：测试集（test set），[一般]。  

 - 机器学习出来的模型适用于新样本的能力为：泛化能力（generalization），即从特殊到一般。

	西瓜的例子中，我们是想计算机通过学习西瓜的特征数据，训练出一个决策模型，来判断一个新的西瓜是否是好瓜。可以得知我们预测的是：西瓜是好是坏，即好瓜与差瓜两种，是离散值。同样地，也有通过历年的人口数据，来预测未来的人口数量，人口数量则是连续值。定义：	

 - 预测值为离散值的问题为：分类（classification）。

 - 预测值为连续值的问题为：回归（regression）。

我们预测西瓜是否是好瓜的过程中，很明显对于训练集中的西瓜，我们事先已经知道了该瓜是否是好瓜，学习器通过学习这些好瓜或差瓜的特征，从而总结出规律，即训练集中的西瓜我们都做了标记，称为标记信息。但也有没有标记信息的情形，例如：我们想将一堆西瓜根据特征分成两个小堆，使得某一堆的西瓜尽可能相似，即都是好瓜或差瓜，对于这种问题，我们事先并不知道西瓜的好坏，样本没有标记信息。定义：	

 - 训练数据有标记信息的学习任务为：监督学习（supervised learning），容易知道上面所描述的分类和回归都是监督学习的范畴。
 - 训练数据没有标记信息的学习任务为：无监督学习（unsupervised learning），常见的有聚类和关联规则。

# 南瓜书

## 1.2

$$
\begin{aligned}
\sum_{f}E_{ote}(\mathfrak{L}_a\vert X,f) &= \sum_f\sum_h\sum_{\boldsymbol{x}\in\mathcal{X}-X}P(\boldsymbol{x})\mathbb{I}(h(\boldsymbol{x})\neq f(\boldsymbol{x}))P(h\vert X,\mathfrak{L}_a) \\
&=\sum_{\boldsymbol{x}\in\mathcal{X}-X}P(\boldsymbol{x}) \sum_hP(h\vert X,\mathfrak{L}_a)\sum_f\mathbb{I}(h(\boldsymbol{x})\neq f(\boldsymbol{x})) \\
&=\sum_{\boldsymbol{x}\in\mathcal{X}-X}P(\boldsymbol{x}) \sum_hP(h\vert X,\mathfrak{L}_a)\cfrac{1}{2}2^{\vert \mathcal{X} \vert} \\
&=\cfrac{1}{2}2^{\vert \mathcal{X} \vert}\sum_{\boldsymbol{x}\in\mathcal{X}-X}P(\boldsymbol{x}) \sum_hP(h\vert X,\mathfrak{L}_a) \\
&=2^{\vert \mathcal{X} \vert-1}\sum_{\boldsymbol{x}\in\mathcal{X}-X}P(\boldsymbol{x}) \cdot 1\\
\end{aligned}
$$

[解析]：第一步到第二步是因为$\sum_i^m\sum_j^n\sum_k^o a_ib_jc_k=\sum_i^m a_i \cdot \sum_j^n b_j \cdot \sum_k^o c_k$；第二步到第三步：首先要知道此时$f$的定义为**任何能将样本映射到{0,1}的函数+均匀分布**，也即不止一个$f$且每个$f$出现的概率相等，例如样本空间只有两个样本时：$ \mathcal{X}=\{\boldsymbol{x}_1,\boldsymbol{x}_2\},\vert \mathcal{X} \vert=2$，那么所有的真实目标函数$f$为：

$$
\begin{aligned}
f_1:f_1(\boldsymbol{x}_1)=0,f_1(\boldsymbol{x}_2)=0;\\
f_2:f_2(\boldsymbol{x}_1)=0,f_2(\boldsymbol{x}_2)=1;\\
f_3:f_3(\boldsymbol{x}_1)=1,f_3(\boldsymbol{x}_2)=0;\\
f_4:f_4(\boldsymbol{x}_1)=1,f_4(\boldsymbol{x}_2)=1;
\end{aligned}
$$

一共$2^{\vert \mathcal{X} \vert}=2^2=4$个真实目标函数。所以此时通过算法$\mathfrak{L}_a$学习出来的模型$h(\boldsymbol{x})$对每个样本无论预测值为0还是1必然有一半的$f$与之预测值相等，例如，现在学出来的模型$h(\boldsymbol{x})$对$\boldsymbol{x}_1$的预测值为1，也即$h(\boldsymbol{x}_1)=1$，那么有且只有$f_3$和$f_4$与$h(\boldsymbol{x})$的预测值相等，也就是有且只有一半的$f$与它预测值相等，所以$\sum_f\mathbb{I}(h(\boldsymbol{x})\neq f(\boldsymbol{x})) = \cfrac{1}{2}2^{\vert \mathcal{X} \vert} $。

第三步一直到最后显然成立。值得一提的是，在这里我们定义真实的目标函数为**“任何能将样本映射到{0,1}的函数+均匀分布”**，但是实际情形并非如此，通常我们只认为能高度拟合已有样本数据的函数才是真实目标函数，例如，现在已有的样本数据为$\{(\boldsymbol{x}_1,0),(\boldsymbol{x}_2,1)\}$，那么此时$f_2$才是我们认为的真实目标函数，由于没有收集到或者压根不存在$\{(\boldsymbol{x}_1,0),(\boldsymbol{x}_2,0)\},\{(\boldsymbol{x}_1,1),(\boldsymbol{x}_2,0)\},\{(\boldsymbol{x}_1,1),(\boldsymbol{x}_2,1)\}$这类样本，所以$f_1,f_3,f_4$都不算是真实目标函数。
