last update: 20230701

author: sichaoY

[toc]

上篇主要介绍了几种常用的聚类算法，首先从距离度量与性能评估出发，列举了常见的距离计算公式与聚类评价指标，接着分别讨论了K-Means、LVQ、高斯混合聚类、密度聚类以及层次聚类算法。K-Means与LVQ都试图以类簇中心作为原型指导聚类，其中K-Means通过EM算法不断迭代直至收敛，LVQ使用真实类标辅助聚类；高斯混合聚类采用高斯分布来描述类簇原型；密度聚类则是将一个核心对象所有密度可达的样本形成类簇，直到所有核心对象都遍历完；最后层次聚类是一种自底向上的树形聚类方法，不断合并最相近的两个小类簇。本篇将讨论机器学习常用的方法--降维与度量学习。

# **10. 降维与度量学习**

样本的特征数称为**维数**（dimensionality），当维数非常大时，也就是现在所说的“**维数灾难**”，具体表现在：在高维情形下，**数据样本将变得十分稀疏**，因为此时要满足训练样本为“**密采样**”的总体样本数目是一个触不可及的天文数字，谓可远观而不可亵玩焉...**训练样本的稀疏使得其代表总体分布的能力大大减弱，从而消减了学习器的泛化能力**；同时当维数很高时，**计算距离也变得十分复杂**，甚至连计算内积都不再容易，这也是为什么支持向量机（SVM）使用核函数**“低维计算，高维表现”**的原因。

缓解维数灾难的一个重要途径就是**降维，即通过某种数学变换将原始高维空间转变到一个低维的子空间**。在这个子空间中，样本的密度将大幅提高，同时距离计算也变得容易。这时也许会有疑问，这样降维之后不是会丢失原始数据的一部分信息吗？这是因为在很多实际的问题中，虽然训练数据是高维的，但是与学习任务相关也许仅仅是其中的一个低维子空间，也称为一个**低维嵌入**，例如：数据属性中存在噪声属性、相似属性或冗余属性等，**对高维数据进行降维能在一定程度上达到提炼低维优质属性或降噪的效果**。

## **10.1 K近邻学习**

k近邻算法简称**kNN（k-Nearest Neighbor）**，是一种经典的监督学习方法，同时也实力担当入选数据挖掘十大算法。其工作机制十分简单粗暴：给定某个测试样本，kNN基于某种**距离度量**在训练集中找出与其距离最近的k个带有真实标记的训练样本，然后给基于这k个邻居的真实标记来进行预测，类似于前面集成学习中所讲到的基学习器结合策略：分类任务采用投票法，回归任务则采用平均法。接下来本篇主要就kNN分类进行讨论。

![1.png](media\Chapter10_降维与度量学习\5bc851a43873a.png)

从上图【来自Wiki】中我们可以看到，图中有两种类型的样本，一类是蓝色正方形，另一类是红色三角形。而那个绿色圆形是我们待分类的样本。基于kNN算法的思路，我们很容易得到以下结论：

> 如果K=3，那么离绿色点最近的有2个红色三角形和1个蓝色的正方形，这3个点投票，于是绿色的这个待分类点属于红色的三角形。
> 如果K=5，那么离绿色点最近的有2个红色三角形和3个蓝色的正方形，这5个点投票，于是绿色的这个待分类点属于蓝色的正方形。

可以发现：**kNN虽然是一种监督学习方法，但是它却没有显式的训练过程**，而是当有新样本需要预测时，才来计算出最近的k个邻居，因此**kNN是一种典型的懒惰学习方法**，再来回想一下朴素贝叶斯的流程，训练的过程就是参数估计，因此朴素贝叶斯也可以懒惰式学习，此类技术在**训练阶段开销为零**，待收到测试样本后再进行计算。相应地我们称那些一有训练数据立马开工的算法为“**急切学习**”，可见前面我们学习的大部分算法都归属于急切学习。

很容易看出：**kNN算法的核心在于k值的选取以及距离的度量**。k值选取太小，模型很容易受到噪声数据的干扰，例如：极端地取k=1，若待分类样本正好与一个噪声数据距离最近，就导致了分类错误；若k值太大， 则在更大的邻域内进行投票，此时模型的预测能力大大减弱，例如：极端取k=训练样本数，就相当于模型根本没有学习，所有测试样本的预测结果都是一样的。**一般地我们都通过交叉验证法来选取一个适当的k值**。

![2.png](media\Chapter10_降维与度量学习\5bc851a47db9a.png)

对于距离度量，**不同的度量方法得到的k个近邻不尽相同，从而对最终的投票结果产生了影响**，因此选择一个合适的距离度量方法也十分重要。在上一篇聚类算法中，在度量样本相似性时介绍了常用的几种距离计算方法，包括**闵可夫斯基距离，曼哈顿距离，VDM**等。在实际应用中，**kNN的距离度量函数一般根据样本的特性来选择合适的距离度量，同时应对数据进行去量纲/归一化处理来消除大量纲属性的强权政治影响**。

### 10.1

$$
P(e r r)=1-\sum_{c \in \mathcal{Y}} P(c | \boldsymbol{x}) P(c | \boldsymbol{z})
$$

[解析]：$P(c | \boldsymbol{x}) P(c | \boldsymbol{z})$表示$x$和$z$同属类$c$的概率，对所有可能的类别$c\in\mathcal{Y}$求和，则得到$x$和$z$同属相同类别的概率，因此$1-\sum_{c \in \mathcal{Y}} P(c | \boldsymbol{x}) P(c | \boldsymbol{z})$表示$x$和$z$分属不同类别的概率。

### 10.2

$$
\begin{aligned}
P(e r r) &=1-\sum_{c \in \mathcal{Y}} P(c | \boldsymbol{x}) P(c | \boldsymbol{z}) \\
& \simeq 1-\sum_{c \in \mathcal{Y}} P^{2}(c | \boldsymbol{x}) \\
& \leqslant 1-P^{2}\left(c^{*} | \boldsymbol{x}\right) \\
&=\left(1+P\left(c^{*} | \boldsymbol{x}\right)\right)\left(1-P\left(c^{*} | \boldsymbol{x}\right)\right) \\
& \leqslant 2 \times\left(1-P\left(c^{*} | \boldsymbol{x}\right)\right)
\end{aligned}
$$

[解析]：第二个式子是来源于前提假设"假设样本独立同分布，且对任意$x$和任意小正数$\delta$，在$x$附近$\delta$距离范围内总能找到一个训练样本"，则$P(c | \boldsymbol{z}) = P(c | \boldsymbol{x\pm\delta})\simeq P(c|\boldsymbol{x})$。第三个式子是应为$c^{*}\in\mathcal{Y}$，因此$P^{2}\left(c^{*} | \boldsymbol{x}\right)$是$\sum_{c \in \mathcal{Y}} P^{2}(c | \boldsymbol{x})$的一个分量，所以$\sum_{c \in \mathcal{Y}} P^{2}(c | \boldsymbol{x}) \geqslant P^{2}\left(c^{*} | \boldsymbol{x}\right)$。第四个式子是平方差公式展开，最后一个式子因为$1 + P^{2}\left(c^{*} | \boldsymbol{x}\right)\leqslant 2$。

## **10.2 低纬嵌入 - MDS算法**

不管是使用核函数升维还是对数据降维，我们都希望**原始空间样本点之间的距离在新空间中基本保持不变**，这样才不会使得原始空间样本之间的关系及总体分布发生较大的改变。**“多维缩放”（MDS）**正是基于这样的思想，**MDS要求原始空间样本之间的距离在降维后的低维空间中得以保持**。

![image-20200910140521330](media\Chapter10_降维与度量学习\image-20200910140521330.png)

假定m个维度为d的样本在原始空间中任意两两样本之间的距离矩阵为D∈R^(mxm)^，我们的目标便是获得样本在低维空间中的表示Z∈R^d'xm^, d'< d)，且任意两个样本在低维空间中的欧式距离等于原始空间中的距离，即$||zi-zj||=\text{Dist}(i,j)$。因此接下来我们要做的就是根据已有的距离矩阵D来求解出降维后的坐标矩阵Z。

![3.png](media\Chapter10_降维与度量学习\5bc851a4b664e.png)

令降维后的样本坐标矩阵Z被中心化，**中心化是指将每个样本向量减去整个样本集的均值向量，故所有样本向量求和得到一个零向量**。这样易知：矩阵B的每一列以及每一列求和均为0，因为提取公因子后可得  $Z_1\sum_{k=1}^m{Z_k}=Z_1*0=0$。

![4.png](media\Chapter10_降维与度量学习\5bc851a4a4ee2.png)

根据上面矩阵B的特征，我们很容易得到等式（2）、（3）以及（4）：

![5.png](media\Chapter10_降维与度量学习\5bc851a4a777b.png)

这时根据(1)-(4)式我们便可以推导出内积距离矩阵的元素b~ij~的计算公式，然后我们对矩阵B做特征分解，对eigenvalue做筛除，然后做重构出低纬矩阵Z，其中每一行就是一个低维坐标点。

### 10.3

$$
\begin{aligned}
\operatorname{dist}_{i j}^{2} &=\left\|\boldsymbol{z}_{i}\right\|^{2}+\left\|\boldsymbol{z}_{j}\right\|^{2}-2 \boldsymbol{z}_{i}^{\mathrm{T}} \boldsymbol{z}_{j} \\
&=b_{i i}+b_{j j}-2 b_{i j}
\end{aligned}
$$

[推导]：
$$
\begin{aligned}
\operatorname{dist}_{i j}^{2} &=\left\|\boldsymbol{z}_{i}-\boldsymbol{z}_{j}\right\|^{2}=\left(\boldsymbol{z}_{i}-\boldsymbol{z}_{j}\right)^{\top}\left(\boldsymbol{z}_{i}-\boldsymbol{z}_{j}\right) \\
&=\boldsymbol{z}_{i}^{\top} \boldsymbol{z}_{i}-\boldsymbol{z}_{i}^{\top} \boldsymbol{z}_{j}-\boldsymbol{z}_{j}^{\top} \boldsymbol{z}_{i}+\boldsymbol{z}_{j}^{\top} \boldsymbol{z}_{j} \\
&=\boldsymbol{z}_{i}^{\top} \boldsymbol{z}_{i}+\boldsymbol{z}_{j}^{\top} \boldsymbol{z}_{j}-2 \boldsymbol{z}_{i}^{\top} \boldsymbol{z}_{j} \\
&=\left\|\boldsymbol{z}_{i}\right\|^{2}+\left\|\boldsymbol{z}_{j}\right\|^{2}-2 \boldsymbol{z}_{i}^{\top} \boldsymbol{z}_{j} \\
&=b_{i i}+b_{j j}-2 b_{i j}
\end{aligned}
$$


### 10.4

$$
\sum^m_{i=1}dist^2_{ij}=tr(\boldsymbol B)+mb_{jj}
$$

[解析]：首先根据式10.3有
$$
\sum^m_{i=1}dist^2_{ij}= \sum^m_{i=1}b_{ii}+\sum^m_{i=1}b_{jj}-2\sum^m_{i=1}b_{ij}
$$
对于第一项，根据矩阵迹的定义，$\sum^m_{i=1}b_{ii} = tr(\boldsymbol B)$，对于第二项，由于求和号内元素和$i$无关，因此$\sum^m_{i=1}b_{jj}=mb_{jj}$，对于第三项有，
$$
\sum_{i=1}^{m} b_{i j}=\sum_{i=1}^{m} \boldsymbol{z}_{i}^{\top} \boldsymbol{z}_{j}=
\sum_{i=1}^{m} \boldsymbol{z}_{j}^{\top} \boldsymbol{z}_{i}=
\boldsymbol{z}_{j}^{\top} \sum_{i=1}^{m} \boldsymbol{z}_{i}=
\boldsymbol{z}_{j}^{\top} \cdot \mathbf{0}=0
$$
其中$\sum_{i=1}^{m} \boldsymbol{z}_{i}=\mathbf{0}$是利用了书上的前提条件，即将降维后的样本被中心化。

### 10.5

$$
\sum_{j=1}^{m} \operatorname{dist}_{i j}^{2}=\operatorname{tr}(\mathbf{B})+m b_{i i}
$$

[解析]：参考10.4



### 10.6

$$
\sum_{i=1}^{m} \sum_{j=1}^{m} \operatorname{dist}_{i j}^{2}=2 m \operatorname{tr}(\mathbf{B})
$$

[推导]：
$$
\begin{aligned}
\sum_{i=1}^{m} \sum_{j=1}^{m} \operatorname{dist}_{i j}^{2} &=\sum_{i=1}^{m} \sum_{j=1}^{m}\left(\left\|z_{i}\right\|^{2}+\left\|\boldsymbol{z}_{j}\right\|^{2}-2 \boldsymbol{z}_{i}^{\top} \boldsymbol{z}_{j}\right) \\
&=\sum_{i=1}^{m} \sum_{j=1}^{m}\left\|\boldsymbol{z}_{i}\right\|^{2}+\sum_{i=1}^{m} \sum_{j=1}^{m}\left\|\boldsymbol{z}_{j}\right\|^{2}-2 \sum_{i=1}^{m} \sum_{j=1}^{m} \boldsymbol{z}_{i}^{\top} \boldsymbol{z}_{j} \\
\end{aligned}
$$
其中
$$
\sum_{i=1}^{m} \sum_{j=1}^{m}\left\|\boldsymbol{z}_{i}\right\|^{2}=m \sum_{i=1}^{m}\left\|\boldsymbol{z}_{i}\right\|^{2}=m \operatorname{tr}(\mathbf{B})
$$

$$
\sum_{i=1}^{m} \sum_{j=1}^{m}\left\|\boldsymbol{z}_{j}\right\|^{2}=m \sum_{j=1}^{m}\left\|\boldsymbol{z}_{j}\right\|^{2}=m \operatorname{tr}(\mathbf{B})
$$

$$
\sum_{i=1}^{m} \sum_{j=1}^{m} \boldsymbol{z}_{i}^{\top} \boldsymbol{z}_{j}=0
$$

最后一个式子是来自于书中的假设，假设降维后的样本$\mathbf{Z}$被中心化。

### 10.10

$$
b_{ij}=-\frac{1}{2}(dist^2_{ij}-dist^2_{i\cdot}-dist^2_{\cdot j}+dist^2_{\cdot\cdot})
$$



[推导]：由公式（10.3）可得
$$
b_{ij}=-\frac{1}{2}(dist^2_{ij}-b_{ii}-b_{jj})
$$


由公式（10.6）和（10.9）可得
$$
\begin{aligned}
tr(\boldsymbol B)&=\frac{1}{2m}\sum^m_{i=1}\sum^m_{j=1}dist^2_{ij}\\
&=\frac{m}{2}dist^2_{\cdot}
\end{aligned}
$$
由公式（10.4）和（10.8）可得
$$
\begin{aligned}
b_{jj}&=\frac{1}{m}\sum^m_{i=1}dist^2_{ij}-\frac{1}{m}tr(\boldsymbol B)\\
&=dist^2_{\cdot j}-\frac{1}{2}dist^2_{\cdot}
\end{aligned}
$$
由公式（10.5）和（10.7）可得
$$
\begin{aligned}
b_{ii}&=\frac{1}{m}\sum^m_{j=1}dist^2_{ij}-\frac{1}{m}tr(\boldsymbol B)\\
&=dist^2_{i\cdot}-\frac{1}{2}dist^2_{\cdot}
\end{aligned}
$$
综合可得
$$
\begin{aligned}
b_{ij}&=-\frac{1}{2}(dist^2_{ij}-b_{ii}-b_{jj})\\
&=-\frac{1}{2}(dist^2_{ij}-dist^2_{i\cdot}+\frac{1}{2}dist^2_{\cdot\cdot}-dist^2_{\cdot j}+\frac{1}{2}dist^2_{\cdot\cdot})\\
&=-\frac{1}{2}(dist^2_{ij}-dist^2_{i\cdot}-dist^2_{\cdot j}+dist^2_{\cdot\cdot})
\end{aligned}
$$

这时只需对B进行特征值分解便可以得到Z。MDS的算法流程如下图所示：

![6.png](media\Chapter10_降维与度量学习\5bc851a5340dd.png)

<font color='magenta'>我一开始看这个算法有个疑问：既然目的是要在低维空间上保留高维空间里的距离D，那为什么还有必要去计算低维空间里的内积矩阵B，难道不应该是B==D的吗？其实不然，因为高维空间里的**距离**并没有限制，它不一定是欧式距离，可以是流形的上黎曼’距离‘，甚至这个距离可以不是对称的（我猜），wiki上对它的用词是proximity matrix，自己品。所以B的元素值是可能和D不一样的</font>

## **10.3 主成分分析（PCA）**

不同于MDS采用距离保持的方法，**主成分分析（PCA）直接通过一个线性变换，将原始空间中的样本投影到新的低维空间中**。简单来理解这一过程便是：**PCA采用一组新的基来表示样本点，其中每一个基向量都是原来基向量的线性组合，通过使用尽可能少的新基向量来表示样本，从而达到降维的目的。**

假设使用d'个新基向量来表示原来样本，实质上是将样本投影到一个由d'个基向量确定的一个**超平面**上（**即舍弃了一些维度**），要用一个超平面对空间中所有高维样本进行恰当的表达，最理想的情形是：**若这些样本点都能在超平面上表出且这些表出在超平面上都能够很好地分散开来**。但是一般使用较原空间低一些维度的超平面来做到这两点十分不容易，因此我们退一步海阔天空，要求这个超平面应具有如下两个性质：

> **最近重构性**：样本点到超平面的距离足够近，即尽可能在超平面附近；
> **最大可分性**：样本点在超平面上的投影尽可能地分散开来，即投影后的坐标具有区分性。

这里十分神奇的是：**最近重构性与最大可分性虽然从不同的出发点来定义优化问题中的目标函数，但最终这两种特性得到了完全相同的优化问题**：

![7.png](media\Chapter10_降维与度量学习\5bc851a5213c1.png)

接着使用拉格朗日乘子法求解上面的优化问题，得到：

![8.png](media\Chapter10_降维与度量学习\5bc851a4a102a.png)

因此只需对协方差矩阵进行特征值分解即可求解出W，PCA算法的整个流程如下图所示：

![9.png](media\Chapter10_降维与度量学习\5bc851a540eb3.png)

### 10.14

$$
\begin{aligned}
\sum^m_{i=1}\left\| \sum^{d'}_{j=1}z_{ij}\boldsymbol{w}_j-\boldsymbol x_i \right\|^2_2&=\sum^m_{i=1}\boldsymbol z^{\mathrm{T}}_i\boldsymbol z_i-2\sum^m_{i=1}\boldsymbol z^{\mathrm{T}}_i\mathbf{W}^{\mathrm{T}}\boldsymbol x_i +\text { const }\\
&\propto -\operatorname{tr}(\mathbf{W}^{\mathrm{T}}(\sum^m_{i=1}\boldsymbol x_i\boldsymbol x^{\mathrm{T}}_i)\mathbf{W})
\end{aligned}
$$

[推导]：已知$\mathbf{W}^{\mathrm{T}} \mathbf{W}=\mathbf{I},\boldsymbol z_i=\mathbf{W}^{\mathrm{T}} \boldsymbol x_i$，则
$$
\begin{aligned}
\sum^m_{i=1}\left\| \sum^{d'}_{j=1}z_{ij}\boldsymbol{w}_j-\boldsymbol x_i \right\|^2_2&=\sum^m_{i=1}\left\|\mathbf{W}\boldsymbol z_i-\boldsymbol x_i \right\|^2_2\\
&= \sum^m_{i=1} \left(\mathbf{W}\boldsymbol z_i-\boldsymbol x_i\right)^{\mathrm{T}}\left(\mathbf{W}\boldsymbol z_i-\boldsymbol x_i\right)\\
&= \sum^m_{i=1} \left(\boldsymbol z_i^{\mathrm{T}}\mathbf{W}^{\mathrm{T}}\mathbf{W}\boldsymbol z_i- \boldsymbol z_i^{\mathrm{T}}\mathbf{W}^{\mathrm{T}}\boldsymbol x_i-\boldsymbol x_i^{\mathrm{T}}\mathbf{W}\boldsymbol z_i+\boldsymbol x_i^{\mathrm{T}}\boldsymbol x_i \right)\\
&= \sum^m_{i=1} \left(\boldsymbol z_i^{\mathrm{T}}\boldsymbol z_i- 2\boldsymbol z_i^{\mathrm{T}}\mathbf{W}^{\mathrm{T}}\boldsymbol x_i+\boldsymbol x_i^{\mathrm{T}}\boldsymbol x_i \right)\\
&=\sum^m_{i=1}\boldsymbol z_i^{\mathrm{T}}\boldsymbol z_i-2\sum^m_{i=1}\boldsymbol z_i^{\mathrm{T}}\mathbf{W}^{\mathrm{T}}\boldsymbol x_i+\sum^m_{i=1}\boldsymbol x^{\mathrm{T}}_i\boldsymbol x_i\\
&=\sum^m_{i=1}\boldsymbol z_i^{\mathrm{T}}\boldsymbol z_i-2\sum^m_{i=1}\boldsymbol z_i^{\mathrm{T}}\mathbf{W}^{\mathrm{T}}\boldsymbol x_i+\text { const }\\
&=\sum^m_{i=1}\boldsymbol z_i^{\mathrm{T}}\boldsymbol z_i-2\sum^m_{i=1}\boldsymbol z_i^{\mathrm{T}}\boldsymbol z_i+\text { const }\\
&=-\sum^m_{i=1}\boldsymbol z_i^{\mathrm{T}}\boldsymbol z_i+\text { const }\\
&=-\sum^m_{i=1}\operatorname{tr}\left(\boldsymbol z_i\boldsymbol z_i^{\mathrm{T}}\right)+\text { const }\\
&=-\operatorname{tr}\left(\sum^m_{i=1}\boldsymbol z_i\boldsymbol z_i^{\mathrm{T}}\right)+\text { const }\\
&=-\operatorname{tr}\left(\sum^m_{i=1}\mathbf{W}^{\mathrm{T}} \boldsymbol x_i\boldsymbol x_i^{\mathrm{T}}\mathbf{W}\right)+\text { const }\\
&= -\operatorname{tr}\left(\mathbf{W}^{\mathrm{T}}\left(\sum^m_{i=1}\boldsymbol x_i\boldsymbol x^{\mathrm{T}}_i\right)\mathbf{W}\right)+\text { const }\\
&\propto-\operatorname{tr}\left(\mathbf{W}^{\mathrm{T}}\left(\sum^m_{i=1}\boldsymbol x_i\boldsymbol x^{\mathrm{T}}_i\right)\mathbf{W}\right)\\
\end{aligned}
$$


### 10.17

$$
\mathbf X\mathbf X^{\mathrm{T}} \boldsymbol w_i=\lambda _i\boldsymbol w_i
$$

[推导]：由式（10.15）可知，主成分分析的优化目标为
$$
\begin{aligned}
&\min\limits_{\mathbf W} \quad-\text { tr }(\mathbf W^{\mathrm{T}} \mathbf X\mathbf X^{\mathrm{T}} \mathbf W)\\
&s.t. \quad\mathbf W^{\mathrm{T}} \mathbf W=\mathbf I
\end{aligned}
$$
其中，$\mathbf{X}=\left(\boldsymbol{x}_{1}, \boldsymbol{x}_{2}, \ldots, \boldsymbol{x}_{m}\right) \in \mathbb{R}^{d \times m},\mathbf{W}=\left(\boldsymbol{w}_{1}, \boldsymbol{w}_{2}, \ldots, \boldsymbol{w}_{d^{\prime}}\right) \in \mathbb{R}^{d \times d^{\prime}}$，$\mathbf{I} \in \mathbb{R}^{d^{\prime} \times d^{\prime}}$为单位矩阵。对于带矩阵约束的优化问题，根据<a href="#ref1">[1]</a>中讲述的方法可得此优化目标的拉格朗日函数为
$$
\begin{aligned}
L(\mathbf W,\Theta)&=-\text { tr }(\mathbf W^{\mathrm{T}} \mathbf X\mathbf X^{\mathrm{T}} \mathbf W)+\langle \Theta,\mathbf W^{\mathrm{T}} \mathbf W-\mathbf I\rangle \\
&=-\text { tr }(\mathbf W^{\mathrm{T}} \mathbf X\mathbf X^{\mathrm{T}} \mathbf W)+\text { tr }\left(\Theta^{\mathrm{T}} (\mathbf W^{\mathrm{T}} \mathbf W-\mathbf I)\right) 
\end{aligned}
$$


其中，$\Theta  \in \mathbb{R}^{d^{\prime} \times d^{\prime}}$为拉格朗日乘子矩阵，其维度恒等于约束条件的维度，且其中的每个元素均为未知的拉格朗日乘子，$\langle \Theta,\mathbf W^{\mathrm{T}} \mathbf W-\mathbf I\rangle = \text { tr }\left(\Theta^{\mathrm{T}} (\mathbf W^{\mathrm{T}} \mathbf W-\mathbf I)\right)$为矩阵的内积<sup><a href="#ref2">[2]</a></sup>。若此时仅考虑约束$\boldsymbol{w}_i^{\mathrm{T}}\boldsymbol{w}_i=1(i=1,2,...,d^{\prime})$，则拉格朗日乘子矩阵$\Theta$此时为对角矩阵，令新的拉格朗日乘子矩阵为$\Lambda=diag(\lambda_1,\lambda_2,...,\lambda_{d^{\prime}})\in \mathbb{R}^{d^{\prime} \times d^{\prime}}$，则新的拉格朗日函数为
$$
L(\mathbf W,\Lambda)=-\text { tr }(\mathbf W^{\mathrm{T}} \mathbf X\mathbf X^{\mathrm{T}} \mathbf W)+\text { tr }\left(\Lambda^{\mathrm{T}} (\mathbf W^{\mathrm{T}} \mathbf W-\mathbf I)\right)
$$
对拉格朗日函数关于$\mathbf{W}$求导可得
$$
\begin{aligned}
\cfrac{\partial L(\mathbf W,\Lambda)}{\partial \mathbf W}&=\cfrac{\partial}{\partial \mathbf W}\left[-\text { tr }(\mathbf W^{\mathrm{T}} \mathbf X\mathbf X^{\mathrm{T}} \mathbf W)+\text { tr }\left(\Lambda^{\mathrm{T}} (\mathbf W^{\mathrm{T}} \mathbf W-\mathbf I)\right)\right] \\
&=-\cfrac{\partial}{\partial \mathbf W}\text { tr }(\mathbf W^{\mathrm{T}} \mathbf X\mathbf X^{\mathrm{T}} \mathbf W)+\cfrac{\partial}{\partial \mathbf W}\text { tr }\left(\Lambda^{\mathrm{T}} (\mathbf W^{\mathrm{T}} \mathbf W-\mathbf I)\right) \\
\end{aligned}
$$
由矩阵微分公式$\cfrac{\partial}{\partial \mathbf{X}} \text { tr }(\mathbf{X}^{\mathrm{T}}  \mathbf{B} \mathbf{X})=\mathbf{B X}+\mathbf{B}^{\mathrm{T}}  \mathbf{X},\cfrac{\partial}{\partial \mathbf{X}} \text { tr }\left(\mathbf{B X}^{\mathrm{T}}  \mathbf{X}\right)=\mathbf{X B}^{\mathrm{T}} +\mathbf{X B}$可得
$$
\begin{aligned}
\cfrac{\partial L(\mathbf W,\Lambda)}{\partial \mathbf W}&=-2\mathbf X\mathbf X^{\mathrm{T}} \mathbf W+\mathbf{W}\Lambda+\mathbf{W}\Lambda^{\mathrm{T}}  \\
&=-2\mathbf X\mathbf X^{\mathrm{T}} \mathbf W+\mathbf{W}(\Lambda+\Lambda^{\mathrm{T}} ) \\
&=-2\mathbf X\mathbf X^{\mathrm{T}} \mathbf W+2\mathbf{W}\Lambda
\end{aligned}
$$
令$\cfrac{\partial L(\mathbf W,\Lambda)}{\partial \mathbf W}=\mathbf 0$可得
$$
\begin{aligned}
-2\mathbf X\mathbf X^{\mathrm{T}} \mathbf W+2\mathbf{W}\Lambda&=\mathbf 0\\
\mathbf X\mathbf X^{\mathrm{T}} \mathbf W&=\mathbf{W}\Lambda\\
\end{aligned}
$$
将$\mathbf W$和$\Lambda$展开可得
$$
\mathbf X\mathbf X^{\mathrm{T}} \boldsymbol w_i=\lambda _i\boldsymbol w_i,\quad i=1,2,...,d^{\prime}
$$
显然，此式为矩阵特征值和特征向量的定义式，其中$\lambda_i,\boldsymbol w_i$分别表示矩阵$\mathbf X\mathbf X^{\mathrm{T}}$的特征值和单位特征向量。由于以上是仅考虑约束$\boldsymbol{w}_i^{\mathrm{T}}\boldsymbol{w}_i=1$所求得的结果，而$\boldsymbol{w}_i$还需满足约束$\boldsymbol{w}_{i}^{\mathrm{T}}\boldsymbol{w}_{j}=0(i\neq j)$。观察$\mathbf X\mathbf X^{\mathrm{T}}$的定义可知，$\mathbf X\mathbf X^{\mathrm{T}}$是一个实对称矩阵，实对称矩阵的不同特征值所对应的特征向量之间相互正交，同一特征值的不同特征向量可以通过施密特正交化使其变得正交，所以通过上式求得的$\boldsymbol w_i$可以同时满足约束$\boldsymbol{w}_i^{\mathrm{T}}\boldsymbol{w}_i=1,\boldsymbol{w}_{i}^{\mathrm{T}}\boldsymbol{w}_{j}=0(i\neq j)$。根据拉格朗日乘子法的原理可知，此时求得的结果仅是最优解的必要条件，而且$\mathbf X\mathbf X^{\mathrm{T}}$有$d$个相互正交的单位特征向量，所以还需要从这$d$个特征向量里找出$d^{\prime}$个能使得目标函数达到最优值的特征向量作为最优解。将$\mathbf X\mathbf X^{\mathrm{T}} \boldsymbol w_i=\lambda _i\boldsymbol w_i$代入目标函数可得
$$
\begin{aligned}
\min\limits_{\mathbf W}-\text { tr }(\mathbf W^{\mathrm{T}} \mathbf X\mathbf X^{\mathrm{T}} \mathbf W)&=\max\limits_{\mathbf W}\text { tr }(\mathbf W^{\mathrm{T}} \mathbf X\mathbf X^{\mathrm{T}} \mathbf W) \\
&=\max\limits_{\mathbf W}\sum_{i=1}^{d^{\prime}}\boldsymbol w_i^{\mathrm{T}}\mathbf X\mathbf X^{\mathrm{T}} \boldsymbol w_i \\
&=\max\limits_{\mathbf W}\sum_{i=1}^{d^{\prime}}\boldsymbol w_i^{\mathrm{T}}\cdot\lambda _i\boldsymbol w_i \\
&=\max\limits_{\mathbf W}\sum_{i=1}^{d^{\prime}}\lambda _i\boldsymbol w_i^{\mathrm{T}}\boldsymbol w_i \\
&=\max\limits_{\mathbf W}\sum_{i=1}^{d^{\prime}}\lambda _i \\
\end{aligned}
$$


显然，此时只需要令$\lambda_1,\lambda_2,...,\lambda_{d^{\prime}}$和$\boldsymbol{w}_{1}, \boldsymbol{w}_{2}, \ldots, \boldsymbol{w}_{d^{\prime}}$分别为矩阵$\mathbf X\mathbf X^{\mathrm{T}}$的前$d^{\prime}$个最大的特征值和单位特征向量就能使得目标函数达到最优值。

另一篇博客给出更通俗更详细的理解：[主成分分析解析（基于最大方差理论）](http://blog.csdn.net/u011826404/article/details/57472730)

## **10.4 核化线性降维**

说起机器学习你中有我/我中有你/水乳相融...在这里能够得到很好的体现。正如SVM在处理非线性可分时，通过引入核函数将样本投影到高维特征空间，接着在高维空间再对样本点使用超平面划分。这里也是相同的问题：若我们的样本数据点本身就不是线性分布，那还如何使用一个超平面去近似表达呢？因此也就引入了核函数，**即先将样本映射到高维空间，再在高维空间中使用线性降维的方法**。下面主要介绍**核化主成分分析（KPCA）**的思想。

![image-20200911073136808](media\Chapter10_降维与度量学习\image-20200911073136808.png)

若核函数的形式已知，即我们知道如何将低维的坐标变换为高维坐标，且在高维空间里样本是线性可分的，这时我们只需先将数据映射到高维特征空间，再在高维空间中运用PCA即可，即：

$$
\left(\sum_{i=1}^{m} z_{i} z_{i}^{\mathrm{T}}\right) \mathbf{W}=\lambda \mathbf{W}
$$
进一步变形
$$
\begin{aligned}
\mathbf{W} &=\frac{1}{\lambda}\left(\sum_{i=1}^{m} z_{i} z_{i}^{\mathrm{T}}\right) \mathbf{W}=\sum_{i=1}^{m} z_{i} \frac{\boldsymbol{z}_{i}^{\mathrm{T}} \mathbf{W}}{\lambda} \\
&=\sum_{i=1}^{m} \boldsymbol{z}_{i} \boldsymbol{\alpha}_{i}
\end{aligned}
$$
然后求出核矩阵的表达

### 10.24

$$
\mathbf{K}\boldsymbol{\alpha}^j=\lambda_j\boldsymbol{\alpha}^j
$$

[推导]：已知$\boldsymbol z_i=\phi(\boldsymbol x_i)$，类比$\mathbf{X}=\{\boldsymbol x_1,\boldsymbol x_2,...,\boldsymbol x_m\}$可以构造$\mathbf{Z}=\{\boldsymbol z_1,\boldsymbol z_2,...,\boldsymbol z_m\}$，所以公式(10.21)可变换为
$$
\left(\sum_{i=1}^{m} \phi(\boldsymbol{x}_{i}) \phi(\boldsymbol{x}_{i})^{\mathrm{T}}\right)\boldsymbol w_j=\left(\sum_{i=1}^{m} \boldsymbol z_i \boldsymbol z_i^{\mathrm{T}}\right)\boldsymbol w_j=\mathbf{Z}\mathbf{Z}^{\mathrm{T}}\boldsymbol w_j=\lambda_j\boldsymbol w_j
$$
又由公式(10.22)可知
$$
\boldsymbol w_j=\sum_{i=1}^{m} \phi\left(\boldsymbol{x}_{i}\right) \alpha_{i}^j=\sum_{i=1}^{m} \boldsymbol z_i \alpha_{i}^j=\mathbf{Z}\boldsymbol{\alpha}^j
$$
其中，$\boldsymbol{\alpha}^j=(\alpha_{1}^j;\alpha_{2}^j;...;\alpha_{m}^j)\in \mathbb{R}^{m \times 1} $。所以公式(10.21)可以进一步变换为
$$
\begin{aligned}
\mathbf{Z}\mathbf{Z}^{\mathrm{T}}\mathbf{Z}\boldsymbol{\alpha}^j&=\lambda_j\mathbf{Z}\boldsymbol{\alpha}^j \\
\mathbf{Z}\mathbf{Z}^{\mathrm{T}}\mathbf{Z}\boldsymbol{\alpha}^j&=\mathbf{Z}\lambda_j\boldsymbol{\alpha}^j
\end{aligned}
$$
由于此时的目标是要求出$\boldsymbol w_j$，也就等价于要求出满足上式的$\boldsymbol{\alpha}^j$，显然，此时满足$\mathbf{Z}^{\mathrm{T}}\mathbf{Z}\boldsymbol{\alpha}^j=\lambda_j\boldsymbol{\alpha}^j $的$\boldsymbol{\alpha}^j$一定满足上式，所以问题转化为了求解满足下式的$\boldsymbol{\alpha}^j$：
$$
\mathbf{Z}^{\mathrm{T}}\mathbf{Z}\boldsymbol{\alpha}^j=\lambda_j\boldsymbol{\alpha}^j
$$


令$\mathbf{Z}^{\mathrm{T}}\mathbf{Z}=\mathbf{K}$，那么上式可化为
$$
\mathbf{K}\boldsymbol{\alpha}^j=\lambda_j\boldsymbol{\alpha}^j
$$


此式即为公式(10.24)，其中矩阵$\mathbf{K}$的第i行第j列的元素$(\mathbf{K})_{ij}=\boldsymbol z_i^{\mathrm{T}}\boldsymbol z_j=\phi(\boldsymbol x_i)^{\mathrm{T}}\phi(\boldsymbol x_j)=\kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)$

化简到最后一步，发现结果十分的美妙，只需对核矩阵K进行特征分解，便可以得出投影向量wi对应的系数向量α，因此选取特征值前d'大对应的特征向量便是d'个系数向量。这时对于需要降维的样本点，只需按照以下步骤便可以求出其降维后的坐标。可以看出：KPCA在计算降维后的坐标表示时，需要与所有样本点计算核函数值并求和，因此该算法的计算开销十分大。

$$
\begin{aligned}
z_{j} &=\boldsymbol{w}_{j}^{\mathrm{T}} \phi(\boldsymbol{x})=\sum_{i=1}^{m} \alpha_{i}^{j} \phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}} \phi(\boldsymbol{x}) \\
&=\sum_{i=1}^{m} \alpha_{i}^{j} \kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}\right)
\end{aligned}
$$

## **10.5 流形学习**

**流形学习（manifold learning）是一种借助拓扑流形概念的降维方法**，**流形是指在局部与欧式空间同胚的空间**，即在局部与欧式空间具有相同的性质，能用欧氏距离计算样本之间的距离。这样即使高维空间的分布十分复杂，但是在局部上依然满足欧式空间的性质，基于流形学习的降维正是这种**“邻域保持”**的思想。其中**等度量映射（Isomap）试图在降维前后保持邻域内样本之间的距离，而局部线性嵌入（LLE）则是保持邻域内样本之间的线性关系**，下面将分别对这两种著名的流行学习方法进行介绍。

### **10.5.1 等度量映射（Isomap）**

等度量映射的基本出发点是：高维空间中的直线距离具有误导性，因为有时高维空间中的直线距离在低维空间中是不可达的。**因此利用流形在局部上与欧式空间同胚的性质，可以使用近邻距离来逼近测地线距离**，即对于一个样本点，它与近邻内的样本点之间是可达的，且距离使用欧式距离计算，这样整个样本空间就形成了一张近邻图，高维空间中两个样本之间的距离就转为最短路径问题。可采用著名的**Dijkstra算法**或**Floyd算法**计算最短距离，得到高维空间中任意两点之间的距离后便可以使用MDS算法来其计算低维空间中的坐标。所以流形学习的核心就是怎样利用局部等效欧式距离的性质，来构建高维空间里的真实距离。

![13.png](media\Chapter10_降维与度量学习\5bc851b731a1e.png)

Isomap算法流程如下图：

![14.png](media\Chapter10_降维与度量学习\5bc851b6c7e37.png)

对于近邻图的构建，常用的有两种方法：**一种是指定近邻点个数**，像kNN一样选取k个最近的邻居；**另一种是指定邻域半径**，距离小于该阈值的被认为是它的近邻点。但两种方法均会出现下面的问题：

* 若**邻域范围指定过大，则会造成“短路问题”**，即本身距离很远却成了近邻，将距离近的那些样本扼杀在摇篮。
* 若**邻域范围指定过小，则会造成“断路问题”**，即有些样本点无法可达了，整个世界村被划分为互不可达的小部落。

从MDS算法的描述中我们可以知道：MDS先求出了低维空间的内积矩阵B，接着使用特征值分解计算出了样本在低维空间中的坐标，但是并没有给出通用的投影向量w，因此对于需要降维的新样本无从下手。这个问题的常用解决方案，是将训练样本的高维空间坐标作为输入、低维空间坐标作为输出，训练一个回归学习器来对新样本的低维空间坐标进行预测.这显然仅是一个权宜之计，但目前似乎并没有更好的办法.  

### **10.5.2 局部线性嵌入(LLE)**

不同于Isomap算法去保持邻域距离，LLE算法试图去保持邻域内的线性关系，假定样本xi的坐标可以通过它的邻域样本线性表出：

![15.png](media\Chapter10_降维与度量学习\5bc851b64236f.png)

![16.png](media\Chapter10_降维与度量学习\5bc851b6a7b9a.png)

LLE算法分为两步走，第一步先为每个样本 xi 找到其近邻下标集合 Qi ，根据近邻关系计算出所有样本的邻域重构系数w：

![17.png](media\Chapter10_降维与度量学习\5bc851b662815.png)

接着根据邻域重构系数不变，去求解低维坐标：

![18.png](media\Chapter10_降维与度量学习\5bc851b648b98.png)

可以看到这两步利用的优化函数是同形的，只是每一步优化的对象不同。这样利用矩阵M，优化问题可以重写为：

![19.png](media\Chapter10_降维与度量学习\5bc851b6948d7.png)

M特征值分解后最小的d'个特征值对应的特征向量组成Z。

### 10.28

$$
w_{i j}=\frac{\sum\limits_{k \in Q_{i}} C_{j k}^{-1}}{\sum\limits_{l, s \in Q_{i}} C_{l s}^{-1}}
$$

[推导]：由书中上下文可知，式(10.28)是如下优化问题的解。
$$
\begin{aligned} 
\min _{\boldsymbol{w}_{1}, \boldsymbol{w}_{2}, \ldots, \boldsymbol{w}_{m}} & \sum_{i=1}^{m}\left\|\boldsymbol{x}_{i}-\sum_{j \in Q_{i}} w_{i j} \boldsymbol{x}_{j}\right\|_{2}^{2} \\ 
\text { s.t. } & \sum_{j \in Q_{i}} w_{i j}=1 
\end{aligned}
$$


若令$\boldsymbol{x}_{i}\in \mathbb{R}^{d\times 1},Q_i=\{q_i^1,q_i^2,...,q_i^n\}$，则上述优化问题的目标函数可以进行如下恒等变形。第一步的sum(w)=1，所以恒等。
$$
\begin{aligned} 
\sum_{i=1}^{m}\left\|\boldsymbol{x}_{i}-\sum_{j \in Q_{i}} w_{i j} \boldsymbol{x}_{j}\right\|_{2}^{2}&=\sum_{i=1}^{m}\left\|\sum_{j \in Q_{i}} w_{i j} \boldsymbol{x}_{i}-\sum_{j \in Q_{i}} w_{i j} \boldsymbol{x}_{j}\right\|_{2}^{2} \\ 
&=\sum_{i=1}^{m}\left\|\sum_{j \in Q_{i}} w_{i j}(\boldsymbol{x}_{i}-\boldsymbol{x}_{j}) \right\|_{2}^{2} \\ 
&=\sum_{i=1}^{m}\left\|\mathbf{X}_i\boldsymbol{w_i} \right\|_{2}^{2} \\
&=\sum_{i=1}^{m}\boldsymbol{w_i}^{\mathrm{T}}\mathbf{X}_i^{\mathrm{T}}\mathbf{X}_i\boldsymbol{w_i} \\ 
\end{aligned}
$$


其中$\boldsymbol{w_i}=(w_{iq_i^1},w_{iq_i^2},...,w_{iq_i^n})\in \mathbb{R}^{n\times 1}$，$\mathbf{X}_i=\left( \boldsymbol{x}_{i}-\boldsymbol{x}_{q_i^1}, \boldsymbol{x}_{i}-\boldsymbol{x}_{q_i^2},...,\boldsymbol{x}_{i}-\boldsymbol{x}_{q_i^n}\right)\in \mathbb{R}^{d\times n}$。同理，约束条件也可以进行如下恒等变形
$$
\sum_{j \in Q_{i}} w_{i j}=\boldsymbol{w_i}^{\mathrm{T}}\boldsymbol{I}=1 
$$


其中$\boldsymbol{I}=(1,1,...,1)\in \mathbb{R}^{n\times 1}$为$n$行1列的单位向量。因此，上述优化问题可以重写为
$$
\begin{aligned} 
\min _{\boldsymbol{w}_{1}, \boldsymbol{w}_{2}, \ldots, \boldsymbol{w}_{m}} & \sum_{i=1}^{m}\boldsymbol{w_i}^{\mathrm{T}}\mathbf{X}_i^{\mathrm{T}}\mathbf{X}_i\boldsymbol{w_i} \\ 
\text { s.t. } & \boldsymbol{w_i}^{\mathrm{T}}\boldsymbol{I}=1
\end{aligned}
$$


显然，此问题为带约束的优化问题，因此可以考虑使用拉格朗日乘子法来进行求解。由拉格朗日乘子法可得此优化问题的拉格朗日函数为
$$
L(\boldsymbol{w}_{1}, \boldsymbol{w}_{2}, \ldots, \boldsymbol{w}_{m},\lambda)=\sum_{i=1}^{m}\boldsymbol{w_i}^{\mathrm{T}}\mathbf{X}_i^{\mathrm{T}}\mathbf{X}_i\boldsymbol{w_i}+\lambda\left(\boldsymbol{w_i}^{\mathrm{T}}\boldsymbol{I}-1\right)
$$


对拉格朗日函数关于$\boldsymbol{w_i}$求偏导并令其等于0可得
$$
\begin{aligned} 
\cfrac{\partial L(\boldsymbol{w}_{1}, \boldsymbol{w}_{2}, \ldots, \boldsymbol{w}_{m},\lambda)}{\partial \boldsymbol{w_i}}&=\cfrac{\partial \left[\sum_{i=1}^{m}\boldsymbol{w_i}^{\mathrm{T}}\mathbf{X}_i^{\mathrm{T}}\mathbf{X}_i\boldsymbol{w_i}+\lambda\left(\boldsymbol{w_i}^{\mathrm{T}}\boldsymbol{I}-1\right)\right]}{\partial \boldsymbol{w_i}}=0\\
&=\cfrac{\partial \left[\boldsymbol{w_i}^{\mathrm{T}}\mathbf{X}_i^{\mathrm{T}}\mathbf{X}_i\boldsymbol{w_i}+\lambda\left(\boldsymbol{w_i}^{\mathrm{T}}\boldsymbol{I}-1\right)\right]}{\partial \boldsymbol{w_i}}=0\\
\end{aligned}
$$


又由矩阵微分公式$\cfrac{\partial \boldsymbol{x}^{T} \mathbf{B} \boldsymbol{x}}{\partial \boldsymbol{x}}=\left(\mathbf{B}+\mathbf{B}^{\mathrm{T}}\right) \boldsymbol{x},\cfrac{\partial \boldsymbol{x}^{T} \boldsymbol{a}}{\partial \boldsymbol{x}}=\boldsymbol{a}$可得
$$
\begin{aligned}
\cfrac{\partial \left[\boldsymbol{w_i}^{\mathrm{T}}\mathbf{X}_i^{\mathrm{T}}\mathbf{X}_i\boldsymbol{w_i}+\lambda\left(\boldsymbol{w_i}^{\mathrm{T}}\boldsymbol{I}-1\right)\right]}{\partial \boldsymbol{w_i}}&=2\mathbf{X}_i^{\mathrm{T}}\mathbf{X}_i\boldsymbol{w_i}+\lambda \boldsymbol{I}=0\\
\mathbf{X}_i^{\mathrm{T}}\mathbf{X}_i\boldsymbol{w_i}&=-\frac{1}{2}\lambda \boldsymbol{I}
\end{aligned}
$$
若$\mathbf{X}_i^{\mathrm{T}}\mathbf{X}_i$可逆，则
$$
\boldsymbol{w_i}=-\frac{1}{2}\lambda(\mathbf{X}_i^{\mathrm{T}}\mathbf{X}_i)^{-1}\boldsymbol{I}
$$


又因为$\boldsymbol{w_i}^{\mathrm{T}}\boldsymbol{I}=\boldsymbol{I}^{\mathrm{T}}\boldsymbol{w_i}=1$，则上式两边同时左乘$\boldsymbol{I}^{\mathrm{T}}$可得
$$
\begin{aligned}
\boldsymbol{I}^{\mathrm{T}}\boldsymbol{w_i}&=-\frac{1}{2}\lambda\boldsymbol{I}^{\mathrm{T}}(\mathbf{X}_i^{\mathrm{T}}\mathbf{X}_i)^{-1}\boldsymbol{I}=1\\
-\frac{1}{2}\lambda&=\cfrac{1}{\boldsymbol{I}^{\mathrm{T}}(\mathbf{X}_i^{\mathrm{T}}\mathbf{X}_i)^{-1}\boldsymbol{I}}
\end{aligned}
$$
将其代回$\boldsymbol{w_i}=-\frac{1}{2}\lambda(\mathbf{X}_i^{\mathrm{T}}\mathbf{X}_i)^{-1}\boldsymbol{I}$即可解得
$$
\boldsymbol{w_i}=\cfrac{(\mathbf{X}_i^{\mathrm{T}}\mathbf{X}_i)^{-1}\boldsymbol{I}}{\boldsymbol{I}^{\mathrm{T}}(\mathbf{X}_i^{\mathrm{T}}\mathbf{X}_i)^{-1}\boldsymbol{I}}
$$


若令矩阵$(\mathbf{X}_i^{\mathrm{T}}\mathbf{X}_i)^{-1}$第$j$行第$k$列的元素为$C_{jk}^{-1}$，则
$$
w_{ij}=w_{i q_i^j}=\frac{\sum\limits_{k \in Q_{i}} C_{j k}^{-1}}{\sum\limits_{l, s \in Q_{i}} C_{l s}^{-1}}
$$


此即为公式(10.28)。显然，若$\mathbf{X}_i^{\mathrm{T}}\mathbf{X}_i$可逆，此优化问题即为凸优化问题，且此时用拉格朗日乘子法求得的$\boldsymbol{w_i}$为全局最优解。

### 10.31

$$
\begin{aligned}
&\min\limits_{\boldsymbol Z}tr(\boldsymbol Z \boldsymbol M \boldsymbol Z^T)\\
&s.t. \boldsymbol Z^T\boldsymbol Z=\boldsymbol I. 
\end{aligned}
$$

[推导]：
$$
\begin{aligned}
\min\limits_{\boldsymbol Z}\sum^m_{i=1}\| \boldsymbol z_i-\sum_{j \in Q_i}w_{ij}\boldsymbol z_j \|^2_2&=\sum^m_{i=1}\|\boldsymbol Z\boldsymbol I_i-\boldsymbol Z\boldsymbol W_i\|^2_2\\
&=\sum^m_{i=1}\|\boldsymbol Z(\boldsymbol I_i-\boldsymbol W_i)\|^2_2\\
&=\sum^m_{i=1}(\boldsymbol Z(\boldsymbol I_i-\boldsymbol W_i))^T\boldsymbol Z(\boldsymbol I_i-\boldsymbol W_i)\\
&=\sum^m_{i=1}(\boldsymbol I_i-\boldsymbol W_i)^T\boldsymbol Z^T\boldsymbol Z(\boldsymbol I_i-\boldsymbol W_i)\\
&=tr((\boldsymbol I-\boldsymbol W)^T\boldsymbol Z^T\boldsymbol Z(\boldsymbol I-\boldsymbol W))\\
&=tr(\boldsymbol Z(\boldsymbol I-\boldsymbol W)(\boldsymbol I-\boldsymbol W)^T\boldsymbol Z^T)\\
&=tr(\boldsymbol Z\boldsymbol M\boldsymbol Z^T)
\end{aligned}
$$

其中，$\boldsymbol M=(\boldsymbol I-\boldsymbol W)(\boldsymbol I-\boldsymbol W)^T$。
[解析]：约束条件$\boldsymbol Z^T\boldsymbol Z=\boldsymbol I$是为了得到标准化（标准正交空间）的低维数据。

LLE算法的具体流程如下图所示：

![20.png](media\Chapter10_降维与度量学习\5bc851b757d8c.png)

## **10.6 度量学习**

本篇一开始就提到维数灾难，即在高维空间进行机器学习任务遇到样本稀疏、距离难计算等诸多的问题，因此前面讨论的降维方法都试图将原空间投影到一个合适的低维空间中，接着在低维空间进行学习任务从而产生较好的性能。事实上，不管高维空间还是低维空间都潜在对应着一个距离度量，那可不可以直接学习出一个距离度量来等效降维呢？例如：**咋们就按照降维后的方式来进行距离的计算，这便是度量学习的初衷**。

**首先要学习出距离度量必须先定义一个合适的距离度量形式**。对两个样本xi与xj，它们之间的平方欧式距离为：

![21.png](media\Chapter10_降维与度量学习\5bc851d3ca3d5.png)

若各个属性重要程度不一样即都有一个权重，则得到加权的平方欧式距离：

![22.png](media\Chapter10_降维与度量学习\5bc851d3d82c5.png)

此时各个属性之间都是相互独立无关的，但现实中往往会存在属性之间有关联的情形，例如：身高和体重，一般人越高，体重也会重一些，他们之间存在较大的相关性。这样计算距离就不能分属性单独计算，于是就引入经典的**马氏距离(Mahalanobis distance)**:

![23.png](media\Chapter10_降维与度量学习\5bc851d3dc303.png)

标准的马氏距离中M是协方差矩阵的逆，马氏距离是一种考虑属性之间相关性且尺度无关（即无须去量纲）的距离度量。

![24.png](media\Chapter10_降维与度量学习\5bc851d3e17c0.png)

矩阵M也称为“度量矩阵”，为保证距离度量的非负性与对称性，M必须为(半)正定对称矩阵，这样就为度量学习定义好了距离度量的形式，换句话说：**度量学习便是对度量矩阵进行学习**。现在来回想一下前面我们接触的机器学习不难发现：**机器学习算法几乎都是在优化目标函数，从而求解目标函数中的参数**。同样对于度量学习，也需要设置一个优化目标，书中简要介绍了错误率和相似性两种优化目标，此处限于篇幅不进行展开。

在此，降维和度量学习就介绍完毕。**降维是将原高维空间嵌入到一个合适的低维子空间中，接着在低维空间中进行学习任务；度量学习则是试图去学习出一个距离度量来等效降维的效果**，两者都是为了解决==维数灾难==带来的诸多问题。也许大家最后心存疑惑，那kNN呢？为什么一开头就说了kNN算法，但是好像和后面没有半毛钱关系？正是因为在降维算法中，低维子空间的维数d'通常都由人为指定，因此我们需要使用一些低开销的学习器来选取合适的d'，kNN这家伙懒到家了根本无心学习，在训练阶段开销为零，测试阶段也只是遍历计算了距离，因此拿kNN来进行交叉验证就十分有优势了~同时降维后样本密度增大同时距离计算变易，更为kNN来展示它独特的十八般手艺提供了用武之地。

## 参考文献

<span id="ref1">[1][How to set up Lagrangian optimization with matrix constrains](https://math.stackexchange.com/questions/1104376/how-to-set-up-lagrangian-optimization-with-matrix-constrains)</span><br>
<span id="ref2">[2][Frobenius inner product](https://en.wikipedia.org/wiki/Frobenius_inner_product)</span>



