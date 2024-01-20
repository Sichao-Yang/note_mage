上篇主要介绍了一种机器学习的通用框架--集成学习方法，首先从准确性和差异性两个重要概念引出集成学习"**好而不同**"的四字真言，接着介绍了现阶段主流的三种集成学习方法：AdaBoost、Bagging及Random Forest，AdaBoost采用最小化指数损失函数迭代式更新样本分布权重和计算基学习器权重，Bagging通过自助采样引入样本扰动增加了基学习器之间的差异性，随机森林则进一步引入了属性扰动，最后简单概述了集成模型中的三类结合策略：平均法、投票法及学习法，其中Stacking是学习法的典型代表。本篇将讨论无监督学习中应用最为广泛的学习算法--聚类。

# **9. 聚类算法**

聚类是一种经典的**无监督学习**方法，**无监督学习的目标是通过对无标记训练样本的学习，发掘和揭示数据集本身潜在的结构与规律**，即不依赖于训练数据集的类标记信息。聚类则是试图将数据集的样本划分为若干个互不相交的类簇，从而每个簇对应一个潜在的类别。

聚类直观上来说是将相似的样本聚在一起，从而形成一个**类簇（cluster）**。那首先的问题是如何来**度量相似性**（similarity measure）呢？这便是**距离度量**，在生活中我们说差别小则相似，对应到多维样本，每个样本可以对应于高维空间中的一个数据点，若它们的距离相近，我们便可以称它们相似。那接着如何来评价聚类结果的好坏呢？这便是**性能度量**，性能度量为评价聚类结果的好坏提供了一系列有效性指标。

## **9.1 距离度量**

谈及距离度量，最熟悉的莫过于欧式距离了，从年头一直用到年尾的距离计算公式：即对应属性之间相减的平方和再开根号。度量距离还有其它的很多经典方法，通常它们需要满足一些基本性质：

![1.png](media\Chapter9_聚类\5bc84ed4c0390.png)

最常用的距离度量方法是**"闵可夫斯基距离"（Minkowski distance)**：

![2.png](media\Chapter9_聚类\5bc84ed49e31f.png)

当p=1时，闵可夫斯基距离即**曼哈顿距离（Manhattan distance）**：

![3.png](media\Chapter9_聚类\5bc84ed49c31f.png)

当p=2时，闵可夫斯基距离即**欧氏距离（Euclidean distance）**：

![4.png](media\Chapter9_聚类\5bc84ed497613.png)

我们知道属性分为两种：**连续属性**和**离散属性**（有限个取值）。对于连续值的属性，一般都可以被学习器所用，有时会根据具体的情形作相应的预处理，例如：归一化等；而对于离散值的属性，需要作下面进一步的处理：

> 若属性值之间**存在序关系**，则可以将其转化为连续值，例如：身高属性"高""中等""矮"，可转化为{1, 0.5, 0}。
> 若属性值之间**不存在序关系**，则通常将其转化为向量的形式，例如：性别属性"男""女"，可转化为{（1,0），（0,1）}。

在进行距离度量时，易知**连续属性和存在序关系的离散属性都可以直接参与计算**，因为它们都可以反映一种程度，我们称其为"**有序属性**"；而对于不存在序关系的离散属性，我们称其为："**无序属性**"，显然无序属性再使用闵可夫斯基距离就行不通了。

**对于无序属性，我们一般采用VDM进行距离的计算**，例如：对于离散属性的两个取值a和b，定义：

![5.png](media\Chapter9_聚类\5bc84ed4e9560.png)

于是，在计算两个样本之间的距离时，我们可以将闵可夫斯基距离和VDM混合在一起进行计算：

![6.png](media\Chapter9_聚类\5bc84ed507bc7.png)

若我们定义的距离计算方法是用来度量相似性，例如下面将要讨论的聚类问题，即距离越小，相似性越大，反之距离越大，相似性越小。这时距离的度量方法并不一定需要满足前面所说的四个基本性质，这样的方法称为：**非度量距离（non-metric distance）**。

## **9.2 性能度量**

由于聚类算法不依赖于样本的真实类标，就不能像监督学习的分类那般，通过计算分对分错（即精确度或错误率）来评价学习器的好坏或作为学习过程中的优化目标。一般聚类有两类性能度量指标：**外部指标**和**内部指标**。

* 外部指标

即将聚类结果与某个参考模型的结果进行比较，**以参考模型的输出作为标准，来评价聚类好坏**。假设聚类给出的结果为λ，参考模型给出的结果是λ*，则我们将样本进行两两配对，定义：

![7.png](media\Chapter9_聚类\5bc84ed59160e.png)

显然a和b代表着聚类结果好坏的正能量，b和c则表示参考结果和聚类结果相矛盾，基于这四个值可以导出以下常用的外部评价指标：

![8.png](media\Chapter9_聚类\5bc84ed587438.png)

### 9.5

$$
\mathrm{JC}=\frac{a}{a+b+c}
$$

[解析]：给定两个集合$A$和$B$，则Jaccard系数定义为如下公式

$$
\mathrm{JC}=\frac{|A\bigcap B|}{|A\bigcup B|}=\frac{|A\bigcap B|}{|A|+|B|-|A\bigcap B|}
$$
Jaccard系数可以用来描述两个集合的相似程度。

推论：假设全集$U$共有$n$个元素，且$A\subseteq U$，$B\subseteq U$，则每一个元素的位置共有四种情况：

1、元素同时在集合$A$和$B$中，这样的元素个数记为$M_{11}$；

2、元素出现在集合$A$中，但没有出现在集合$B$中，这样的元素个数记为$M_{10}$；

3、元素没有出现在集合$A$中，但出现在集合$B$中，这样的元素个数记为$M_{01}$；

4、元素既没有出现在集合$A$中，也没有出现在集合$B$中，这样的元素个数记为$M_{00}$。

根据Jaccard系数的定义，此时的Jaccard系数为如下公式
$$
\mathrm{JC}=\frac{M_{11}}{M_{11}+M_{10}+M_{01}}
$$
由于聚类属于无监督学习，事先并不知道聚类后样本所属类别的类别标记所代表的意义，即便参考模型的类别标记意义是已知的，我们也无法知道聚类后的类别标记与参考模型的类别标记是如何对应的，况且聚类后的类别总数与参考模型的类别总数还可能不一样，因此只用单个样本无法衡量聚类性能的好坏。

由于外部指标的基本思想就是以参考模型的类别划分为参照，因此如果某一个样本对中的两个样本在聚类结果中同属于一个类，在参考模型中也同属于一个类，或者这两个样本在聚类结果中不同属于一个类，在参考模型中也不同属于一个类，那么对于这两个样本来说这是一个好的聚类结果。

总的来说所有样本对中的两个样本共存在四种情况：
1、样本对中的两个样本在聚类结果中属于同一个类，在参考模型中也属于同一个类；
2、样本对中的两个样本在聚类结果中属于同一个类，在参考模型中不属于同一个类；
3、样本对中的两个样本在聚类结果中不属于同一个类，在参考模型中属于同一个类；
4、样本对中的两个样本在聚类结果中不属于同一个类，在参考模型中也不属于同一个类。

综上所述，即所有样本对存在着书中公式(9.1)-(9.4)的四种情况，现在假设集合$A$中存放着两个样本都同属于聚类结果的同一个类的样本对，即$A=SS\bigcup SD$，集合$B$中存放着两个样本都同属于参考模型的同一个类的样本对，即$B=SS\bigcup DS$，那么根据Jaccard系数的定义有：
$$
\mathrm{JC}=\frac{|A\bigcap B|}{|A\bigcup B|}=\frac{|SS|}{|SS\bigcup SD\bigcup DS|}=\frac{a}{a+b+c}
$$
也可直接将书中公式(9.1)-(9.4)的四种情况类比推论，即$M_{11}=a$，$M_{10}=b$，$M_{01}=c$，所以
$$
\mathrm{JC}=\frac{M_{11}}{M_{11}+M_{10}+M_{01}}=\frac{a}{a+b+c}
$$

### 9.6

$$
\mathrm{FMI}=\sqrt{\frac{a}{a+b}\cdot \frac{a}{a+c}}
$$

[解析]：其中$\frac{a}{a+b}$和$\frac{a}{a+c}$为Wallace提出的两个非对称指标，$a$代表两个样本在聚类结果和参考模型中均属于同一类的样本对的个数，$a+b$代表两个样本在聚类结果中属于同一类的样本对的个数，$a+c$代表两个样本在参考模型中属于同一类的样本对的个数，这两个非对称指标均可理解为样本对中的两个样本在聚类结果和参考模型中均属于同一类的概率。由于指标的非对称性，这两个概率值往往不一样，因此Fowlkes和Mallows提出利用几何平均数将这两个非对称指标转化为一个对称指标，即Fowlkes and Mallows Index, FMI。

### 9.7

$$
\mathrm{RI}=\frac{2(a+d)}{m(m-1)}
$$

[解析]：Rand Index定义如下：
$$
\mathrm{RI}=\frac{a+d}{a+b+c+d}=\frac{a+d}{m(m-1)/2}=\frac{2(a+d)}{m(m-1)}
$$
其可以理解为两个样本都属于聚类结果和参考模型中的同一类的样本对的个数与两个样本都分别不属于聚类结果和参考模型中的同一类的样本对的个数的总和在所有样本对中出现的频率，可以简单理解为聚类结果与参考模型的一致性。

### 9.8

$$
\operatorname{avg}(C)=\frac{2}{|C|(|C|-1)} \sum_{1 \leqslant i<j \leqslant|C|} \operatorname{dist}\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)
$$

[解析]：簇内距离的定义式：求和号左边是$(x_i, x_j)$组合个数的倒数，求和号右边是这些组合的距离和，所以两者相乘定义为平均距离。

* 内部指标

内部指标即不依赖任何外部模型，直接对聚类的结果进行评估，聚类的目的是想将那些相似的样本尽可能聚在一起，不相似的样本尽可能分开，直观来说：**簇内高内聚紧紧抱团，簇间低耦合老死不相往来**。定义：

![9.png](media\Chapter9_聚类\5bc84ed581852.png)

基于上面的四个距离，可以导出下面这些常用的内部评价指标：

![10.png](media\Chapter9_聚类\5bc84ed582854.png)

## **9.3 原型聚类**

原型聚类即"**基于原型的聚类**"（prototype-based clustering），原型表示模板的意思，就是通过参考一个模板向量或模板分布的方式来完成聚类的过程，常见的K-Means便是基于簇中心来实现聚类，混合高斯聚类则是基于簇分布来实现聚类。

### **9.3.1 K-Means**

K-Means的思想十分简单，**首先随机指定类中心，根据样本与类中心的远近划分类簇，接着重新计算类中心，迭代直至收敛**。但是其中迭代的过程并不是主观地想象得出，事实上，若将样本的类别看做为"隐变量"（latent variable），类中心看作样本的分布参数，这一过程正是通过**EM算法**的两步走策略而计算出，其根本的目的是为了最小化平方误差函数E：

![11.png](media\Chapter9_聚类\5bc84fb82b5d3.png)

K-Means的算法流程如下所示：

![12.png](media\Chapter9_聚类\5bc84fb9c0817.png)

### **9.3.2 学习向量量化（LVQ）**

LVQ也是基于原型的聚类算法，与K-Means不同的是，**LVQ使用样本真实类标记辅助聚类**，首先LVQ根据样本的类标记，从各类中分别随机选出一个样本作为该类簇的原型，从而组成了一个**原型特征向量组**，接着从样本集中随机挑选一个样本，计算其与原型向量组中每个向量的距离，并选取距离最小的原型向量所在的类簇作为它的划分结果，再与真实类标比较。

> **若划分结果正确，则对应原型向量向这个样本靠近一些**
> **若划分结果不正确，则对应原型向量向这个样本远离一些**

LVQ算法的流程如下所示：

![13.png](media\Chapter9_聚类\5bc84fb9d59f2.png)

### **9.3.3 高斯混合聚类**

现在可以看出K-Means与LVQ都试图以类中心作为原型指导聚类，高斯混合聚类则采用高斯分布来描述原型。现假设**每个类簇中的样本都服从一个多维高斯分布，那么空间中的样本可以看作由k个多维高斯分布混合而成**。

对于多维高斯分布，其概率密度函数如下所示：

![14.png](media\Chapter9_聚类\5bc84fb870d98.png)

其中u表示均值向量，∑表示协方差矩阵，可以看出一个多维高斯分布完全由这两个参数所确定。接着定义高斯混合分布为：

![15.png](media\Chapter9_聚类\5bc84fb876794.png)

α称为混合系数，这样空间中样本的采集过程则可以抽象为：**（1）先选择一个类簇（高斯分布），（2）再根据对应高斯分布的密度函数进行采样**，这时候贝叶斯公式又能大展身手了：

![16.png](media\Chapter9_聚类\5bc84fb9191d9.png)

此时只需要选择PM最大时的类簇并将该样本划分到其中，看到这里很容易发现：这和那个传说中的贝叶斯分类不是神似吗，都是通过贝叶斯公式展开，然后计算类先验概率和类条件概率。但遗憾的是：**这里没有真实类标信息，对于类条件概率，并不能像贝叶斯分类那样通过最大似然法美好地计算出来**，因为这里的样本可能属于所有的类簇，这里的似然函数变为：

![17.png](media\Chapter9_聚类\5bc84fb871d4a.png)

可以看出：简单的最大似然法根本无法求出所有的参数，这样PM也就没法计算。**这里就要召唤出之前的EM大法，首先对高斯分布的参数及混合系数进行随机初始化，计算出各个PM（即γji，第i个样本属于j类），再最大化似然函数（即LL（D）分别对α、u和∑求偏导 ），对参数进行迭代更新**。

![18.png](media\Chapter9_聚类\5bc84fb8a6f32.png)

### 9.33

$$
\sum_{j=1}^{m} \frac{\alpha_{i} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)}{\sum_{l=1}^{k} \alpha_{l} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{l}, \boldsymbol{\Sigma}_{l}\right)}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)=0
$$

[推导]：根据公式(9.28)可知：
$$
p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)=\frac{1}{(2 \pi)^{\frac{n}{2}}\left|\boldsymbol{\Sigma}_{i}\right|^{\frac{1}{2}}} \exp \left(-\frac{1}{2}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{T} \boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\right)
$$
又根据公式(9.32)，由
$$
\frac{\partial L L(D)}{\partial \boldsymbol{\mu}_{i}}=\frac{\partial L L(D)}{\partial p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)} \cdot \frac{\partial p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)}{\partial \boldsymbol{\mu}_{i}}=0
$$
其中：
$$
\begin{aligned}
\frac{\partial L L(D)}{\partial p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \mathbf{\Sigma}_{i}\right)} &=\frac{\partial \sum_{j=1}^{m} \ln \left(\sum_{l=1}^{k} \alpha_{l} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{l}, \boldsymbol{\Sigma}_{l}\right)\right)}{\partial p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)} \\
&=\sum_{j=1}^{m} \frac{\partial \ln \left(\sum_{l=1}^{k} \alpha_{l} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{l}, \boldsymbol{\Sigma}_{l}\right)\right)}{\partial p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)} \\
&=\sum_{j=1}^{m} \frac{\alpha_{i}}{\sum_{l=1}^{k} \alpha_{l} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{l}, \boldsymbol{\Sigma}_{l}\right)}
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)}{\partial \boldsymbol{\mu}_{i}} &=\frac{\partial \frac{1}{(2 \pi)^{\frac{n}{2}}\left|\Sigma_{i}\right|^{\frac{1}{2}}} \exp\left({-\frac{1}{2}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{\top}\boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)}\right)}{\partial \boldsymbol{\mu}_{i}} \\
&=\frac{1}{(2 \pi)^{\frac{n}{2}}\left|\boldsymbol{\Sigma}_{i}\right|^{\frac{1}{2}}} \cdot \frac{\partial \exp\left({-\frac{1}{2}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{\top} \boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)}\right)}{\partial \boldsymbol{\mu}_{i}}\\
&=\frac{1}{(2 \pi)^{\frac{n}{2}}\left|\boldsymbol{\Sigma}_{i}\right|^{\frac{1}{2}}}\cdot \exp\left({-\frac{1}{2}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{\top} \boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)}\right) \cdot-\frac{1}{2} \frac{\partial\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{\top} \boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)}{\partial \boldsymbol{\mu}_{i}}\\
&=\frac{1}{(2 \pi)^{\frac{n}{2}}\left|\boldsymbol{\Sigma}_{i}\right|^{\frac{1}{2}}}\cdot \exp\left({-\frac{1}{2}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{\top} \boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)}\right) \cdot\boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\\
&=p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right) \cdot \boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)
\end{aligned}
$$

其中，由矩阵求导的法则$\frac{\partial \mathbf{a}^{T} \mathbf{X} \mathbf{a}}{\partial \mathbf{a}}=2\mathbf{X} \mathbf{a}$可得：
$$
\begin{aligned}
-\frac{1}{2} \frac{\partial\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{\top} \boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)}{\partial \boldsymbol{\mu}_{i}} &=\frac{1}{2} \cdot 2 \boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right) =\boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)
\end{aligned}
$$


因此有：
$$
\frac{\partial L L(D)}{\partial \boldsymbol{\mu}_{i}}=\sum_{j=1}^{m} \frac{\alpha_{i}}{\sum_{l=1}^{k} \alpha_{l} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{l}, \mathbf{\Sigma}_{l}\right)} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right) \cdot \boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)=0
$$

### 9.34

$$
\mu_{i}=\frac{\sum_{j=1}^{m} \gamma_{j i} \boldsymbol{x}_{j}}{\sum_{j=1}^{m} \gamma_{j i}}
$$



[推导]：由式9.30
$$
\gamma_{j i}=p_{\mathcal{M}}\left(z_{j}=i | \boldsymbol{x}_{j}\right)=\frac{\alpha_{i} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)}{\sum_{l=1}^{k} \alpha_{l} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{l}, \mathbf{\Sigma}_{l}\right)}
$$
带入9.33
$$
\sum_{j=1}^{m} \gamma_{j i}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)=0
$$
因此有
$$
\mu_{i}=\frac{\sum_{j=1}^{m} \gamma_{j i} \boldsymbol{x}_{j}}{\sum_{j=1}^{m} \gamma_{j i}}
$$

### 9.35

$$
\mathbf\Sigma_{i}=\cfrac{\sum_{j=1}^m\gamma_{ji}(\boldsymbol x_{j}-\boldsymbol \mu_{i})(\boldsymbol x_{j}-\boldsymbol\mu_{i})^T}{\sum_{j=1}^m\gamma_{ji}}
$$

[推导]：根据公式(9.28)可知：
$$
p(\boldsymbol x_{j}|\boldsymbol\mu_{i},\mathbf\Sigma_{i})=\cfrac{1}{(2\pi)^\frac{n}{2}\left| \mathbf\Sigma_{i}\right |^\frac{1}{2}}\exp\left({-\frac{1}{2}(\boldsymbol x_{j}-\boldsymbol\mu_{i})^T\mathbf\Sigma_{i}^{-1}(\boldsymbol x_{j}-\boldsymbol\mu_{i})}\right)
$$
又根据公式(9.32)，由
$$
\cfrac{\partial LL(D)}{\partial \mathbf\Sigma_{i}}=0
$$
可得
$$
\begin{aligned}
\cfrac {\partial LL(D)}{\partial\mathbf\Sigma_{i}}&=\cfrac {\partial}{\partial \mathbf\Sigma_{i}}\left[\sum_{j=1}^m\ln\Bigg(\sum_{i=1}^k \alpha_{i}\cdot p(\boldsymbol x_{j}|\boldsymbol\mu_{i},\mathbf\Sigma_{i})\Bigg)\right] \\
&=\sum_{j=1}^m\frac{\partial}{\partial\mathbf\Sigma_{i}}\left[\ln\Bigg(\sum_{i=1}^k \alpha_{i}\cdot p(\boldsymbol x_{j}|\boldsymbol\mu_{i},\mathbf\Sigma_{i})\Bigg)\right] \\
&=\sum_{j=1}^m\cfrac{\alpha_{i}\cdot \cfrac{\partial}{\partial\mathbf\Sigma_{i}}\left(p(\boldsymbol x_{j}|\boldsymbol\mu_{i},\mathbf\Sigma_{i})\right)}{\sum_{l=1}^k\alpha_{l}\cdot p(\boldsymbol x_{j}|\boldsymbol\mu_{l},\mathbf\Sigma_{l})} \\
\end{aligned}
$$

其中
$$
\begin{aligned}
\cfrac{\partial}{\partial\mathbf\Sigma_{i}}\left(p(\boldsymbol x_{j}|\boldsymbol\mu_{i},\mathbf\Sigma_{i})\right)&=\cfrac{\partial}{\partial\mathbf\Sigma_{i}}\left[\cfrac{1}{(2\pi)^\frac{n}{2}\left| \mathbf\Sigma_{i}\right |^\frac{1}{2}}\exp\left({-\frac{1}{2}(\boldsymbol x_{j}-\boldsymbol\mu_{i})^T\mathbf\Sigma_{i}^{-1}(\boldsymbol x_{j}-\boldsymbol\mu_{i})}\right)\right] \\
&=\cfrac{\partial}{\partial\mathbf\Sigma_{i}}\left\{\exp\left[\ln\left(\cfrac{1}{(2\pi)^\frac{n}{2}\left| \mathbf\Sigma_{i}\right |^\frac{1}{2}}\exp\left({-\frac{1}{2}(\boldsymbol x_{j}-\boldsymbol\mu_{i})^T\mathbf\Sigma_{i}^{-1}(\boldsymbol x_{j}-\boldsymbol\mu_{i})}\right)\right)\right]\right\} \\
&=p(\boldsymbol x_{j}|\boldsymbol\mu_{i},\mathbf\Sigma_{i})\cdot\cfrac{\partial}{\partial\mathbf\Sigma_{i}}\left[\ln\left(\cfrac{1}{(2\pi)^\frac{n}{2}\left| \mathbf\Sigma_{i}\right |^\frac{1}{2}}\exp\left({-\frac{1}{2}(\boldsymbol x_{j}-\boldsymbol\mu_{i})^T\mathbf\Sigma_{i}^{-1}(\boldsymbol x_{j}-\boldsymbol\mu_{i})}\right)\right)\right]\\
&=p(\boldsymbol x_{j}|\boldsymbol\mu_{i},\mathbf\Sigma_{i})\cdot\cfrac{\partial}{\partial\mathbf\Sigma_{i}}\left[\ln\cfrac{1}{(2\pi)^{\frac{n}{2}}}-\cfrac{1}{2}\ln{|\mathbf{\Sigma}_i|}-\frac{1}{2}(\boldsymbol x_j-\boldsymbol\mu_i)^T\mathbf{\Sigma}_i^{-1}(\boldsymbol x_j-\boldsymbol\mu_i)\right]\\
&=p(\boldsymbol x_{j}|\boldsymbol\mu_{i},\mathbf\Sigma_{i})\cdot\left[-\cfrac{1}{2}\cfrac{\partial\left(\ln{|\mathbf{\Sigma}_i|}\right) }{\partial \mathbf{\Sigma}_i}-\cfrac{1}{2}\cfrac{\partial \left[(\boldsymbol x_j-\boldsymbol\mu_i)^T\mathbf{\Sigma}_i^{-1}(\boldsymbol x_j-\boldsymbol\mu_i)\right]}{\partial \mathbf{\Sigma}_i}\right]\\
\end{aligned}
$$
由矩阵微分公式$\cfrac{\partial |\mathbf{X}|}{\partial \mathbf{X}}=|\mathbf{X}|\cdot(\mathbf{X}^{-1})^{T},\cfrac{\partial \boldsymbol{a}^{T} \mathbf{X}^{-1} \boldsymbol{b}}{\partial \mathbf{X}}=-\mathbf{X}^{-T} \boldsymbol{a b}^{T} \mathbf{X}^{-T}$可得
$$
\begin{aligned}
\cfrac{\partial}{\partial\mathbf\Sigma_{i}}\left(p(\boldsymbol x_{j}|\boldsymbol\mu_{i},\mathbf\Sigma_{i})\right)&=p(\boldsymbol x_{j}|\boldsymbol\mu_{i},\mathbf\Sigma_{i})\cdot\left[-\cfrac{1}{2}\mathbf{\Sigma}_i^{-1}+\cfrac{1}{2}\mathbf{\Sigma}_i^{-1}(\boldsymbol x_j-\boldsymbol\mu_i)(\boldsymbol x_j-\boldsymbol\mu_i)^T\mathbf{\Sigma}_i^{-1}\right]\\
\end{aligned}
$$
将此式代回$\cfrac {\partial LL(D)}{\partial\mathbf\Sigma_{i}}$中可得
$$
\cfrac {\partial LL(D)}{\partial\mathbf\Sigma_{i}}=\sum_{j=1}^m\cfrac{\alpha_{i}\cdot p(\boldsymbol x_{j}|\boldsymbol\mu_{i},\mathbf\Sigma_{i})}{\sum_{l=1}^k\alpha_{l}\cdot p(\boldsymbol x_{j}|\boldsymbol\mu_{l},\mathbf\Sigma_{l})}\cdot\left[-\cfrac{1}{2}\mathbf{\Sigma}_i^{-1}+\cfrac{1}{2}\mathbf{\Sigma}_i^{-1}(\boldsymbol x_j-\boldsymbol\mu_i)(\boldsymbol x_j-\boldsymbol\mu_i)^T\mathbf{\Sigma}_i^{-1}\right]
$$


又由公式(9.30)可知$\cfrac{\alpha_{i}\cdot p(\boldsymbol x_{j}|\boldsymbol\mu_{i},\mathbf\Sigma_{i})}{\sum_{l=1}^k\alpha_{l}\cdot p(\boldsymbol x_{j}|\boldsymbol\mu_{l},\mathbf\Sigma_{l})}=\gamma_{ji}$，所以上式可进一步化简为
$$
\cfrac {\partial LL(D)}{\partial\mathbf\Sigma_{i}}=\sum_{j=1}^m\gamma_{ji}\cdot\left[-\cfrac{1}{2}\mathbf{\Sigma}_i^{-1}+\cfrac{1}{2}\mathbf{\Sigma}_i^{-1}(\boldsymbol x_j-\boldsymbol\mu_i)(\boldsymbol x_j-\boldsymbol\mu_i)^T\mathbf{\Sigma}_i^{-1}\right]
$$
令上式等于0可得
$$
\cfrac {\partial LL(D)}{\partial\mathbf\Sigma_{i}}=\sum_{j=1}^m\gamma_{ji}\cdot\left[-\cfrac{1}{2}\mathbf{\Sigma}_i^{-1}+\cfrac{1}{2}\mathbf{\Sigma}_i^{-1}(\boldsymbol x_j-\boldsymbol\mu_i)(\boldsymbol x_j-\boldsymbol\mu_i)^T\mathbf{\Sigma}_i^{-1}\right]=0
$$
移项推导有：
$$
\begin{aligned}
\sum_{j=1}^m\gamma_{ji}\cdot\left[-\boldsymbol{I}+(\boldsymbol x_j-\boldsymbol\mu_i)(\boldsymbol x_j-\boldsymbol\mu_i)^T\mathbf{\Sigma}_i^{-1}\right]&=0\\
\sum_{j=1}^m\gamma_{ji}(\boldsymbol x_j-\boldsymbol\mu_i)(\boldsymbol x_j-\boldsymbol\mu_i)^T\mathbf{\Sigma}_i^{-1}&=\sum_{j=1}^m\gamma_{ji}\boldsymbol{I}\\
\sum_{j=1}^m\gamma_{ji}(\boldsymbol x_j-\boldsymbol\mu_i)(\boldsymbol x_j-\boldsymbol\mu_i)^T&=\sum_{j=1}^m\gamma_{ji}\mathbf{\Sigma}_i\\
\mathbf{\Sigma}_i&=\cfrac{\sum_{j=1}^m\gamma_{ji}(\boldsymbol x_j-\boldsymbol\mu_i)(\boldsymbol x_j-\boldsymbol\mu_i)^T}{\sum_{j=1}^m\gamma_{ji}}
\end{aligned}
$$
此即为公式(9.35)

### 9.38

$$
\alpha_{i}=\frac{1}{m}\sum_{j=1}^m\gamma_{ji}
$$

[推导]：对公式(9.37)两边同时乘以$\alpha_{i}$可得
$$
\begin{aligned}
\sum_{j=1}^m\frac{\alpha_{i}\cdot p(\boldsymbol x_{j}|\boldsymbol\mu_{i},\mathbf\Sigma_{i})}{\sum_{l=1}^k\alpha_{l}\cdot p(\boldsymbol x_{j}|\boldsymbol\mu_{l},\mathbf\Sigma_{l})}+\lambda\alpha_{i}=0\\
\sum_{j=1}^m\frac{\alpha_{i}\cdot p(\boldsymbol x_{j}|\boldsymbol\mu_{i},\mathbf\Sigma_{i})}{\sum_{l=1}^k\alpha_{l}\cdot p(\boldsymbol x_{j}|\boldsymbol\mu_{l},\mathbf\Sigma_{l})}=-\lambda\alpha_{i}
\end{aligned}
$$
两边对所有混合成分求和可得
$$
\begin{aligned}\sum_{i=1}^k\sum_{j=1}^m\frac{\alpha_{i}\cdot p(\boldsymbol x_{j}|\boldsymbol\mu_{i},\mathbf\Sigma_{i})}{\sum_{l=1}^k\alpha_{l}\cdot p(\boldsymbol x_{j}|\boldsymbol\mu_{l},\mathbf\Sigma_{l})}&=-\lambda\sum_{i=1}^k\alpha_{i}\\
\sum_{j=1}^m\sum_{i=1}^k\frac{\alpha_{i}\cdot p(\boldsymbol x_{j}|\boldsymbol\mu_{i},\mathbf\Sigma_{i})}{\sum_{l=1}^k\alpha_{l}\cdot p(\boldsymbol x_{j}|\boldsymbol\mu_{l},\mathbf\Sigma_{l})}&=-\lambda\sum_{i=1}^k\alpha_{i}
\end{aligned}
$$
由$$m=-\lambda$$有
$$
\sum_{j=1}^m\frac{\alpha_{i}\cdot p(\boldsymbol x_{j}|\boldsymbol\mu_{i},\mathbf\Sigma_{i})}{\sum_{l=1}^k\alpha_{l}\cdot p(\boldsymbol x_{j}|\boldsymbol\mu_{l},\mathbf\Sigma_{l})}=-\lambda\alpha_{i}=m\alpha_{i}
$$
因此
$$
\alpha_{i}=\cfrac{1}{m}\sum_{j=1}^m\frac{\alpha_{i}\cdot p(\boldsymbol x_{j}|\boldsymbol\mu_{i},\mathbf\Sigma_{i})}{\sum_{l=1}^k\alpha_{l}\cdot p(\boldsymbol x_{j}|\boldsymbol\mu_{l},\mathbf\Sigma_{l})}
$$
又由公式(9.30)可知$\cfrac{\alpha_{i}\cdot p(\boldsymbol x_{j}|\boldsymbol\mu_{i},\mathbf\Sigma_{i})}{\sum_{l=1}^k\alpha_{l}\cdot p(\boldsymbol x_{j}|\boldsymbol\mu_{l},\mathbf\Sigma_{l})}=\gamma_{ji}$，所以上式可进一步化简为
$$
\alpha_{i}=\cfrac{1}{m}\sum_{j=1}^m\gamma_{ji}
$$

此即为公式(9.38)

高斯混合聚类的算法流程如下图所示：

![19.png](media\Chapter9_聚类\5bc84fb9c4fa4.png)

## **9.4 密度聚类**

密度聚类则是基于密度的聚类，它从样本分布的角度来考察样本之间的可连接性，并基于可连接性（密度可达）不断拓展疆域（类簇）。其中最著名的便是**DBSCAN**算法，首先定义以下概念：

![20.png](media\Chapter9_聚类\5bc84fb9bd69c.png)

![21.png](media\Chapter9_聚类\5bc8509f8d619.png)

简单来理解DBSCAN便是：**找出一个核心对象所有密度可达的样本集合形成簇**。首先从数据集中任选一个核心对象A，找出所有A密度可达的样本集合，将这些样本形成一个密度相连的类簇，直到所有的核心对象都遍历完。DBSCAN算法的流程如下图所示：

![22.png](media\Chapter9_聚类\5bc8509feb587.png)

## **9.5 层次聚类**

层次聚类是一种基于树形结构的聚类方法，常用的是**自底向上**的结合策略（**AGNES算法**）。假设有N个待聚类的样本，其基本步骤是：

> 1.初始化-->把每个样本归为一类，计算每两个类之间的距离，也就是样本与样本之间的相似度；
> 2.寻找各个类之间最近的两个类，把他们归为一类（这样类的总数就少了一个）；
> 3.重新计算新生成的这个**类与各个旧类之间的相似度**；
> 4.重复2和3直到所有样本点都归为一类，结束。

可以看出其中最关键的一步就是**计算两个类簇的相似度**，这里有多种度量方法：

* 单链接（single-linkage）:取类间最小距离。

![23.png](https://i.loli.net/2018/10/18/5bc8509ebb022.png)

* 全链接（complete-linkage）:取类间最大距离

![24.png](https://i.loli.net/2018/10/18/5bc8509eb2b30.png)

* 均链接（average-linkage）:取类间两两的平均距离

![25.png](https://i.loli.net/2018/10/18/5bc8509f089a7.png)

很容易看出：**单链接的包容性极强，稍微有点暧昧就当做是自己人了，全链接则是坚持到底，只要存在缺点就坚决不合并，均连接则是从全局出发顾全大局**。层次聚类法的算法流程如下所示：

![26.png](media\Chapter9_聚类\5bc8509f9d4a0.png)

![image-20200910122633906](media\Chapter9_聚类\image-20200910122633906.png)

在此聚类算法就介绍完毕，分类/聚类都是机器学习中最常见的任务之一。
