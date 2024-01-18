上篇主要介绍了机器学习的理论基础，首先从独立同分布引入泛化误差与经验误差，接着介绍了PAC可学习的基本概念，即以较大的概率学习出与目标概念近似的假设（泛化误差满足预设上限），对于有限假设空间：（1）可分情形时，假设空间都是PAC可学习的，即当样本满足一定的数量之后，总是可以在与训练集一致的假设中找出目标概念的近似；（2）不可分情形时，假设空间都是不可知PAC可学习的，即以较大概率学习出与当前假设空间中泛化误差最小的假设的有效近似（Hoeffding不等式）。对于无限假设空间，通过增长函数与VC维来描述其复杂度，若学习算法满足经验风险最小化原则，则任何VC维有限的假设空间都是（不可知）PAC可学习的，同时也给出了泛化误差界与样本复杂度。稳定性则考察的是输入发生变化时输出的波动，稳定性通过损失函数与假设空间的可学习理论联系在了一起。本篇将讨论一种介于监督与非监督学习之间的学习算法--半监督学习。

# **13. 半监督学习**

前面我们一直围绕的都是监督学习与无监督学习，监督学习指的是训练样本包含标记信息的学习任务，例如：常见的分类与回归算法；无监督学习则是训练样本不包含标记信息的学习任务，例如：聚类算法。在实际生活中，常常会出现一部分样本有标记和较多样本无标记的情形，例如：做网页推荐时需要让用户标记出感兴趣的网页，但是少有用户愿意花时间来提供标记。若直接丢弃掉无标记样本集，使用传统的监督学习方法，常常会由于训练样本的不充足，使得其刻画总体分布的能力减弱，从而影响了学习器泛化性能。那如何利用未标记的样本数据呢？

一种简单的做法是通过专家知识对这些未标记的样本进行打标，但随之而来的就是巨大的人力耗费。若我们先使用有标记的样本数据集训练出一个学习器，再基于该学习器对未标记的样本进行预测，从中**挑选出不确定性高或分类置信度低的样本来咨询专家并进行打标**，最后使用扩充后的训练集重新训练学习器，这样便能大幅度降低标记成本，这便是**主动学习**（active learning），其目标是**使用尽量少的/有价值的咨询来获得更好的性能**。

显然，**主动学习需要与外界进行交互/查询/打标，其本质上仍然属于一种监督学习**。事实上，无标记样本虽未包含标记信息，但它们与有标记样本一样都是从总体中独立同分布采样得到，因此**它们所包含的数据分布信息对学习器的训练大有裨益**。如何让学习过程不依赖外界的咨询交互，自动利用未标记样本所包含的分布信息的方法便是**半监督学习**（semi-supervised learning），**即训练集同时包含有标记样本数据和未标记样本数据**。

![1.png](media\Chapter13_半监督学习\5bc856e39801d.png)

此外，半监督学习还可以进一步划分为**纯半监督学习**和**直推学习**，两者的区别在于：前者假定训练数据集中的未标记数据并非待预测数据，而后者假定学习过程中的未标记数据就是待预测数据。主动学习、纯半监督学习以及直推学习三者的概念如下图所示：

![iwJFJS.png](media\Chapter13_半监督学习\iwJFJS.png)

## **13.1 生成式方法**

**生成式方法**（generative methods）是基于生成式模型的方法，即先对联合分布P（x,c）建模，从而进一步求解 P（c | x），**此类方法假定样本数据服从一个潜在的分布，因此需要充分可靠的先验知识**。例如：前面已经接触到的贝叶斯分类器与高斯混合聚类，都属于生成式模型。现假定总体是一个高斯混合分布，即由多个高斯分布组合形成，从而一个子高斯分布就代表一个类簇（类别）。高斯混合分布的概率密度函数如下所示：

![3.png](media\Chapter13_半监督学习\5bc856e3b82dc.png)

不失一般性，假设类簇与真实的类别按照顺序一一对应，即第i个类簇对应第i个高斯混合成分。与高斯混合聚类类似地，这里的主要任务也是估计出各个高斯混合成分的参数以及混合系数，不同的是：对于有标记样本，不再是可能属于每一个类簇，而是只能属于真实类标对应的特定类簇。

![4.png](media\Chapter13_半监督学习\5bc856e431d30.png)

直观上来看，**基于半监督的高斯混合模型有机地整合了贝叶斯分类器与高斯混合聚类的核心思想**，有效地利用了未标记样本数据隐含的分布信息，从而使得参数的估计更加准确。同样地，这里也要召唤出之前的EM大法进行求解，首先对各个高斯混合成分的参数及混合系数进行随机初始化，计算出各个PM（即γji，第i个样本属于j类，有标记样本则直接属于特定类），再最大化似然函数（即LL（D）分别对α、u和∑求偏导 ），对参数进行迭代更新。

![5.png](media\Chapter13_半监督学习\5bc856e43ff08.png)

当参数迭代更新收敛后，对于待预测样本x，便可以像贝叶斯分类器那样计算出样本属于每个类簇的后验概率，接着找出概率最大的即可：

![6.png](media\Chapter13_半监督学习\5bc856e3dfb1c.png)

可以看出：基于生成式模型的方法十分依赖于对潜在数据分布的假设，即假设的分布要能和真实分布相吻合，否则利用未标记的样本数据反倒会在错误的道路上渐行渐远，从而降低学习器的泛化性能。因此，**此类方法要求极强的领域知识和掐指观天的本领**。

## **13.2 半监督SVM**

监督学习中的SVM试图找到一个划分超平面，使得两侧支持向量之间的间隔最大，即“**最大划分间隔**”思想。对于半监督学习，S3VM则考虑超平面需穿过数据低密度的区域。TSVM是半监督支持向量机中的最著名代表，其核心思想是：尝试为未标记样本找到合适的标记指派，使得超平面划分后的间隔最大化。TSVM采用局部搜索的策略来进行迭代求解，即首先使用有标记样本集训练出一个初始SVM，接着使用该学习器对未标记样本进行打标，这样所有样本都有了标记，并基于这些有标记的样本重新训练SVM，之后再寻找易出错样本不断调整。整个算法流程如下所示：

![7.png](media\Chapter13_半监督学习\5bc856e427830.png)

![iwJZss.png](media\Chapter13_半监督学习\iwJZss.png)

## **13.3 基于分歧的方法**

基于分歧的方法通过多个学习器之间的**分歧（disagreement）/多样性（diversity）**来利用未标记样本数据，协同训练就是其中的一种经典方法。**协同训练最初是针对于多视图（multi-view）数据而设计的，多视图数据指的是样本对象具有多个属性集，每个属性集则对应一个试图**。例如：电影数据中就包含画面类属性和声音类属性，这样画面类属性的集合就对应着一个视图。首先引入两个关于视图的重要性质：

> **相容性**：即使用单个视图数据训练出的学习器的输出空间是一致的。例如都是{好，坏}、{+1,-1}等。
> **互补性**：即不同视图所提供的信息是互补/相辅相成的，实质上这里体现的就是集成学习的思想。

协同训练正是很好地利用了多视图数据的“**相容互补性**”，其基本的思想是：首先基于有标记样本数据在每个视图上都训练一个初始分类器，然后让每个分类器去挑选分类置信度最高的样本并赋予标记，并将带有伪标记的样本数据传给另一个分类器去学习，从而**你依我侬/共同进步**。

![iwJVMj.png](media\Chapter13_半监督学习\iwJVMj.png)

![iwJeLn.png](media\Chapter13_半监督学习\iwJeLn.png)

## **13.4 半监督聚类**

前面提到的几种方法都是借助无标记样本数据来辅助监督学习的训练过程，从而使得学习更加充分/泛化性能得到提升；半监督聚类则是借助已有的监督信息来辅助聚类的过程。一般而言，监督信息大致有两种类型：

> **必连与勿连约束**：必连指的是两个样本必须在同一个类簇，勿连则是必不在同一个类簇。
> **标记信息**：少量的样本带有真实的标记。

下面主要介绍两种基于半监督的K-Means聚类算法：第一种是数据集包含一些必连与勿连关系，另外一种则是包含少量带有标记的样本。两种算法的基本思想都十分的简单：对于带有约束关系的k-均值算法，在迭代过程中对每个样本划分类簇时，需要**检测当前划分是否满足约束关系**，若不满足则会将该样本划分到距离次小对应的类簇中，再继续检测是否满足约束关系，直到完成所有样本的划分。算法流程如下图所示：

![iwJAzQ.png](media\Chapter13_半监督学习\iwJAzQ.png)

对于带有少量标记样本的k-均值算法，则可以**利用这些有标记样本进行类中心的指定，同时在对样本进行划分时，不需要改变这些有标记样本的簇隶属关系**，直接将其划分到对应类簇即可。算法流程如下所示：

![iwJkRg.png](media\Chapter13_半监督学习\iwJkRg.png)



在此，半监督学习就介绍完毕。十分有趣的是：半监督学习将前面许多知识模块联系在了一起，足以体现了作者编排的用心。结合本篇的新知识再来回想之前自己做过的一些研究，发现还是蹚了一些浑水，也许越是觉得过去的自己傻，越就是好的兆头吧~

### 13.1
$$
p(\boldsymbol{x})=\sum_{i=1}^{N} \alpha_{i} \cdot p\left(\boldsymbol{x} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)
$$

[解析]： 高斯混合分布的定义式。

### 13.2

$$
\begin{aligned} f(\boldsymbol{x}) &=\underset{j \in \mathcal{Y}}{\arg \max } p(y=j | \boldsymbol{x}) \\ &=\underset{j \in \mathcal{Y}}{\arg \max } \sum_{i=1}^{N} p(y=j, \Theta=i | \boldsymbol{x}) \\ &=\underset{j \in \mathcal{Y}}{\arg \max } \sum_{i=1}^{N} p(y=j | \Theta=i, \boldsymbol{x}) \cdot p(\Theta=i | \boldsymbol{x}) \end{aligned}
$$

[解析]：从公式第 1 行到第 2 行是对概率进行边缘化(marginalization)；通过引入$\Theta$并对其求和 $\sum_{i=1}^N$以抵消引入的影响。从公式第 2 行到第 3 行推导如下
$$
\begin{aligned}p(y=j, \Theta=i | \boldsymbol{x}) &=\frac{p(y=j, \Theta=i, \boldsymbol{x})}{p(\boldsymbol{x})} \\&=\frac{p(y=j, \Theta=i, \boldsymbol{x})}{p(\Theta=i, \boldsymbol{x})} \cdot \frac{p(\Theta=i, \boldsymbol{x})}{p(\boldsymbol{x})} \\&=p(y=j | \Theta=i, \boldsymbol{x}) \cdot p(\Theta=i | \boldsymbol{x})\end{aligned}
$$

### 13.3

$$
p(\Theta=i | \boldsymbol{x})=\frac{\alpha_{i} \cdot p\left(\boldsymbol{x} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)}{\sum_{i=1}^{N} \alpha_{i} \cdot p\left(\boldsymbol{x} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)}
$$

[解析]：根据 13.1 
$$
p(\boldsymbol{x})=\sum_{i=1}^{N} \alpha_{i} \cdot p\left(\boldsymbol{x} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)
$$
因此
$$
\begin{aligned}p(\Theta=i | \boldsymbol{x})&=\frac{p(\Theta=i , \boldsymbol{x})}{P(x)}\\&=\frac{\alpha_{i} \cdot p\left(\boldsymbol{x} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)}{\sum_{i=1}^{N} \alpha_{i} \cdot p\left(\boldsymbol{x} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)}\end{aligned}
$$

### 13.4

$$
\begin{aligned} L L\left(D_{l} \cup D_{u}\right)=& \sum_{\left(x_{j}, y_{j}\right) \in D_{l}} \ln \left(\sum_{i=1}^{N} \alpha_{i} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right) \cdot p\left(y_{j} | \Theta=i, \boldsymbol{x}_{j}\right)\right) \\ &+\sum_{x_{j} \in D_{u}} \ln \left(\sum_{i=1}^{N} \alpha_{i} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)\right) \end{aligned}
$$

[解析]：第二项很好解释，当不知道类别信息的时候，样本$x_j$的概率可以用式 13.1 表示，所有无类别信息的样本$D_u$的似然是所有样本的乘积，因为$\ln$函数是单调的，所以也可以将$\ln$函数作用于这个乘积消除因为连乘产生的数值计算问题。第一项引入了样本的标签信息，由
$$
p(y=j | \Theta=i, \boldsymbol{x})=\left\{\begin{array}{ll}1, & i=j \\0, & i \neq j\end{array}\right.
$$
可知，这项限定了样本$x_j$只可能来自于$y_j$所对应的高斯分布。

### 13.5

$$
\gamma_{j i}=\frac{\alpha_{i} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)}{\sum_{i=1}^{N} \alpha_{i} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)}
$$

[解析]：参见式 13.3，这项可以理解成样本$x_j$属于类别标签$i$(或者说由第$i$个高斯分布生成)的后验概率。其中$\alpha_i,\boldsymbol{\mu}_{i}\boldsymbol{\Sigma}_i$可以通过有标记样本预先计算出来。即：
$$
\begin{array}{l}\alpha_{i}=\frac{l_{i}}{\left|D_{l}\right|}, \text { where }\left|D_{l}\right|=\sum_{i=1}^{N} l_{i} \\\boldsymbol{\mu}_{i}=\frac{1}{l_{i}} \sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i} \boldsymbol{x}_{j} \\\boldsymbol{\Sigma}_{i}=\frac{1}{l_{i}} \sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{\top}\end{array}
$$


### 13.6

$$
\boldsymbol{\mu}_{i}=\frac{1}{\sum_{\boldsymbol{x}_{j} \in D_{u}} \gamma_{j i}+l_{i}}\left(\sum_{\boldsymbol{x}_{j} \in D_{u}} \gamma_{j i} \boldsymbol{x}_{j}+\sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i} \boldsymbol{x}_{j}\right)
$$

[推导]：这项可以由$$\cfrac{\partial LL(D_l \cup D_u) }{\partial \mu_i}=0$$而得，将式 13.4 的两项分别记为：
$$
\begin{aligned}LL(D_l)&=\sum_{(\boldsymbol{x_j},y_j \in D_l)}\ln\left(\sum_{s=1}^{N}\alpha_s \cdot p(\boldsymbol{x_j}\vert \boldsymbol{\mu}_s,\boldsymbol{\Sigma}_s) \cdot p(y_i|\Theta = s,\boldsymbol{x_j})\right)\\&=\sum_{(\boldsymbol{x_j},y_j \in D_l)}\ln\left(\sum_{s=1}^{N}\alpha_{y_j} \cdot p(\boldsymbol{x_j} \vert \boldsymbol{\mu}_{y_j},\boldsymbol{\Sigma}_{y_j})\right)\\LL(D_u)&=\sum_{\boldsymbol{x_j} \in D_u} \ln\left(\alpha_s \cdot p(\boldsymbol{x_j} | \boldsymbol{\mu}_s,\boldsymbol{\Sigma}_s)\right)\end{aligned}
$$
首先，$LL(D_l)$对$\boldsymbol{\mu_i}$求偏导，$LL(D_l)$求和号中只有$y_j=i$ 的项能留下来，即
$$
\begin{aligned}\frac{\partial L L\left(D_{l}\right)}{\partial \boldsymbol{\mu}_{i}} &=\sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i} \frac{\partial \ln \left(\alpha_{i} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)\right)}{\partial \boldsymbol{\mu}_{i}} \\&=\sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i} \frac{1}{p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)} \cdot \frac{\partial p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)}{\partial \boldsymbol{\mu}_{i}} \\&=\sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i} \frac{1}{p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right) \cdot \boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right) \\&=\sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i} \boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\end{aligned}
$$
$LL(D_u)$对$\boldsymbol{\mu_i}$求导，参考 9.33 的推导：
$$
\begin{aligned}
\frac{\partial L L\left(D_{u}\right)}{\partial \boldsymbol{\mu}_{i}} &=\sum_{\boldsymbol{x}_{j} \in D_{u}} \frac{\alpha_{i}}{\sum_{s=1}^{N} \alpha_{s} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{s}, \boldsymbol{\Sigma}_{s}\right)} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right) \cdot \boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right) \\
&=\sum_{\boldsymbol{x}_{j} \in D_{u}} \gamma_{j i} \cdot \boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)
\end{aligned}
$$


综上，
$$
\begin{aligned}\frac{\partial L L\left(D_{l} \cup D_{u}\right)}{\partial \boldsymbol{\mu}_{i}} &=\sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i} \boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)+\sum_{\boldsymbol{x}_{j} \in D_{u}} \gamma_{j i} \cdot \boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right) \\&=\boldsymbol{\Sigma}_{i}^{-1}\left(\sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)+\sum_{\boldsymbol{x}_{j} \in D_{u}} \gamma_{j i} \cdot\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\right) \\&=\boldsymbol{\Sigma}_{i}^{-1}\left(\sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i} \boldsymbol{x}_{j}+\sum_{\boldsymbol{x}_{j} \in D_{u}} \gamma_{j i} \cdot \boldsymbol{x}_{j}-\sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i} \boldsymbol{\mu}_{i}-\sum_{\boldsymbol{x}_{j} \in D_{u}} \gamma_{j i} \cdot \boldsymbol{\mu}_{i}\right)\end{aligned}
$$
令$\frac{\partial L L\left(D_{l} \cup D_{u}\right)}{\partial \boldsymbol{\mu}_{i}}=0$，两边同时左乘$\Sigma_i$并移项：
$$
\sum_{\boldsymbol{x}_{j} \in D_{u}} \gamma_{j i} \cdot \boldsymbol{\mu}_{i}+\sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i} \boldsymbol{\mu}_{i}=\sum_{\boldsymbol{x}_{j} \in D_{u}} \gamma_{j i} \cdot \boldsymbol{x}_{j}+\sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i} \boldsymbol{x}_{j}
$$
上式中，$\boldsymbol{\mu_i}$ 可以作为常量提到求和号外面，而$\sum_{\left(x_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i} 1=l_{i}$，即第$i$类样本的有标记 样本数目，因此

$$
\left(\sum_{x_{j} \in D_{u}} \gamma_{j i}+\sum_{\left(x_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i} 1\right) \boldsymbol{\mu}_{i}=\sum_{x_{j} \in D_{u}} \gamma_{j i} \cdot \boldsymbol{x}_{j}+\sum_{\left(x_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i} \boldsymbol{x}_{j}
$$


即得式 13.6。

### 13.7

$$
\begin{aligned}\boldsymbol{\Sigma}_{i}=& \frac{1}{\sum_{\boldsymbol{x}_{j} \in D_{u}} \gamma_{j i}+l_{i}}\left(\sum_{\boldsymbol{x}_{j} \in D_{u}} \gamma_{j i} \cdot\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{\top}\right.\\&\left.+\sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{\top}\right)\end{aligned}
$$

[推导]：类似于13.6 由$\cfrac{\partial LL(D_l \cup D_u) }{\partial \Sigma_i}=0$得，化简过程同13.6过程类似
首先$LL(D_l)$对$\boldsymbol{\Sigma_i}$求偏导 ，类似于 13.6 
$$
\begin{aligned} \frac{\partial L L\left(D_{l}\right)}{\partial \boldsymbol{\Sigma}_{i}} &=\sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i} \frac{\partial \ln \left(\alpha_{i} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)\right)}{\partial \boldsymbol{\Sigma}_{i}} \\ &=\sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i} \frac{1}{p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)} \cdot \frac{\partial p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)}{\partial \boldsymbol{\Sigma}_{i}} \\
&=\sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i} \frac{1}{p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right) \cdot\left(\boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{\top}-\boldsymbol{I}\right) \cdot \frac{1}{2} \boldsymbol{\Sigma}_{i}^{-1}\\
&=\sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i}\left(\boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{\top}-\boldsymbol{I}\right) \cdot \frac{1}{2} \boldsymbol{\Sigma}_{i}^{-1}
\end{aligned}
$$
然后$LL(D_u)$ 对$\boldsymbol{\Sigma_i}$求偏导，类似于 9.35
$$
\frac{\partial L L\left(D_{u}\right)}{\partial \boldsymbol{\Sigma}_{i}}=\sum_{\boldsymbol{x}_{j} \in D_{u}} \gamma_{j i} \cdot\left(\boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{\top}-\boldsymbol{I}\right) \cdot \frac{1}{2} \boldsymbol{\Sigma}_{i}^{-1}
$$
综合可得：
$$
\begin{aligned} \frac{\partial L L\left(D_{l} \cup D_{u}\right)}{\partial \boldsymbol{\Sigma}_{i}}=& \sum_{\boldsymbol{x}_{j} \in D_{u}} \gamma_{j i} \cdot\left(\boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{\top}-\boldsymbol{I}\right) \cdot \frac{1}{2} \boldsymbol{\Sigma}_{i}^{-1} \\ &+\sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i}\left(\boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{\top}-\boldsymbol{I}\right) \cdot \frac{1}{2} \boldsymbol{\Sigma}_{i}^{-1} \\=&\left(\sum_{\boldsymbol{x}_{j} \in D_{u}} \gamma_{j i} \cdot\left(\boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{\top}-\boldsymbol{I}\right)\right.\\ &\left.+\sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i}\left(\boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{\top}-\boldsymbol{I}\right)\right) \cdot \frac{1}{2} \boldsymbol{\Sigma}_{i}^{-1} \end{aligned}
$$
令$\frac{\partial L L\left(D_{l} \cup D_{u}\right)}{\partial \boldsymbol{\Sigma}_{i}}=0$，两边同时右乘$2\Sigma_i$并移项：
$$
\begin{aligned} \sum_{\boldsymbol{x}_{j} \in D_{u}} \gamma_{j i} \cdot \boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{\top}+& \sum_{\left(\boldsymbol{x}_{j}, y_{j} \in D_{l} \wedge y_{j}=i\right.} \boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{\top} \\=& \sum_{\boldsymbol{x}_{j} \in D_{u}} \gamma_{j i} \cdot \boldsymbol{I}+\sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i} \boldsymbol{I} \\ &=\left(\sum_{\boldsymbol{x}_{j} \in D_{u}} \gamma_{j i}+l_{i}\right) \boldsymbol{I} \end{aligned}
$$
两边同时左乘以$\Sigma_i$：
$$
\sum_{\boldsymbol{x}_{j} \in D_{u}} \gamma_{j i} \cdot\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{\top}+\sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{\top}=\left(\sum_{\boldsymbol{x}_{j} \in D_{u}} \gamma_{j i}+l_{i}\right) \boldsymbol{\Sigma}_{i}
$$
即得式 13.7。

### 13.8

$$
\alpha_{i}=\frac{1}{m}\left(\sum_{\boldsymbol{x}_{j} \in D_{u}} \gamma_{j i}+l_{i}\right)
$$

[推导]：类似于式 9.36，写出$LL(D_l \cup D_u)$的拉格朗日形式
$$
\begin{aligned}\mathcal{L}\left(D_{l} \cup D_{u}, \lambda\right) &=L L\left(D_{l} \cup D_{u}\right)+\lambda\left(\sum_{s=1}^{N} \alpha_{s}-1\right) \\&=L L\left(D_{l}\right)+L L\left(D_{u}\right)+\lambda\left(\sum_{s=1}^{N} \alpha_{s}-1\right)\end{aligned}
$$


类似于式 9.37，对$\alpha_i$求偏导。对于$LL(D_u)$，求导结果与式 9.37 的推导过程一样
$$
\frac{\partial L L\left(D_{u}\right)}{\partial \alpha_{i}}=\sum_{\boldsymbol{x}_{j} \in D_{u}} \frac{1}{\sum_{s=1}^{N} \alpha_{s} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{s}, \boldsymbol{\Sigma}_{s}\right)} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)
$$


对于$LL(D_l)$，类似于 13.6 和 13.7 的推导过程
$$
\begin{aligned}\frac{\partial L L\left(D_{l}\right)}{\partial \alpha_{i}} &=\sum_{\left(x_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i} \frac{\partial \ln \left(\alpha_{i} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)\right)}{\partial \alpha_{i}} \\&=\sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i} \frac{1}{\alpha_{i} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)} \cdot \frac{\partial\left(\alpha_{i} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)\right)}{\partial \alpha_{i}} \\&=\sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i} \frac{1}{\alpha_{i} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right) \\&=\sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i} \frac{1}{\alpha_{i}}=\frac{1}{\alpha_{i}} \cdot \sum_{\left(\boldsymbol{x}_{j}, y_{j}\right) \in D_{l} \wedge y_{j}=i} 1=\frac{l_{i}}{\alpha_{i}}\end{aligned}
$$


上式推导过程中，重点注意变量是$\alpha_i$ ，$p(x_j|\mu_i,\Sigma_i)$是常量；最后一行$\alpha_i$相对于求和变量为常量，因此作为公因子提到求和号外面； $l_i$ 为第$i$类样本的有标记样本数目。

综合两项结果：
$$
\frac{\partial \mathcal{L}\left(D_{l} \cup D_{u}, \lambda\right)}{\partial \alpha_{i}}=\frac{l_{i}}{\alpha_{i}}+\sum_{\boldsymbol{x}_{j} \in D_{u}} \frac{p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)}{\sum_{s=1}^{N} \alpha_{s} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{s}, \boldsymbol{\Sigma}_{s}\right)}+\lambda
$$


令$\cfrac{\partial LL(D_l \cup D_u) }{\partial \alpha_i}=0$ 并且两边同乘以$\alpha_i$，得
$$
\alpha_{i} \cdot \frac{l_{i}}{\alpha_{i}}+\sum_{\boldsymbol{x}_{j} \in D_{u}} \frac{\alpha_{i} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)}{\sum_{s=1}^{N} \alpha_{s} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{s}, \boldsymbol{\Sigma}_{s}\right)}+\lambda \cdot \alpha_{i}=0
$$


结合式 9.30 发现，求和号内即为后验概率$\gamma_{ji}$,即
$$
l_i+\sum_{x_i \in D_u} \gamma_{ji}+\lambda \alpha_i = 0
$$
对所有混合成分求和，得
$$
\sum_{i=1}^N l_i+\sum_{i=1}^N  \sum_{x_i \in D_u} \gamma_{ji}+\sum_{i=1}^N \lambda \alpha_i = 0
$$


这里$\sum_{i=1}^N \alpha_i =1$ ，因此$\sum_{i=1}^N \lambda \alpha_i=\lambda\sum_{i=1}^N \alpha_i=\lambda$，根据 9.30 中$\gamma_{ji}$表达式可知
$$
\sum_{i=1}^N \gamma_{ji} =  \sum_{i =1}^{N} \cfrac{\alpha_i \cdot  p(x_j|\mu_i,\Sigma_i)}{\Sigma_{s=1}^N \alpha_s \cdot p(x_j| \mu_s, \Sigma_s)}=  \cfrac{\sum_{i =1}^{N}\alpha_i \cdot  p(x_j|\mu_i,\Sigma_i)}{\sum_{s=1}^N \alpha_s \cdot p(x_j| \mu_s, \Sigma_s)}=1
$$


再结合加法满足交换律，所以
$$
\sum_{i=1}^N  \sum_{x_i \in D_u} \gamma_{ji}=\sum_{x_i \in D_u} \sum_{i=1}^N  \gamma_{ji} =\sum_{x_i \in D_u} 1=u
$$


以上分析过程中，$\sum_{x_j\in D_u}$ 形式与$\sum_{j=1}^u$等价，其中u为未标记样本集的样本个数； $\sum_{i=1}^Nl_i=l$其中$l$为有标记样本集的样本个数；将这些结果代入
$$
\sum_{i=1}^N l_i+\sum_{i=1}^N  \sum_{x_i \in D_u} \gamma_{ji}+\sum_{i=1}^N \lambda \alpha_i = 0
$$


解出$l+u+\lambda = 0$  且$l+u =m$ 其中$m$为样本总个数，移项即得$\lambda = -m$
最后带入整理解得
$$
l_i + \sum_{x_j \in{D_u}} \gamma_{ji}-\lambda \alpha_i = 0
$$
整理即得式 13.8。

### 13.9

$$
\min _{\boldsymbol{w}, \boldsymbol{b}, \boldsymbol{y}, \boldsymbol{\xi}} \frac{1}{2}\|\boldsymbol{w}\|_{2}^{2}+C_{l} \sum_{i=1}^{l} \xi_{i}+C_{u} \sum_{i=l+1}^{m} \xi_{i}\\
\begin{aligned}
\text { s.t. } &y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right) \geqslant 1-\xi_{i}, \quad i=1,2, \ldots, l\\
&\hat{y}_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right) \geqslant 1-\xi_{i}, \quad i=l+1, l+2, \ldots, m\\
&\xi_{i} \geqslant 0, \quad i=1,2, \dots, m
\end{aligned}
$$

[解析]：这个公式和公式 6.35 基本一致，除了引入了无标记样本的松弛变量$\xi_i, i=l+1,\cdots m$和对应的权重系数$C_u$和无标记样本的标记指派$\hat{y}_i$。

### 13.12

$$
\begin{aligned}
E(f) &=\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m}(\mathbf{W})_{i j}\left(f\left(\boldsymbol{x}_{i}\right)-f\left(\boldsymbol{x}_{j}\right)\right)^{2} \\
&=\frac{1}{2}\left(\sum_{i=1}^{m} d_{i} f^{2}\left(\boldsymbol{x}_{i}\right)+\sum_{j=1}^{m} d_{j} f^{2}\left(\boldsymbol{x}_{j}\right)-2 \sum_{i=1}^{m} \sum_{j=1}^{m}(\mathbf{W})_{i j} f\left(\boldsymbol{x}_{i}\right) f\left(\boldsymbol{x}_{j}\right)\right) \\
&=\sum_{i=1}^{m} d_{i} f^{2}\left(\boldsymbol{x}_{i}\right)-\sum_{i=1}^{m} \sum_{j=1}^{m}(\mathbf{W})_{i j} f\left(\boldsymbol{x}_{i}\right) f\left(\boldsymbol{x}_{j}\right) \\
&=\boldsymbol{f}^{\mathrm{T}}(\mathbf{D}-\mathbf{W}) \boldsymbol{f}
\end{aligned}
$$

[解析]：首先解释下这个能量函数的定义。原则上，我们希望能量函数$E(f)$越小越好，对于节点$i,j$，如果它们不相邻，则$\mathbf{W}_{i j}=0$，如果它们相邻，则最小化能量函数要求$f(x_i)$和$f(x_j)$尽量相似，和逻辑相符。下面进行公式的推导，首先由二项展开可得：
$$
\begin{aligned}
E(f) &=\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m}(\mathbf{W})_{i j}\left(f\left(\boldsymbol{x}_{i}\right)-f\left(\boldsymbol{x}_{j}\right)\right)^{2} \\
&=\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m}(\mathbf{W})_{i j}\left(f^{2}\left(\boldsymbol{x}_{i}\right)-2 f\left(\boldsymbol{x}_{i}\right) f\left(\boldsymbol{x}_{j}\right)+f^{2}\left(\boldsymbol{x}_{j}\right)\right) \\
&=\frac{1}{2}\left( \sum_{i=1}^{m} \sum_{j=1}^{m}(\mathbf{W})_{i j} f^{2}\left(\boldsymbol{x}_{i}\right)+ \sum_{i=1}^{m} \sum_{j=1}^{m}(\mathbf{W})_{i j} f^{2}\left(\boldsymbol{x}_{j}\right)-2\sum_{i=1}^{m} \sum_{j=1}^{m}(\mathbf{W})_{i j} f\left(\boldsymbol{x}_{i}\right) f\left(\boldsymbol{x}_{j}\right)\right)
\end{aligned}
$$
由于$\mathbf{W}$是一个对称矩阵，可以通过变量替换得到
$$
\begin{aligned}
\sum_{i=1}^{m} \sum_{j=1}^{m}(\mathbf{W})_{i j} f^{2}\left(\boldsymbol{x}_{j}\right)&=\sum_{j=1}^{m} \sum_{i=1}^{m}(\mathbf{W})_{j i} f^{2}\left(\boldsymbol{x}_{i}\right)\\
&=\sum_{i=1}^{m} \sum_{j=1}^{m}(\mathbf{W})_{i j} f^{2}\left(\boldsymbol{x}_{i}\right)\\
&=
\sum_{i=1}^{m} \sum_{j=1}^{m}(\mathbf{W})_{i j} f^{2}\left(\boldsymbol{x}_{j}\right)
\end{aligned}
$$
因此$E(f)$可化简为
$$
\begin{aligned}
E(f) &=  \sum_{i=1}^{m} \sum_{j=1}^{m}(\mathbf{W})_{i j} f^{2}\left(\boldsymbol{x}_{i}\right)-\sum_{i=1}^{m} \sum_{j=1}^{m}(\mathbf{W})_{i j} f\left(\boldsymbol{x}_{i}\right) f\left(\boldsymbol{x}_{j}\right)
\end{aligned}
$$
根据定义 $d_i=\sum_{j=1}^{l+u}\left(\mathbf{W}\right)_{ij}$，且$m=l+u$则
$$
\begin{aligned}
E(f)&=\sum_{i=1}^{m} d_{i} f^{2}\left(\boldsymbol{x}_{i}\right)-\sum_{i=1}^{m} \sum_{j=1}^{m}(\mathbf{W})_{i j} f\left(\boldsymbol{x}_{i}\right) f\left(\boldsymbol{x}_{j}\right)\\
&=\boldsymbol{f}^{\mathrm{T}}\mathbf{D}\boldsymbol{f}-\boldsymbol{f}^{\mathrm{T}}\mathbf{W}\boldsymbol{f}\\
&=\boldsymbol{f}^{\mathrm{T}}(\mathbf{D}-\mathbf{W}) \boldsymbol{f}
\end{aligned}
$$

### 13.13

$$
\begin{aligned}
E(f) &=\left(\boldsymbol{f}_{l}^{\mathrm{T}} \boldsymbol{f}_{u}^{\mathrm{T}}\right)\left(\left[\begin{array}{ll}
\mathbf{D}_{l l} & \mathbf{0}_{l u} \\
\mathbf{0}_{u l} & \mathbf{D}_{u u}
\end{array}\right]-\left[\begin{array}{ll}
\mathbf{W}_{l l} & \mathbf{W}_{l u} \\
\mathbf{W}_{u l} & \mathbf{W}_{u u}
\end{array}\right]\right)\left[\begin{array}{l}
\boldsymbol{f}_{l} \\
\boldsymbol{f}_{u}
\end{array}\right] \\
&=\boldsymbol{f}_{l}^{\mathrm{T}}\left(\mathbf{D}_{l l}-\mathbf{W}_{l l}\right) \boldsymbol{f}_{l}-2 \boldsymbol{f}_{u}^{\mathrm{T}} \mathbf{W}_{u l} \boldsymbol{f}_{l}+\boldsymbol{f}_{u}^{\mathrm{T}}\left(\mathbf{D}_{u u}-\mathbf{W}_{u u}\right) \boldsymbol{f}_{u}
\end{aligned}
$$

[解析]：根据矩阵乘法的定义，有：
$$
\begin{aligned}
E(f) &=\left[\begin{array}{ll}
\boldsymbol{f}_{l}^{\mathrm{T}} & \boldsymbol{f}_{u}^{\mathrm{T}}
\end{array}\right]\left[\begin{array}{cc}
\boldsymbol{D}_{l l}-\boldsymbol{W}_{l l} & -\boldsymbol{W}_{l u} \\
-\boldsymbol{W}_{u l} & \boldsymbol{D}_{u u}-\boldsymbol{W}_{u u}
\end{array}\right]\left[\begin{array}{l}
\boldsymbol{f}_{l} \\
\boldsymbol{f}_{u}
\end{array}\right] \\
&=\left[\begin{array}{ll}\boldsymbol{f}_{l}^{\mathrm{T}}\left(\boldsymbol{D}_{l l}-\boldsymbol{W}_{l l}\right)-\boldsymbol{f}_{u}^{\mathrm{T}} \boldsymbol{W}_{u l} & -\boldsymbol{f}_{l}^{\mathrm{T}} \boldsymbol{W}_{l u}+\boldsymbol{f}_{u}^{\mathrm{T}}\left(\boldsymbol{D}_{u u}-\boldsymbol{W}_{u u}\right)\end{array}\right]\left[\begin{array}{l}
\boldsymbol{f}_{l} \\
\boldsymbol{f}_{u}
\end{array}\right] \\
&=\left(\boldsymbol{f}_{l}^{\mathrm{T}}\left(\boldsymbol{D}_{l l}-\boldsymbol{W}_{l l}\right)-\boldsymbol{f}_{u}^{\mathrm{T}} \boldsymbol{W}_{u l}\right) \boldsymbol{f}_{l}+\left(-\boldsymbol{f}_{l}^{\mathrm{T}} \boldsymbol{W}_{l u}+\boldsymbol{f}_{u}^{\mathrm{T}}\left(\boldsymbol{D}_{u u}-\boldsymbol{W}_{u u}\right)\right) \boldsymbol{f}_{u} \\
&=\boldsymbol{f}_{l}^{\mathrm{T}}\left(\boldsymbol{D}_{l l}-\boldsymbol{W}_{l l}\right) \boldsymbol{f}_{l}-\boldsymbol{f}_{u}^{\mathrm{T}} \boldsymbol{W}_{u l} \boldsymbol{f}_{l}-\boldsymbol{f}_{l}^{\mathrm{T}} \boldsymbol{W}_{l u} \boldsymbol{f}_{u}+\boldsymbol{f}_{u}^{\mathrm{T}}\left(\boldsymbol{D}_{u u}-\boldsymbol{W}_{u u}\right) \boldsymbol{f}_{u} \\
&=\boldsymbol{f}_{l}^{\mathrm{T}}\left(\boldsymbol{D}_{l l}-\boldsymbol{W}_{l l}\right) \boldsymbol{f}_{l}-2 \boldsymbol{f}_{u}^{\mathrm{T}} \boldsymbol{W}_{u l} \boldsymbol{f}_{l}+\boldsymbol{f}_{u}^{\mathrm{T}}\left(\boldsymbol{D}_{u u}-\boldsymbol{W}_{u u}\right) \boldsymbol{f}_{u}
\end{aligned}
$$
其中最后一步，$\boldsymbol{f}_{l}^{\mathrm{T}} \boldsymbol{W}_{l u} \boldsymbol{f}_{u}=\left(\boldsymbol{f}_{l}^{\mathrm{T}} \boldsymbol{W}_{l u} \boldsymbol{f}_{u}\right)^{\mathrm{T}}=f_{u}^{\mathrm{T}} \boldsymbol{W}_{u l} \boldsymbol{f}_{l}$，因为这个式子的结果是一个标量。

### 13.14

$$
\begin{aligned}
E(f) &=\left(\boldsymbol{f}_{l}^{\mathrm{T}} \boldsymbol{f}_{u}^{\mathrm{T}}\right)\left(\left[\begin{array}{ll}
\mathbf{D}_{l l} & \mathbf{0}_{l u} \\
\mathbf{0}_{u l} & \mathbf{D}_{u u}
\end{array}\right]-\left[\begin{array}{ll}
\mathbf{W}_{l l} & \mathbf{W}_{l u} \\
\mathbf{W}_{u l} & \mathbf{W}_{u u}
\end{array}\right]\right)\left[\begin{array}{l}
\boldsymbol{f}_{l} \\
\boldsymbol{f}_{u}
\end{array}\right] \\
&=\boldsymbol{f}_{l}^{\mathrm{T}}\left(\mathbf{D}_{l l}-\mathbf{W}_{l l}\right) \boldsymbol{f}_{l}-2 \boldsymbol{f}_{u}^{\mathrm{T}} \mathbf{W}_{u l} \boldsymbol{f}_{l}+\boldsymbol{f}_{u}^{\mathrm{T}}\left(\mathbf{D}_{u u}-\mathbf{W}_{u u}\right) \boldsymbol{f}_{u}
\end{aligned}
$$



[解析]：参考 13.13

### 13.15

$$
\boldsymbol{f}_{u}=\left(\mathbf{D}_{u u}-\mathbf{W}_{u u}\right)^{-1} \mathbf{W}_{u l} \boldsymbol{f}_{l}
$$

[解析]：由 13.13，有
$$
\begin{aligned}
\frac{\partial E(f)}{\partial \boldsymbol{f}_{u}} &=\frac{\partial \boldsymbol{f}_{l}^{\mathrm{T}}\left(\boldsymbol{D}_{l l}-\boldsymbol{W}_{l l}\right) \boldsymbol{f}_{l}-2 \boldsymbol{f}_{u}^{\mathrm{T}} \boldsymbol{W}_{u l} \boldsymbol{f}_{l}+\boldsymbol{f}_{u}^{\mathrm{T}}\left(\boldsymbol{D}_{u u}-\boldsymbol{W}_{u u}\right) \boldsymbol{f}_{u}}{\partial \boldsymbol{f}_{u}} \\
&=-2 \boldsymbol{W}_{u l} \boldsymbol{f}_{l}+2\left(\boldsymbol{D}_{u u}-\boldsymbol{W}_{u u}\right) \boldsymbol{f}_{u}
\end{aligned}
$$
令结果等于 0 即得 13.15。

### 13.16

$$
\begin{aligned}
\mathbf{P} &=\mathbf{D}^{-1} \mathbf{W}=\left[\begin{array}{cc}
\mathbf{D}_{l l}^{-1} & \mathbf{0}_{l u} \\
\mathbf{0}_{u l} & \mathbf{D}_{u u}^{-1}
\end{array}\right]\left[\begin{array}{ll}
\mathbf{W}_{l l} & \mathbf{W}_{l u} \\
\mathbf{W}_{u l} & \mathbf{W}_{u u}
\end{array}\right] \\
&=\left[\begin{array}{ll}
\mathbf{D}_{l l}^{-1} \mathbf{W}_{l l} & \mathbf{D}_{l l}^{-1} \mathbf{W}_{l u} \\
\mathbf{D}_{u u}^{-1} \mathbf{W}_{u l} & \mathbf{D}_{u u}^{-1} \mathbf{W}_{u u}
\end{array}\right]
\end{aligned}
$$

[解析]：根据矩阵乘法的定义计算可得该式，其中需要注意的是，对角矩阵$\mathbf{D}$的拟等于其各个对角元素的逆。

### 13.17

$$
\begin{aligned}
\boldsymbol{f}_{u} &=\left(\mathbf{D}_{u u}\left(\mathbf{I}-\mathbf{D}_{u u}^{-1} \mathbf{W}_{u u}\right)\right)^{-1} \mathbf{W}_{u l} \boldsymbol{f}_{l} \\
&=\left(\mathbf{I}-\mathbf{D}_{u u}^{-1} \mathbf{W}_{u u}\right)^{-1} \mathbf{D}_{u u}^{-1} \mathbf{W}_{u l} \boldsymbol{f}_{l} \\
&=\left(\mathbf{I}-\mathbf{P}_{u u}\right)^{-1} \mathbf{P}_{u l} \boldsymbol{f}_{l}
\end{aligned}
$$

[解析]：第一项到第二项是根据矩阵乘法逆的定义：$(\mathbf{A}\mathbf{B})^{-1}=\mathbf{B}^{-1}\mathbf{A}^{-1}$，在这个式子中​
$$
\begin{aligned}
\mathbf{P}_{u u}&=\mathbf{D}_{u u}^{-1} \mathbf{W}_{u u}\\
\mathbf{P}_{ul}&=\mathbf{D}_{u u}^{-1} \mathbf{W}_{u l}
\end{aligned}
$$
均可以根据$\mathbf{W}_{ij}$计算得到，因此可以通过标记$\mathbf{f}_l$计算未标记数据的标签$\mathbf{f}_u$。

### 13.20

$$
\mathbf{F}^{*}=\lim _{t \rightarrow \infty} \mathbf{F}(t)=(1-\alpha)(\mathbf{I}-\alpha \mathbf{S})^{-1} \mathbf{Y}
$$

[解析]：由 13.19
$$
\mathbf{F}(t+1)=\alpha \mathbf{S} \mathbf{F}(t)+(1-\alpha) \mathbf{Y}
$$
当 t取不同的值时，有：
$$
\begin{aligned}
t=0: \mathbf{F}(1) &=\alpha \mathbf{S F}(0)+(1-\alpha) \mathbf{Y}\\
&=\alpha \mathbf{S} \mathbf{Y}+(1-\alpha) \mathbf{Y} \\
t=1: \mathbf{F}(2) &=\alpha \mathbf{S F}(1)+(1-\alpha) \mathbf{Y}=\alpha \mathbf{S}(\alpha \mathbf{S} \mathbf{Y}+(1-\alpha) \mathbf{Y})+(1-\alpha) \mathbf{Y} \\
&=(\alpha \mathbf{S})^{2} \mathbf{Y}+(1-\alpha)\left(\sum_{i=0}^{1}(\alpha \mathbf{S})^{i}\right) \mathbf{Y} \\
t=2:\mathbf{F}(3)&=\alpha\mathbf{S}\mathbf{F}(2)+(1-\alpha)\mathbf{Y}\\&=\alpha \mathbf{S}\left((\alpha \mathbf{S})^{2} \mathbf{Y}+(1-\alpha)\left(\sum_{i=0}^{1}(\alpha \mathbf{S})^{i}\right) \mathbf{Y}\right)+(1-\alpha) \mathbf{Y} \\
&=(\alpha \mathbf{S})^{3} \mathbf{Y}+(1-\alpha)\left(\sum_{i=0}^{2}(\alpha \mathbf{S})^{i}\right) \mathbf{Y}\\
\end{aligned}
$$
可以观察到规律
$$
\mathbf{F}(t)=(\alpha \mathbf{S})^{t} \mathbf{Y}+(1-\alpha)\left(\sum_{i=0}^{t-1}(\alpha \mathbf{S})^{i}\right) \mathbf{Y}
$$
则
$$
\mathbf{F}^{*}=\lim _{t \rightarrow \infty}\mathbf{F}(t)=\lim _{t \rightarrow \infty}(\alpha \mathbf{S})^{t} \mathbf{Y}+\lim _{t \rightarrow \infty}(1-\alpha)\left(\sum_{i=0}^{t-1}(\alpha \mathbf{S})^{i}\right) \mathbf{Y}
$$
其中第一项由于$\mathbf{S}=\mathbf{D}^{-\frac{1}{2}} \mathbf{W} \mathbf{D}^{-\frac{1}{2}}$的特征值介于[-1, 1]之间(这里省略详细推导，可以参见 https://en.wikipedia.org/wiki/Laplacian_matrix 其中对称拉普拉斯矩阵的特征值介于 0 和 2 之间)，而$\alpha\in(0,1)$，所以$\lim _{t \rightarrow \infty}(\alpha \mathbf{S})^{t}=0$，第二项由等比数列公式
$$
\lim _{t \rightarrow \infty} \sum_{i=0}^{t-1}(\alpha \mathbf{S})^{i}=\frac{\mathbf{I}-\lim _{t \rightarrow \infty}(\alpha \mathbf{S})^{t}}{\mathbf{I}-\alpha \mathbf{S}}=\frac{\mathbf{I}}{\mathbf{I}-\alpha \mathbf{S}}=(\mathbf{I}-\alpha \mathbf{S})^{-1}
$$
综合可得式 13.20。



