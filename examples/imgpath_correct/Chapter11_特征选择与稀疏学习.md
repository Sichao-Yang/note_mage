上篇主要介绍了经典的降维方法与度量学习，首先从"维数灾难"导致的样本稀疏以及距离难计算两大难题出发，引出了降维的概念，即通过某种数学变换将原始高维空间转变到一个低维的子空间，接着分别介绍了kNN、MDS、PCA、KPCA以及两种经典的流形学习方法，k近邻算法的核心在于k值的选取以及距离的度量，MDS要求原始空间样本之间的距离在降维后的低维空间中得以保持，主成分分析试图找到一个低维超平面来表出原空间样本点，核化主成分分析先将样本点映射到高维空间，再在高维空间中使用线性降维的方法，从而解决了原空间样本非线性分布的情形，基于流形学习的降维则是一种"邻域保持"的思想，最后度量学习试图去学习出一个距离度量来等效降维的效果。本篇将讨论另一种常用方法--特征选择与稀疏学习。

# **11. 特征选择与稀疏学习**

最近在看论文的过程中，发现对于数据集行和列的叫法颇有不同，故在介绍本篇之前，决定先将最常用的术语罗列一二，以后再见到了不管它脚扑朔还是眼迷离就能一眼识破真身了~对于数据集中的一个对象及组成对象的零件元素：

* 统计学家常称它们为**观测**（**observation**）和**变量**（**variable**）；
* 数据库分析师则称其为**记录**（**record**）和**字段**（**field**）；
* 数据挖掘/机器学习学科的研究者则习惯把它们叫做**样本**/**示例**（**example**/**instance**）和**属性**/**特征**（**attribute**/**feature**）。


回归正题，在机器学习中特征选择是一个重要的"**数据预处理**"（**data** **preprocessing**）过程，即试图从数据集的所有特征中挑选出与当前学习任务相关的特征子集，接着再利用数据子集来训练学习器；稀疏学习则是围绕着稀疏矩阵的优良性质，来完成相应的学习任务。

## **11.1 子集搜索与评价**

一般地，我们可以用很多属性/特征来描述一个示例，例如对于一个人可以用性别、身高、体重、年龄、学历、专业、是否吃货等属性来描述，那现在想要训练出一个学习器来预测人的收入。根据生活经验易知：并不是所有的特征都与学习任务相关，例如年龄/学历/专业可能很大程度上影响了收入，身高/体重这些外貌属性也有较小的可能性影响收入，但像是否是一个地地道道的吃货这种属性就八杆子打不着了。因此我们只需要那些与学习任务紧密相关的特征，**特征选择便是从给定的特征集合中选出相关特征子集的过程**。

与上篇中降维技术有着异曲同工之处的是，特征选择也可以有效地解决维数灾难的难题。具体而言：**降维从一定程度起到了提炼优质低维属性和降噪的效果，特征选择则是直接剔除那些与学习任务无关的属性而选择出最佳特征子集**。若直接遍历所有特征子集，显然当维数过多时遭遇指数爆炸就行不通了；若采取从候选特征子集中不断迭代生成更优候选子集的方法，则时间复杂度大大减小。这时就涉及到了两个关键环节：**1.如何生成候选子集；2.如何评价候选子集的好坏**，这便是早期特征选择的常用方法。书本上介绍了贪心算法，分为三种策略：


> **前向搜索**：初始将每个特征当做一个候选特征子集，然后从当前所有的候选子集中选择出最佳的特征子集；接着在上一轮选出的特征子集中添加一个新的特征，同样地选出最佳特征子集；最后直至选不出比上一轮更好的特征子集。
> **后向搜索**：初始将所有特征作为一个候选特征子集；接着尝试去掉上一轮特征子集中的一个特征并选出当前最优的特征子集；最后直到选不出比上一轮更好的特征子集。
> **双向搜索**：将前向搜索与后向搜索结合起来，即在每一轮中既有添加操作也有剔除操作。


对于特征子集的评价，书中给出了一些想法及基于信息熵的方法。假设数据集的属性皆为离散属性，这样给定一个特征子集，便可以通过这个特征子集的取值将数据集合划分为V个子集。例如：A1={男,女}，A2={本科,硕士}就可以将原数据集划分为2*2=4个子集，其中每个子集的取值完全相同。这时我们就可以像决策树选择划分属性那样，通过计算信息增益来评价该属性子集的好坏。

![1.png](https://i.loli.net/2018/10/18/5bc853eca1a43.png)

此时，信息增益越大表示该属性子集包含有助于分类的特征越多，使用上述这种**子集搜索与子集评价相结合的机制，便可以得到特征选择方法**。值得一提的是若将前向搜索策略与信息增益结合在一起，与前面我们讲到的ID3决策树十分地相似。事实上，决策树也可以用于特征选择，树节点划分属性组成的集合便是选择出的特征子集。

## **11.2 过滤式选择（Relief）**

过滤式方法是一种将特征选择与学习器训练相分离的特征选择技术，即首先将相关特征挑选出来，再使用选择出的数据子集来训练学习器。Relief是其中著名的代表性算法，它使用一个"**相关统计量**"来度量特征的重要性，该统计量是一个向量，其中每个分量代表着相应特征的重要性，因此我们最终可以根据这个统计量各个分量的大小来选择出合适的特征子集。

易知Relief算法的核心在于如何计算出该相关统计量。对于数据集中的每个样例xi，Relief首先找出与xi同类别的最近邻与不同类别的最近邻，分别称为**猜中近邻（near-hit）**与**猜错近邻（near-miss）**，接着便可以分别计算出相关统计量中的每个分量。对于j分量：

![2.png](https://i.loli.net/2018/10/18/5bc853ec70c88.png)

直观上理解：对于猜中近邻，两者j属性的距离越小越好，对于猜错近邻，j属性距离越大越好。更一般地，若xi为离散属性，diff取海明距离，即相同取0，不同取1；若xi为连续属性，则diff为曼哈顿距离，即取差的绝对值。分别计算每个分量，最终取平均便得到了整个相关统计量。

标准的Relief算法只用于二分类问题，后续产生的拓展变体Relief-F则解决了多分类问题。对于j分量，新的计算公式如下：

![3.png](https://i.loli.net/2018/10/18/5bc853ec93042.png)

其中pl表示第l类样本在数据集中所占的比例，易知两者的不同之处在于：**标准Relief 只有一个猜错近邻，而Relief-F有多个猜错近邻**。

## **11.3 包裹式选择（LVW）**

与过滤式选择不同的是，包裹式选择将后续的学习器也考虑进来作为特征选择的评价准则。因此包裹式选择可以看作是为某种学习器**量身定做**的特征选择方法，由于在每一轮迭代中，包裹式选择都需要训练学习器，因此在获得较好性能的同时也产生了较大的开销。下面主要介绍一种经典的包裹式特征选择方法 --LVW（Las Vegas Wrapper），它在拉斯维加斯框架下使用随机策略来进行特征子集的搜索。拉斯维加斯？怎么听起来那么耳熟，不是那个声名显赫的赌场吗？歪果仁真会玩。怀着好奇科普一下，结果又顺带了一个赌场：

> **蒙特卡罗算法**：采样越多，越近似最优解，一定会给出解，但给出的解不一定是正确解；
> **拉斯维加斯算法**：采样越多，越有机会找到最优解，不一定会给出解，且给出的解一定是正确解。

举个例子，假如筐里有100个苹果，让我每次闭眼拿1个，挑出最大的。于是我随机拿1个，再随机拿1个跟它比，留下大的，再随机拿1个……我每拿一次，留下的苹果都至少不比上次的小。拿的次数越多，挑出的苹果就越大，但我除非拿100次，否则无法肯定挑出了最大的。这个挑苹果的算法，就属于蒙特卡罗算法——尽量找较好的，但不保证是最好的。

而拉斯维加斯算法，则是另一种情况。假如有一把锁，给我100把钥匙，只有1把是对的。于是我每次随机拿1把钥匙去试，打不开就再换1把。我试的次数越多，打开（正确解）的机会就越大，但在打开之前，那些错的钥匙都是没有用的。这个试钥匙的算法，就是拉斯维加斯的——尽量找最好的，但不保证能找到。

LVW算法的具体流程如下所示，其中比较特别的是停止条件参数T的设置，即在每一轮寻找最优特征子集的过程中，若随机T次仍没找到，算法就会停止，从而保证了算法运行时间的可行性。

![4.png](https://i.loli.net/2018/10/18/5bc853ed5e08e.png)

## **11.4 嵌入式选择与正则化**

前面提到了的两种特征选择方法：**过滤式中特征选择与后续学习器完全分离，包裹式则是使用学习器作为特征选择的评价准则；嵌入式是一种将特征选择与学习器训练完全融合的特征选择方法，即将特征选择融入学习器的优化过程中**。在之前《经验风险与结构风险》中已经提到：经验风险指的是模型与训练数据的契合度，结构风险则是模型的复杂程度，机器学习的核心任务就是：**在模型简单的基础上保证模型的契合度**。例如：岭回归就是加上了L2范数的最小二乘法，有效地解决了奇异矩阵、过拟合等诸多问题，下面的嵌入式特征选择则是在损失函数后加上了L1范数。

![5.png](https://i.loli.net/2018/10/18/5bc853ec8b203.png)

L1范数美名又约**Lasso Regularization**，指的是向量中每个元素的绝对值之和，这样在优化目标函数的过程中，就会使得w尽可能地小，在一定程度上起到了防止过拟合的作用，同时与L2范数（Ridge Regularization ）不同的是，L1范数会使得部分w变为0， 从而达到了特征选择的效果。

总的来说：**L1范数会趋向产生少量的特征，其他特征的权值都是0；L2会选择更多的特征，这些特征的权值都会接近于0**。这样L1范数在特征选择上就十分有用，而L2范数则具备较强的控制过拟合能力。可以从下面两个方面来理解：

（1）**下降速度**：L1范数按照绝对值函数来下降，L2范数按照二次函数来下降。因此在0附近，L1范数的下降速度大于L2范数，故L1范数能很快地下降到0，而L2范数在0附近的下降速度非常慢，因此较大可能收敛在0的附近。

![6.png](https://i.loli.net/2018/10/18/5bc853ed0aaf5.png)

（2）**空间限制**：L1范数与L2范数都试图在最小化损失函数的同时，让权值W也尽可能地小。我们可以将原优化问题看做为下面的问题，即让后面的规则则都小于某个阈值。这样从图中可以看出：L1范数相比L2范数更容易得到稀疏解。

![7.png](https://i.loli.net/2018/10/18/5bc853ecc223e.png)

![8.png](https://i.loli.net/2018/10/18/5bc853ed51aa1.png)

## **11.5 稀疏表示与字典学习**

当样本数据是一个稀疏矩阵时，对学习任务来说会有不少的好处，例如很多问题变得线性可分，储存更为高效等。这便是稀疏表示与字典学习的基本出发点。稀疏矩阵即矩阵的每一行/列中都包含了大量的零元素，且这些零元素没有出现在同一行/列，对于一个给定的稠密矩阵，若我们能**通过某种方法找到其合适的稀疏表示**，则可以使得学习任务更加简单高效，我们称之为**稀疏编码（sparse coding）**或**字典学习（dictionary learning）**。

给定一个数据集，字典学习/稀疏编码指的便是通过一个字典将原数据转化为稀疏表示，因此最终的目标就是求得字典矩阵B及稀疏表示α，书中使用变量交替优化的策略能较好地求得解，深感陷进去短时间无法自拔，故先不进行深入...

![9.png](https://i.loli.net/2018/10/18/5bc853ed0ca43.png)

## **11.6 压缩感知**

压缩感知在前些年也是风风火火，与特征选择、稀疏表示不同的是：它关注的是通过欠采样信息来恢复全部信息。在实际问题中，为了方便传输和存储，我们一般将数字信息进行压缩，这样就有可能损失部分信息，如何根据已有的信息来重构出全部信号，这便是压缩感知的来历，压缩感知的前提是已知的信息具有稀疏表示。下面是关于压缩感知的一些背景：

![10.png](https://i.loli.net/2018/10/18/5bc853ed431c6.png)

在此，特征选择与稀疏学习就介绍完毕。在很多实际情形中，选了好的特征比选了好的模型更为重要，这也是为什么厉害的大牛能够很快地得出一些结论的原因，谓：吾昨晚夜观天象，星象云是否吃货乃无用也~



### 11.1

$$
\operatorname{Gain}(A)=\operatorname{Ent}(D)-\sum_{v=1}^{V} \frac{\left|D^{v}\right|}{|D|} \operatorname{Ent}\left(D^{v}\right)
$$

[解析]：此为信息增益的定义式，对数据集$D$和属性子集$A$，假设根据$A$的取值将$D$分为了$V$个子集$\{D^1,D^2,\dots,D^V\}$，那么信息增益的定义为划分之前数据集$D$的信息熵和划分之后每个子数据集$D^v$的信息熵的差。熵用来衡量一个系统的混乱程度，因此划分前和划分后熵的差越大，表示划分越有效，划分带来的"信息增益"越大。

### 11.2

$$
\operatorname{Ent}(D)=-\sum_{i=1}^{| \mathcal{Y |}} p_{k} \log _{2} p_{k}
$$

[解析]：此为信息熵的定义式，其中$p_k, k=1, 2, \dots \vert\mathcal{Y}\vert$表示$D$中第$i$类样本所占的比例。可以看出，样本越纯，即$p_k\rightarrow 0$或$p_k\rightarrow 1$时，$\mathrm{Ent}(D)$越小，其最小值为0。此时必有$p_i=1, p_{\backslash i}=0, i=1, 2, \dots, \vert\mathcal{Y}\vert$。

### 11.5

$$
\min _{\boldsymbol{w}} \sum_{i=1}^{m}\left(y_{i}-\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}\right)^{2}
$$

[解析]：该式为线性回归的优化目标式，$y_i$表示样本$i$的真实值，而$w^\top x_i$表示其预测值，这里使用预测值和真实值差的平方衡量预测值偏离真实值的大小。

### 11.6

$$
\min _{\boldsymbol{w}} \sum_{i=1}^{m}\left(y_{i}-\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}\right)^{2}+\lambda\|\boldsymbol{w}\|_{2}^{2}
$$

[解析]：该式为加入了$\mathrm{L}_2$正规化项的优化目标，也叫"岭回归"，$\lambda$用来调节误差项和正规化项的相对重要性，引入正规化项的目的是为了防止$w$的分量过太而导致过拟合的风险。

### 11.7

$$
\min _{\boldsymbol{w}} \sum_{i=1}^{m}\left(y_{i}-\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}\right)^{2}+\lambda\|\boldsymbol{w}\|_{1}
$$

[解析]：该式将11.6中的$\mathrm{L}_2$正规化项替换成了$\mathrm{L}_1$正规化项，也叫LASSO回归。关于$\mathrm{L}_2$和$\mathrm{L}_1$两个正规化项的区别，原书图11.2给出了很形象的解释。具体来说，结合$\mathrm{L}_1$范数优化的模型参数分量更偏向于取0，因此更容易取得稀疏解。

### 11.10

$$
\begin{aligned}
\hat{f}(\boldsymbol{x}) & \simeq f\left(\boldsymbol{x}_{k}\right)+\left\langle\nabla f\left(\boldsymbol{x}_{k}\right), \boldsymbol{x}-\boldsymbol{x}_{k}\right\rangle+\frac{L}{2}\left\|\boldsymbol{x}-\boldsymbol{x}_{k}\right\|^{2} \\
&=\frac{L}{2}\left\|\boldsymbol{x}-\left(\boldsymbol{x}_{k}-\frac{1}{L} \nabla f\left(\boldsymbol{x}_{k}\right)\right)\right\|_{2}^{2}+\mathrm{const}
\end{aligned}
$$

[解析]：首先注意优化目标式和11.7 LASSO回归的联系和区别，该式中的$x$对应到式11.7的$w$，即我们优化的目标。再解释下什么是[$L\mathrm{-Lipschitz}$条件](https://zh.wikipedia.org/wiki/利普希茨連續)，根据维基百科的定义：它是一个比通常[连续](https://zh.wikipedia.org/wiki/連續函數)更强的光滑性条件。直觉上，利普希茨连续函数限制了函数改变的速度，符合利普希茨条件的函数的斜率，必小于一个称为利普希茨常数的实数（该常数依函数而定）。

注意这里可能存在一个笔误，在wiki百科的定义中，式11.7应该写成
$$
\left\vert\nabla f\left(\boldsymbol{x}^{\prime}\right)-\nabla f(\boldsymbol{x})\right\vert \leqslant L\left\vert\boldsymbol{x}^{\prime}-\boldsymbol{x}\right\vert \quad\left(\forall \boldsymbol{x}, \boldsymbol{x}^{\prime}\right)
$$
移项得
$$
\frac{\left|\nabla f\left(\boldsymbol{x}^{\prime}\right)-\nabla f(\boldsymbol{x})\right|}{\vert x^\prime - x\vert}\leqslant L \quad\left(\forall \boldsymbol{x}, \boldsymbol{x}^{\prime}\right)
$$
由于上式对所有的$x, x^\prime$都成立，由[导数的定义](https://zh.wikipedia.org/wiki/导数)，上式可以看成是$f(x)$的二阶导数恒不大于$L$。即
$$
\nabla^2f(x)\leqslant L
$$
得到这个结论之后，我们来推导式11.10。

由[泰勒公式](https://zh.wikipedia.org/wiki/泰勒公式)，$x_k$附近的$f(x)$通过二阶泰勒展开式可近似为
$$
\begin{aligned}
\hat{f}(\boldsymbol{x}) & \simeq f\left(\boldsymbol{x}_{k}\right)+\left\langle\nabla f\left(\boldsymbol{x}_{k}\right), \boldsymbol{x}-\boldsymbol{x}_{k}\right\rangle+\frac{\nabla^2f(x_k)}{2}\left\|\boldsymbol{x}-\boldsymbol{x}_{k}\right\|^{2} \\
&\leqslant
 f\left(\boldsymbol{x}_{k}\right)+\left\langle\nabla f\left(\boldsymbol{x}_{k}\right), \boldsymbol{x}-\boldsymbol{x}_{k}\right\rangle+\frac{L}{2}\left\|\boldsymbol{x}-\boldsymbol{x}_{k}\right\|^{2} \\
&= f\left(\boldsymbol{x}_{k}\right)+\nabla f\left(\boldsymbol{x}_{k}\right)^{\top}\left(\boldsymbol{x}-\boldsymbol{x}_{k}\right)+\frac{L}{2}\left(\boldsymbol{x}-\boldsymbol{x}_{k}\right)^{\top}\left(\boldsymbol{x}-\boldsymbol{x}_{k}\right)\\
&=f(x_k)+\frac{L}{2}\left(\left(\boldsymbol{x}-\boldsymbol{x}_{k}\right)^{\top}\left(\boldsymbol{x}-\boldsymbol{x}_{k}\right)+\frac{2}{L}\nabla f\left(\boldsymbol{x}_{k}\right)^{\top}\left(\boldsymbol{x}-\boldsymbol{x}_{k}\right)\right)\\
&=f(x_k)+\frac{L}{2}\left(\left(\boldsymbol{x}-\boldsymbol{x}_{k}\right)^{\top}\left(\boldsymbol{x}-\boldsymbol{x}_{k}\right)+\frac{2}{L}\nabla f\left(\boldsymbol{x}_{k}\right)^{\top}\left(\boldsymbol{x}-\boldsymbol{x}_{k}\right)+\frac{1}{L^2}\nabla f(x_k)^\top\nabla f(x_k)\right) -\frac{1}{2L}\nabla f(x_k)^\top\nabla f(x_k)\\
&=f(x_k)+\frac{L}{2}\left(\left(\boldsymbol{x}-\boldsymbol{x}_{k}\right)+\frac{1}{L} \nabla f\left(\boldsymbol{x}_{k}\right)\right)^{\top}\left(\left(\boldsymbol{x}-\boldsymbol{x}_{k}\right)+\frac{1}{L} \nabla f\left(\boldsymbol{x}_{k}\right)\right)-\frac{1}{2L}\nabla f(x_k)^\top\nabla f(x_k)\\
&=\frac{L}{2}\left\|\boldsymbol{x}-\left(\boldsymbol{x}_{k}-\frac{1}{L} \nabla f\left(\boldsymbol{x}_{k}\right)\right)\right\|_{2}^{2}+\mathrm{const}
\end{aligned}
$$
其中$\mathrm{const}=f(x_k)-\frac{1}{2 L} \nabla f\left(x_{k}\right)^{\top} \nabla f\left(x_{k}\right)$



### 11.11

$$
\boldsymbol{x}_{k+1}=\boldsymbol{x}_{k}-\frac{1}{L} \nabla f\left(\boldsymbol{x}_{k}\right)
$$

[解析]：这个很容易理解，因为2范数的最小值为0，当$\boldsymbol{x}_{k+1}=\boldsymbol{x}_{k}-\frac{1}{L} \nabla f\left(\boldsymbol{x}_{k}\right)$时，$\hat{f}(x_{k+1})\leqslant\hat{f}(x_k)$恒成立，同理$\hat{f}(x_{k+2})\leqslant\hat{f}(x_{k+1}), \cdots$，因此反复迭代能够使$\hat{f}(x)$的值不断下降。

### 11.12

$$
\boldsymbol{x}_{k+1}=\underset{\boldsymbol{x}}{\arg \min } \frac{L}{2}\left\|\boldsymbol{x}-\left(\boldsymbol{x}_{k}-\frac{1}{L} \nabla f\left(\boldsymbol{x}_{k}\right)\right)\right\|_{2}^{2}+\lambda\|\boldsymbol{x}\|_{1}
$$

[解析]：式11.11是用来优化$\hat{f}(x)$的，而对于式11.8，优化的函数为$f(x)+\lambda\left\Vert x\right\Vert_1$，由泰勒展开公式，优化的目标可近似为$\hat{f}(x)+\lambda\Vert x\Vert_1$，根据式11.10可知，$x$的更新由式11.12决定。

### 11.13

$$
\boldsymbol{x}_{k+1}=\underset{\boldsymbol{x}}{\arg \min } \frac{L}{2}\|\boldsymbol{x}-\boldsymbol{z}\|_{2}^{2}+\lambda\|\boldsymbol{x}\|
$$

[解析]：这里将式11.12的优化步骤拆分成了两步，首先令$z=x_{k}-\frac{1}{L} \nabla f\left(x_{k}\right)$以计算$z$，然后再求解式11.13，得到的结果是一致的。

### 11.14

$$
x_{k+1}^{i}=\left\{\begin{array}{ll}
{z^{i}-\lambda / L,} & {\lambda / L<z^{i}} \\
{0,} & {\left|z^{i}\right| \leqslant \lambda / L} \\
{z^{i}+\lambda / L,} & {z^{i}<-\lambda / L}
\end{array}\right.
$$

[解析]：令优化函数
$$
\begin{aligned}
g(\boldsymbol{x}) &=\frac{L}{2}\|\boldsymbol{x}-\boldsymbol{z}\|_{2}^{2}+\lambda\|\boldsymbol{x}\|_{1} \\
&=\frac{L}{2} \sum_{i=1}^{d}\left\|x^{i}-z^{i}\right\|_{2}^{2}+\lambda \sum_{i=1}^{d}\left\|x^{i}\right\|_{1} \\
&=\sum_{i=1}^{d}\left(\frac{L}{2}\left(x^{i}-z^{i}\right)^{2}+\lambda\left|x^{i}\right|\right)
\end{aligned}
$$

这个式子表明优化$g(\boldsymbol{x})$可以被拆解成优化$\boldsymbol{x}$的各个分量的形式，对分量$x_i$，其优化函数
$$
g\left(x^{i}\right)=\frac{L}{2}\left(x^{i}-z^{i}\right)^{2}+\lambda\left|x^{i}\right|
$$
求导得
$$
\frac{d g\left(x^{i}\right)}{d x^{i}}=L\left(x^{i}-z^{i}\right)+\lambda s g n\left(x^{i}\right)
$$
其中
$$
\operatorname{sign}\left(x^{i}\right)=\left\{\begin{array}{ll}
{1,} & {x^{i}>0} \\
{-1,} & {x^{i}<0}
\end{array}\right.
$$
称为[符号函数](https://en.wikipedia.org/wiki/Sign_function)，对于$x_i=0$的特殊情况，由于$\vert x_i \vert$在$x_i=0$点出不光滑，所以其不可导，需单独讨论。令$\frac{d g\left(x^{i}\right)}{d x^{i}}=0$有
$$
x^{i}=z^{i}-\frac{\lambda}{L} \operatorname{sign}\left(x^{i}\right)
$$
此式的解即为优化目标$g(x^i)$的极值点，因为等式两端均含有未知变量$x^i$，故分情况讨论。

1. 当$z^i>\frac{\lambda}{L}$时：

   a. 假设$x^i<0$，则$\operatorname{sign}(x^i)=-1$，那么有$x^i=z^i+\frac{\lambda}{L}>0$与假设矛盾；

   b. 假设$x^i>0$，则$\operatorname{sign}(x^i)=1$，那么有$x^i=z^i-\frac{\lambda}{L}<0$和假设相符和，下面来检验$x^i=z^i-\frac{\lambda}{L}$是否是使函数$g(x^i)$的取得最小值。当$x^i>0$时，
   $$
   \frac{d g\left(x^{i}\right)}{d x^{i}}=L\left(x^{i}-z^{i}\right)+\lambda
   $$
   在定义域内连续可导，则$g(x^i)$的二阶导数
   $$
   \frac{d^2 g\left(x^{i}\right)}{{d x^{i}}^2}=L
   $$
   由于$L$是Lipschitz常数恒大于0，因为$x^i=z^i-\frac{\lambda}{L}$是函数$g(x^i)$的最小值。

2. 当$z_i<-\frac{\lambda}{L}$时：

   a. 假设$x^i>0$，则$\operatorname{sign}(x^i)=1$，那么有$x^i=z^i-\frac{\lambda}{L}<0$与假设矛盾；

   b. 假设$x^i<0$，则$\operatorname{sign}(x^i)=-1$，那么有$x^i=z^i+\frac{\lambda}{L}<0$与假设相符，由上述二阶导数恒大于0可知，$x^i=z^i+\frac{\lambda}{L}$是$g(x^i)$的最小值。

3. 当$-\frac{\lambda}{L} \leqslant z_i \leqslant \frac{\lambda}{L}$时：

   a. 假设$x^i>0$，则$\operatorname{sign}(x^i)=1$，那么有$x^i=z^i-\frac{\lambda}{L}\leqslant 0$与假设矛盾；

   b. 假设$x^i<0$，则$\operatorname{sign}(x^i)=-1$，那么有$x^i=z^i+\frac{\lambda}{L}\geqslant 0$与假设矛盾。

4. 最后讨论$x_i=0$的情况，此时$g(x^i)=\frac{L}{2}\left({z^i}\right)^2$

   a. 当$\vert z^i\vert>\frac{\lambda}{L}$时，由上述推导可知$g(x_i)$的最小值在$x^i=z^i-\frac{\lambda}{L}$处取得，令
   $$
   \begin{aligned}
   f(x^i)&=g(x^i)\vert_{x^i=0}-g(x^i)\vert_{x_i=z^i-\frac{\lambda}{L}}\\
   &=\frac{L}{2}\left({z^i}\right)^2 - \left(\lambda z^i-\frac{\lambda^2}{2L}\right)\\
   &=\frac{L}{2}\left(z^i-\frac{\lambda}{L}\right)^2\\
   &>0
   \end{aligned}
   $$


   因此当$\vert z^i\vert>\frac{\lambda}{L}$时，$x_i=0$不会是函数$g(x_i)$的最小值。

   b. 当$-\frac{\lambda}{L} \leqslant z_i \leqslant \frac{\lambda}{L}$时，对于任何$\Delta x\neq 0$有
$$
   \begin{aligned}
   g(\Delta x) &=\frac{L}{2}\left(\Delta x-z^{i}\right)^{2}+\lambda|\Delta x| \\
   &=\frac{L}{2}\left((\Delta x)^{2}-2 \Delta x \cdot z^{i}+\frac{2 \lambda}{L}|\Delta x|\right)+\frac{L}{2}\left(z^{i}\right)^{2} \\
   &>\frac{L}{2}\left((\Delta x)^{2}-2 \Delta x \cdot z^{i}+\frac{2 \lambda}{L}\Delta x\right)+\frac{L}{2}\left(z^{i}\right)^{2}\\
   &>\frac{L}{2}\left(\Delta x\right)^2+\frac{L}{2}\left(z^{i}\right)^{2}\\
   &>g(x^i)\vert_{x^i=0}
   \end{aligned}
$$
   因此$x^i=0$是$g(x^i)$的最小值点。

5. 综上所述，11.14成立



### 11.15

$$
\min _{\mathbf{B}, \boldsymbol{\alpha}_{i}} \sum_{i=1}^{m}\left\|\boldsymbol{x}_{i}-\mathbf{B} \boldsymbol{\alpha}_{i}\right\|_{2}^{2}+\lambda \sum_{i=1}^{m}\left\|\boldsymbol{\alpha}_{i}\right\|_{1}
$$

[解析]：这个式子表达的意思很容易理解，即希望样本$x_i$的稀疏表示$\boldsymbol{\alpha}_i$通过字典$\mathbf{B}$重构后和样本$x_i$的原始表示尽量相似，如果满足这个条件，那么稀疏表示$\boldsymbol{\alpha}_i$是比较好的。后面的1范数项是为了使表示更加稀疏。

### 11.16

$$
\min _{\boldsymbol{\alpha}_{i}}\left\|\boldsymbol{x}_{i}-\mathbf{B} \boldsymbol{\alpha}_{i}\right\|_{2}^{2}+\lambda\left\|\boldsymbol{\alpha}_{i}\right\|_{1}
$$

[解析]：为了优化11.15，我们采用变量交替优化的方式(有点类似EM算法)，首先固定变量$\mathbf{B}$，则11.15求解的是$m$个样本相加的最小值，因为公式里没有样本之间的交互(即文中所述$\alpha_{i}^{u} \alpha_{i}^{v}(u \neq v)$这样的形式)，因此可以对每个变量做分别的优化求出$\boldsymbol{\alpha}_i$，求解方法见11.13，11.14。

### 11.17

$$
\min _{\mathbf{B}}\|\mathbf{X}-\mathbf{B} \mathbf{A}\|_{F}^{2}
$$

[解析]：这是优化11.15的第二步，固定住$\boldsymbol{\alpha}_i, i=1, 2,\dots,m$，此时式11.15的第二项为一个常数，优化11.15即优化$\min _{\mathbf{B}} \sum_{i=1}^{m}\left\|\boldsymbol{x}_{i}-\mathbf{B} \boldsymbol{\alpha}_{i}\right\|_{2}^{2}$。其写成矩阵相乘的形式为$\min _{\mathbf{B}}\|\mathbf{X}-\mathbf{B} \mathbf{A}\|_{2}^{2}$，将2范数扩展到$F$范数即得优化目标为$\min _{\mathbf{B}}\|\mathbf{X}-\mathbf{B} \mathbf{A}\|_{F}^{2}$。

### 11.18

$$
\begin{aligned}
\min _{\mathbf{B}}\|\mathbf{X}-\mathbf{B} \mathbf{A}\|_{F}^{2} &=\min _{\boldsymbol{b}_{i}}\left\|\mathbf{X}-\sum_{j=1}^{k} \boldsymbol{b}_{j} \boldsymbol{\alpha}^{j}\right\|_{F}^{2} \\
&=\min _{\boldsymbol{b}_{i}}\left\|\left(\mathbf{X}-\sum_{j \neq i} \boldsymbol{b}_{j} \boldsymbol{\alpha}^{j}\right)-\boldsymbol{b}_{i} \boldsymbol{\alpha}^{i}\right\| _{F}^{2} \\
&=\min _{\boldsymbol{b}_{i}}\left\|\mathbf{E}_{i}-\boldsymbol{b}_{i} \boldsymbol{\alpha}^{i}\right\|_{F}^{2}
\end{aligned}
$$

[解析]：这个公式难点在于推导$\mathbf{B}\mathbf{A}=\sum_{j=1}^k\boldsymbol{b}_j\boldsymbol{\alpha}^j$。大致的思路是$\boldsymbol{b}_{j} \boldsymbol{\alpha}^{j}$会生成和矩阵$\mathbf{B}\mathbf{A}$同样维度的矩阵，这个矩阵对应位置的元素是$\mathbf{B}\mathbf{A}$中对应位置元素的一个分量，这样的分量矩阵一共有$k$个，把所有分量矩阵加起来就得到了最终结果。推导过程如下：
$$
\begin{aligned}
\boldsymbol B\boldsymbol A
& =\begin{bmatrix}
b_{1}^{1} &b_{2}^{1}  & \cdot  & \cdot  & \cdot  & b_{k}^{1}\\
b_{1}^{2} &b_{2}^{2}  & \cdot  & \cdot  & \cdot  & b_{k}^{2}\\
\cdot  & \cdot  & \cdot  &  &  & \cdot \\
\cdot  &  \cdot &  & \cdot  &  &\cdot  \\
 \cdot & \cdot  &  &  & \cdot  & \cdot \\
 b_{1}^{d}& b_{2}^{d}  & \cdot  & \cdot  &\cdot   &  b_{k}^{d}
\end{bmatrix}_{d\times k}\cdot
\begin{bmatrix}
\alpha_{1}^{1} &\alpha_{2}^{1}  & \cdot  & \cdot  & \cdot  & \alpha_{m}^{1}\\
\alpha_{1}^{2} &\alpha_{2}^{2}  & \cdot  & \cdot  & \cdot  & \alpha_{m}^{2}\\
\cdot  & \cdot  & \cdot  &  &  & \cdot \\
\cdot  &  \cdot &  & \cdot  &  &\cdot  \\
 \cdot & \cdot  &  &  & \cdot  & \cdot \\
 \alpha_{1}^{k}& \alpha_{2}^{k}  & \cdot  & \cdot  &\cdot   &  \alpha_{m}^{k}
\end{bmatrix}_{k\times m} \\
& =\begin{bmatrix}
\sum_{j=1}^{k}b_{j}^{1}\alpha _{1}^{j} &\sum_{j=1}^{k}b_{j}^{1}\alpha _{2}^{j} & \cdot  & \cdot  & \cdot  & \sum_{j=1}^{k}b_{j}^{1}\alpha _{m}^{j}\\
\sum_{j=1}^{k}b_{j}^{2}\alpha _{1}^{j} &\sum_{j=1}^{k}b_{j}^{2}\alpha _{2}^{j}  & \cdot  & \cdot  & \cdot  & \sum_{j=1}^{k}b_{j}^{2}\alpha _{m}^{j}\\
\cdot  & \cdot  & \cdot  &  &  & \cdot \\
\cdot  &  \cdot &  & \cdot  &  &\cdot  \\
 \cdot & \cdot  &  &  & \cdot  & \cdot \\
\sum_{j=1}^{k}b_{j}^{d}\alpha _{1}^{j}& \sum_{j=1}^{k}b_{j}^{d}\alpha _{2}^{j}  & \cdot  & \cdot  &\cdot   &  \sum_{j=1}^{k}b_{j}^{d}\alpha _{m}^{j}
\end{bmatrix}_{d\times m} &
\end{aligned}
$$

$$
\begin{aligned}
\boldsymbol b_{\boldsymbol j}\boldsymbol \alpha ^{\boldsymbol j}
& =\begin{bmatrix}
b_{j}^{1}\\ b_{j}^{2}
\\ \cdot
\\ \cdot
\\ \cdot
\\ b_{j}^{d}
\end{bmatrix}\cdot
\begin{bmatrix}
 \alpha _{1}^{j}& \alpha _{2}^{j} & \cdot  & \cdot  & \cdot  & \alpha _{m}^{j}
\end{bmatrix}\\
& =\begin{bmatrix}
b_{j}^{1}\alpha _{1}^{j} &b_{j}^{1}\alpha _{2}^{j} & \cdot  & \cdot  & \cdot  & b_{j}^{1}\alpha _{m}^{j}\\
b_{j}^{2}\alpha _{1}^{j} &b_{j}^{2}\alpha _{2}^{j}  & \cdot  & \cdot  & \cdot  & b_{j}^{2}\alpha _{m}^{j}\\
\cdot  & \cdot  & \cdot  &  &  & \cdot \\
\cdot  &  \cdot &  & \cdot  &  &\cdot  \\
 \cdot & \cdot  &  &  & \cdot  & \cdot \\
b_{j}^{d}\alpha _{1}^{j}& b_{j}^{d}\alpha _{2}^{j}  & \cdot  & \cdot  &\cdot   &  b_{j}^{d}\alpha _{m}^{j}
\end{bmatrix}_{d\times m} &
\end{aligned}
$$

求和可得：
$$
\begin{aligned}
\sum_{j=1}^{k}\boldsymbol b_{\boldsymbol j}\boldsymbol \alpha ^{\boldsymbol j}
& = \sum_{j=1}^{k}\left (\begin{bmatrix}
b_{1}^{j}\\ b_{w}^{j}
\\ \cdot
\\ \cdot
\\ \cdot
\\ b_{d}^{j}
\end{bmatrix}\cdot
\begin{bmatrix}
 \alpha _{1}^{j}& \alpha _{2}^{j} & \cdot  & \cdot  & \cdot  & \alpha _{m}^{j}
\end{bmatrix} \right )\\
& =\begin{bmatrix}
\sum_{j=1}^{k}b_{j}^{1}\alpha _{1}^{j} &\sum_{j=1}^{k}b_{j}^{1}\alpha _{2}^{j} & \cdot  & \cdot  & \cdot  & \sum_{j=1}^{k}b_{j}^{1}\alpha _{m}^{j}\\
\sum_{j=1}^{k}b_{j}^{2}\alpha _{1}^{j} &\sum_{j=1}^{k}b_{j}^{2}\alpha _{2}^{j}  & \cdot  & \cdot  & \cdot  & \sum_{j=1}^{k}b_{j}^{2}\alpha _{m}^{j}\\
\cdot  & \cdot  & \cdot  &  &  & \cdot \\
\cdot  &  \cdot &  & \cdot  &  &\cdot  \\
 \cdot & \cdot  &  &  & \cdot  & \cdot \\
\sum_{j=1}^{k}b_{j}^{d}\alpha _{1}^{j}& \sum_{j=1}^{k}b_{j}^{d}\alpha _{2}^{j}  & \cdot  & \cdot  &\cdot   &  \sum_{j=1}^{k}b_{j}^{d}\alpha _{m}^{j}
\end{bmatrix}_{d\times m} &
\end{aligned}
$$
得证。

将矩阵$\mathbf{B}$分解成矩阵列$\boldsymbol{b}_j,j=1,2,\dots,k$带来一个好处，即和11.16的原理相同，矩阵列与列之间无关，因此可以分别优化各个列，即将$\min_\mathbf{B}\Vert\dots\mathbf{B}\dots\Vert^2_F$转化成了$\min_{b_i}\Vert\cdots\boldsymbol{b}_i\cdots\Vert^2_F$，得到第三行的等式之后，再利用文中介绍的KSVD算法求解即可。
