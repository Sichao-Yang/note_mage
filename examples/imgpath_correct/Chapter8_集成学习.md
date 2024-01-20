上篇主要介绍了鼎鼎大名的EM算法，从算法思想到数学公式推导（边际似然引入隐变量，Jensen不等式简化求导），EM算法实际上可以理解为一种坐标下降法，首先固定一个变量，接着求另外变量的最优解，通过其优美的"两步走"策略能较好地估计隐变量的值。本篇将继续讨论下一类经典算法--集成学习。

# 8. 集成学习

顾名思义，集成学习（ensemble learning）指的是将多个学习器进行有效地结合，组建一个"学习器委员会"，其中每个学习器担任委员会成员并行使投票表决权，使得委员会最后的决定更能够四方造福普度众生~...~，即其泛化性能要能优于其中任何一个学习器。

## 8.1 个体与集成

集成学习的基本结构为：先产生一组个体学习器，再使用某种策略将它们结合在一起。集成模型如下图所示：

![1.png](media\Chapter8_集成学习\5bc84d0c15683.png)

在上图的集成模型中，若个体学习器都属于同一类别，例如都是决策树或都是神经网络，则称该集成为同质的（homogeneous）;若个体学习器包含多种类型的学习算法，例如既有决策树又有神经网络，则称该集成为异质的（heterogenous）。


> **同质集成**：个体学习器称为"基学习器"（base learner），对应的学习算法为"基学习算法"（base learning algorithm）。
> **异质集成**：个体学习器称为"组件学习器"（component learner）或直称为"个体学习器"。


上面我们已经提到要让集成起来的泛化性能比单个学习器都要好，虽说团结力量大但也有木桶短板理论调皮捣蛋，那如何做到呢？这就引出了集成学习的两个重要概念：**准确性**和**多样性**（diversity）。准确性指的是个体学习器不能太差，要有一定的准确度；多样性则是个体学习器之间的输出要具有差异性。通过下面的这三个例子可以很容易看出这一点，准确度较高，差异度也较高，可以较好地提升集成性能。

![2.png](media\Chapter8_集成学习\5bc84d0d23e13.png)

现在考虑二分类的简单情形，假设基分类器之间相互独立（能提供较高的差异度），且错误率相等为 ε，则可以将集成器的预测看做一个伯努利实验，易知当所有基分类器中不足一半预测正确的情况下，集成器预测错误，所以集成器的错误率可以计算为：

![3.png](media\Chapter8_集成学习\5bc84d0cce0bb.png)

此时，集成器错误率随着基分类器的个数的增加呈指数下降，但前提是基分类器之间相互独立，然而在现实任务中，个体学习器是为解决同一个问题训练出来的，它们显然不可能相互独立。因此，**个体学习器的"准确性"和"差异性"本身就是一对矛盾的变量**，准确性高意味着牺牲多样性，所以产生"**好而不同**"的个体学习器正是集成学习研究的核心。现阶段有三种主流的集成学习方法：Boosting、Bagging以及随机森林（Random Forest），接下来将进行逐一介绍。

## 8.2 Boosting

Boosting是一种 **串行** 的工作机制，即个体学习器的训练存在依赖关系，必须一步一步序列化进行。其基本思想是：增加前一个基学习器在训练训练过程中预测错误样本的权重，使得后续基学习器更加关注这些打标错误的训练样本，尽可能纠正这些错误，一直向下串行直至产生需要的T个基学习器，Boosting最终对这T个学习器进行加权结合，产生学习器委员会。

Boosting族算法最著名、使用最为广泛的就是AdaBoost，因此下面主要是对AdaBoost算法进行介绍。AdaBoost使用的是**指数损失函数**，因此AdaBoost的权值与样本分布的更新都是围绕着最小化指数损失函数进行的。看到这里回想一下之前的机器学习算法，**我们可以发现机器学习的大部分模型只是改变了最优化目标中的损失函数**：如果是Square loss，那就是最小二乘了；如果是Hinge Loss，那就是著名的SVM了；如果是log-Loss，那就是Logistic Regression了。

定义基学习器的集成为加权结合，则有：

$$
H(\boldsymbol{x})=\sum_{t=1}^{T} \alpha_{t} h_{t}(\boldsymbol{x})
$$
AdaBoost算法的指数损失函数定义为：

![5.png](media\Chapter8_集成学习\5bc84d0d10461.png)

具体说来，整个Adaboost 迭代算法分为3步：

- 初始化训练数据的权值分布。如果有N个样本，则每一个训练样本最开始时都被赋予相同的权值：1/N。
- 训练弱分类器。具体训练过程中，如果某个样本点已经被准确地分类，那么在构造下一个训练集中，它的概率分布权值就被降低；相反，如果某个样本点没有被准确地分类，那么它的权值就得到提高。然后，权值更新过的样本集被用于训练下一个分类器，整个训练过程如此迭代地进行下去。
- 当某个分类器的误差率大于0.5，也就是说分得还不如瞎猜好的话，或者T个分类器已经训练完成，就结束分类过程。最后我们按照调整每一次样本分布的权重αt将各个弱分类器ht集成成强分类器H。

整个AdaBoost的算法流程如下所示：

![6.png](media\Chapter8_集成学习\5bc84d0d7c057.png)

可以看出：AdaBoost的核心就是基于上一次分类错误的经验来调整样本权重分布进而学到专门解决这些错误样本的新的分类器。这是一个顺序的过程，类似于子空间下降法（CG），在每一次展开的子空间里找到最优解。

Boosting算法要求基学习器能对特定分布的数据进行学习，即每次都更新样本分布权重，这里书上提到了两种方法："重赋权法"（re-weighting）和"重采样法"（re-sampling）


> **重赋权法** : 对每个样本附加一个权重，这时涉及到样本属性与标签的计算，都需要乘上一个权值。
> **重采样法** : 对于一些无法接受带权样本的及学习算法，适合用"重采样法"进行处理。方法大致过程是，根据各个样本的权重，对训练数据进行重采样，初始时样本权重一样，每个样本被采样到的概率一致，每次从N个原始的训练样本中按照权重有放回采样N个样本作为训练集，然后计算训练集错误率，然后调整权重，重复采样，集成多个基学习器。


从偏差-方差分解来看：Boosting算法主要关注于降低偏差，每轮的迭代都关注于训练过程中预测错误的样本，将弱学习提升为强学习器。从AdaBoost的算法流程来看，标准的AdaBoost只适用于二分类问题。另外AdaBoost也是数据挖掘的十大算法。

### 8.1

$$
P\left(h_{i}(\boldsymbol{x}) \neq f(\boldsymbol{x})\right)=\epsilon
$$

[解析]：$h_{i}(\boldsymbol{x})$是编号为$i$的基分类器给$x$的预测标记，$f(\boldsymbol{x})$是$x$的真实标记，它们之间不一致的概率记为$\epsilon$。

### 8.2

$$
H(\boldsymbol{x})=\operatorname{sign}\left(\sum_{i=1}^{T} h_{i}(\boldsymbol{x})\right)
$$

[解析]：$h_i(\boldsymbol{x})$当把$\boldsymbol{x}$分成1时，$h_i(\boldsymbol{x})=1$，否则$h_i(\boldsymbol{x})=-1$。各个基分类器$h_i$的分类结果求和之后数字的正、负或0，代表投票法产生的结果，即"少数服从多数"，符号函数$\operatorname{sign}$，将正数变成1，负数变成-1，0仍然是0，所以$H(\boldsymbol{x})$是由投票法产生的分类结果。

### 8.3

$$
\begin{aligned} P(H(\boldsymbol{x}) \neq f(\boldsymbol{x})) &=\sum_{k=0}^{\lfloor T / 2\rfloor} \left( \begin{array}{c}{T} \\ {k}\end{array}\right)(1-\epsilon)^{k} \epsilon^{T-k} \\ & \leqslant \exp \left(-\frac{1}{2} T(1-2 \epsilon)^{2}\right) \end{aligned}
$$

[推导]：由基分类器相互独立，假设随机变量$X$为$T$个基分类器分类正确的次数，因此$\mathrm{X} \sim \mathrm{B}(\mathrm{T}, 1-\mathrm{\epsilon})$，设$x_i$为每一个分类器分类正确的次数，则$x_i\sim \mathrm{B}(1, 1-\mathrm{\epsilon})（i=1，2，3，...，\mathrm{T}）$，那么有
$$
\begin{aligned}
\mathrm{X}&=\sum_{i=1}^{\mathrm{T}} x_i\\
\mathbb{E}(X)&=\sum_{i=1}^{\mathrm{T}}\mathbb{E}(x_i)=(1-\epsilon)T
\end{aligned}
$$
则：
$$
\begin{aligned} P(H(x) \neq f(x))=& P(X \leq\lfloor T / 2\rfloor) \\ & \leqslant P(X \leq T / 2)
\\ & =P\left[X-(1-\epsilon) T \leqslant \frac{T}{2}-(1-\epsilon) T\right]
\\ & =P\left[X-
(1-\epsilon) T \leqslant -\frac{T}{2}\left(1-2\epsilon\right)]\right]
\\ &=P\left[\sum_{i=1}^{\mathrm{T}} x_i-
\sum_{i=1}^{\mathrm{T}}\mathbb{E}(x_i) \leqslant -\frac{T}{2}\left(1-2\epsilon\right)]\right]
\\ &=P\left[\frac{1}{\mathrm{T}}\sum_{i=1}^{\mathrm{T}} x_i-\frac{1}{\mathrm{T}}
\sum_{i=1}^{\mathrm{T}}\mathbb{E}(x_i) \leqslant -\frac{1}{2}\left(1-2\epsilon\right)]\right]
 \end{aligned}
$$
根据Hoeffding不等式知
$$
P\left(\frac{1}{m} \sum_{i=1}^{m} x_{i}-\frac{1}{m} \sum_{i=1}^{m} \mathbb{E}\left(x_{i}\right) \leqslant -\delta\right) \leqslant  \exp \left(-2 m \delta^{2}\right)
$$
令$\delta=\frac {(1-2\epsilon)}{2},m=T$得
$$
\begin{aligned} P(H(\boldsymbol{x}) \neq f(\boldsymbol{x})) &=\sum_{k=0}^{\lfloor T / 2\rfloor} \left( \begin{array}{c}{T} \\ {k}\end{array}\right)(1-\epsilon)^{k} \epsilon^{T-k} \\ & \leqslant \exp \left(-\frac{1}{2} T(1-2 \epsilon)^{2}\right) \end{aligned}
$$

### 8.4

$$
H(\boldsymbol{x})=\sum_{t=1}^{T} \alpha_{t} h_{t}(\boldsymbol{x})
$$



[解析]：这个式子是集成学习的加性模型，加性模型不采用梯度下降的思想，而是$H(\boldsymbol{x})=\sum_{t=1}^{T-1} \alpha_{t} h_{t}(\boldsymbol{x})+\alpha_{T}h_{T}(\boldsymbol{x})$每次更新求解一个理论上最优的$h_T$（见式8.18）和$\alpha_T$（见式8.11）

### 8.5

$$
\ell_{\mathrm{exp}}(H | \mathcal{D})=\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H(\boldsymbol{x})}\right]
$$

[解析]：由式(8.4)知
$$
H(\boldsymbol{x})=\sum_{t=1}^{T} \alpha_{t} h_{t}(\boldsymbol{x})
$$
又由式(8.11)可知
$$
\alpha_{t}=\frac{1}{2} \ln \left(\frac{1-\epsilon_{t}}{\epsilon_{t}}\right)
$$
由$\ln$函数的单调性可知，该分类器的权重只与分类器的错误率负相关(即错误率越大，权重越低)，下面解释指数损失函数的意义：

1. 先考虑指数损失函数$e^{-f(x) H(x)}$的含义：$f$为真实函数，对于样本$x$来说，$f(\boldsymbol{x}) \in\{+1,-1\}$只能取$+1$和$-1$，而$H(\boldsymbol{x})$是一个实数；
   当$H(\boldsymbol{x})$的符号与$f(x)$一致时，$f(\boldsymbol{x}) H(\boldsymbol{x})>0$，因此$e^{-f(\boldsymbol{x}) H(\boldsymbol{x})}=e^{-|H(\boldsymbol{x})|}<1$，且$|H(\boldsymbol{x})|$越大指数损失函数$e^{-f(\boldsymbol{x}) H(\boldsymbol{x})}$越小（这很合理：此时$|H(\boldsymbol{x})|$越大意味着分类器本身对预测结果的信心越大，损失应该越小；若$|H(\boldsymbol{x})|$在零附近，虽然预测正确，但表示分类器本身对预测结果信心很小，损失应该较大）；
   当$H(\boldsymbol{x})$的符号与$f(\boldsymbol{x})$不一致时，$f(\boldsymbol{x}) H(\boldsymbol{x})<0$，因此$e^{-f(\boldsymbol{x}) H(\boldsymbol{x})}=e^{|H(\boldsymbol{x})|}>1$，且$| H(\boldsymbol{x}) |$越大指数损失函数越大（这很合理：此时$| H(\boldsymbol{x}) |$越大意味着分类器本身对预测结果的信心越大，但预测结果是错的，因此损失应该越大；若$| H(\boldsymbol{x}) |$在零附近，虽然预测错误，但表示分类器本身对预测结果信心很小，虽然错了，损失应该较小）；
2. 符号$\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}[\cdot]$的含义：$\mathcal{D}$为概率分布，可简单理解为在数据集$D$中进行一次随机抽样，每个样本被取到的概率；$\mathbb{E}[\cdot]$为经典的期望，则综合起来$\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}[\cdot]$表示在概率分布$\mathcal{D}$上的期望，可简单理解为对数据集$D$以概率$\mathcal{D}$进行加权后的期望。即
   $$\begin{aligned}
   \ell_{\exp }(H | \mathcal{D}) &=\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H(\boldsymbol{x})}\right] \\
   &=\sum_{\boldsymbol{x} \in D} \mathcal{D}(\boldsymbol{x}) e^{-f(\boldsymbol{x}) H(\boldsymbol{x})}
   \end{aligned}$$

### 8.6

$$
\frac{\partial \ell_{\exp }(H | \mathcal{D})}{\partial H(\boldsymbol{x})}=-e^{-H(\boldsymbol{x})} P(f(\boldsymbol{x})=1 | \boldsymbol{x})+e^{H(\boldsymbol{x})} P(f(\boldsymbol{x})=-1 | \boldsymbol{x})
$$

[解析]：由公式(8.5)中对于符号$\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}[\cdot]$的解释可知
$$
\begin{aligned}
\ell_{\exp }(H | \mathcal{D}) &=\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H(\boldsymbol{x})}\right] \\
&=\sum_{\boldsymbol{x} \in D} \mathcal{D}(\boldsymbol{x}) e^{-f(\boldsymbol{x}) H(\boldsymbol{x})} \\
&=\sum_{i=1}^{|D|} \mathcal{D}\left(\boldsymbol{x}_{i}\right)\left(e^{-H\left(\boldsymbol{x}_{i}\right)} \mathbb{I}\left(f\left(\boldsymbol{x}_{i}\right)=1\right)+e^{H\left(\boldsymbol{x}_{i}\right)} \mathbb{I}\left(f\left(\boldsymbol{x}_{i}\right)=-1\right)\right)\\
&=e^{-H\left(\boldsymbol{x}_{i}\right)} P\left(f\left(\boldsymbol{x}_{i}\right)=1 | \boldsymbol{x}_{i}\right)+e^{H\left(\boldsymbol{x}_{i}\right)} P\left(f\left(\boldsymbol{x}_{i}\right)=-1 | \boldsymbol{x}_{i}\right)
\end{aligned}
$$

因此

$$
\frac{\partial \ell_{\exp }(H | \mathcal{D})}{\partial H(\boldsymbol{x})}=-e^{-H(\boldsymbol{x})} P(f(\boldsymbol{x})=1 | \boldsymbol{x})+e^{H(\boldsymbol{x})} P(f(\boldsymbol{x})=-1 | \boldsymbol{x})
$$

### 8.7

$$
H(\boldsymbol{x})=\frac{1}{2} \ln \frac{P(f(x)=1 | \boldsymbol{x})}{P(f(x)=-1 | \boldsymbol{x})}
$$

[解析]：令式(8.6)等于0，移项并分离$H(\boldsymbol{x})$，即可得到式(8.7)。

### 8.8

$$
\begin{aligned}
\operatorname{sign}(H(\boldsymbol{x}))&=\operatorname{sign}\left(\frac{1}{2} \ln \frac{P(f(x)=1 | \boldsymbol{x})}{P(f(x)=-1 | \boldsymbol{x})}\right)
\\ & =\left\{\begin{array}{ll}{1,} & {P(f(x)=1 | \boldsymbol{x})>P(f(x)=-1 | \boldsymbol{x})} \\ {-1,} & {P(f(x)=1 | \boldsymbol{x})<P(f(x)=-1 | \boldsymbol{x})}\end{array}\right.
\\ & =\underset{y \in\{-1,1\}}{\arg \max } P(f(x)=y | \boldsymbol{x})
\end{aligned}
$$

[解析]：第一行到第二行显然成立，第二行到第三行是利用了$\arg\max$函数的定义。$\underset{y \in\{-1,1\}}{\arg \max } P(f(x)=y | \boldsymbol{x})$表示使得函数$P(f(x)=y | \boldsymbol{x})$取得最大值的$y$的值，展开刚好是第二行的式子。

### 8.9

$$
\begin{aligned}
\ell_{\exp }\left(\alpha_{t} h_{t} | \mathcal{D}_{t}\right) &=\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}_{t}}\left[e^{-f(\boldsymbol{x}) \alpha_{t} h_{t}(\boldsymbol{x})}\right] \\
&=\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}_{t}}\left[e^{-\alpha_{t}} \mathbb{I}\left(f(\boldsymbol{x})=h_{t}(\boldsymbol{x})\right)+e^{\alpha_{t}} \mathbb{I}\left(f(\boldsymbol{x}) \neq h_{t}(\boldsymbol{x})\right)\right] \\
&=e^{-\alpha_{t}} P_{\boldsymbol{x} \sim \mathcal{D}_{t}}\left(f(\boldsymbol{x})=h_{t}(\boldsymbol{x})\right)+e^{\alpha_{t}} P_{\boldsymbol{x} \sim \mathcal{D}_{t}}\left(f(\boldsymbol{x}) \neq h_{t}(\boldsymbol{x})\right) \\
&=e^{-\alpha_{t}}\left(1-\epsilon_{t}\right)+e^{\alpha_{t}} \epsilon_{t}
\end{aligned}
$$

[解析]：$\epsilon_t$与式(8.1)一致，表示$h_t(\boldsymbol{x})$分类错误的概率。

### 8.10

$$
\frac{\partial \ell_{\exp }\left(\alpha_{t} h_{t} | \mathcal{D}_{t}\right)}{\partial \alpha_{t}}=-e^{-\alpha_{t}}\left(1-\epsilon_{t}\right)+e^{\alpha_{t}} \epsilon_{t}
$$

[解析]：指数损失函数对$\alpha_t$求偏导，为了得到使得损失函数取最小值时$\alpha_t$的值。

### 8.11

$$
\alpha_{t}=\frac{1}{2} \ln \left(\frac{1-\epsilon_{t}}{\epsilon_{t}}\right)
$$

[解析]：令公式(8.10)等于0移项即得到的该式。此时$\alpha_t$的取值使得该基分类器经$\alpha_t$加权后的损失函数最小。

### 8.12

$$
\begin{aligned}
\ell_{\exp }\left(H_{t-1}+h_{t} | \mathcal{D}\right) &=\mathbb{E}_{x \sim \mathcal{D}}\left[e^{-f(x)\left(H_{t-1}(x)+h_{t}(x)\right)}\right] \\
&=\mathbb{E}_{x \sim \mathcal{D}}\left[e^{-f(x) H_{t-1}(x)} e^{-f(x) h_{t}(x)}\right]
\end{aligned}
$$

[解析]：将$H_{t}(\boldsymbol{x})=H_{t-1}(\boldsymbol{x})+h_{t}(\boldsymbol{x})$带入公式(8.5)即可，因为理想的$h_t(\boldsymbol{x})$可以纠正理想的$h_t$可以纠正$H_{t-1}$的全部错误，所以权重系数为1。如果权重系数$\alpha_t$是个常数的话，对后续结果也没有影响。

### 8.13

$$
\ell_{\exp }\left(H_{t-1}+h_{t} | \mathcal{D}\right)=\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}\left(1-f(\boldsymbol{x}) h_{t}(\boldsymbol{x})+\frac{1}{2}\right)\right]
$$

[推导]：由$e^x$的二阶泰勒展开为$1+x+\frac{x^2}{2}+o(x^2)$得:
$$
\begin{aligned}
\ell_{\exp }\left(H_{t-1}+h_{t} | \mathcal{D}\right) &=\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})} e^{-f(\boldsymbol{x}) h_{t}(\boldsymbol{x})}\right]
\\ & \simeq \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}\left(1-f(\boldsymbol{x}) h_{t}(\boldsymbol{x})+\frac{f^{2}(\boldsymbol{x}) h_{t}^{2}(\boldsymbol{x})}{2}\right)\right]
\end{aligned}
$$
因为$f(\boldsymbol{x})$与$h_t(\boldsymbol{x})$取值都为1或-1，所以$f^2(\boldsymbol{x})=h_t^2(\boldsymbol{x})=1$，所以得:
$$
\ell_{\exp }\left(H_{t-1}+h_{t} | \mathcal{D}\right)= \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}\left(1-f(\boldsymbol{x}) h_{t}(\boldsymbol{x})+\frac{1}{2}\right)\right]
$$

### 8.14

$$
\begin{aligned}
h_{t}(\boldsymbol{x})&=\underset{h}{\arg \min } \ell_{\exp }\left(H_{t-1}+h | \mathcal{D}\right)\\
&=\underset{h}{\arg \min } \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}\left(1-f(\boldsymbol{x}) h(\boldsymbol{x})+\frac{1}{2}\right)\right]\\
&=\underset{h}{\arg \max } \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})} f(\boldsymbol{x}) h(\boldsymbol{x})\right]\\
&=\underset{h}{\arg \max } \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[\frac{e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}}{\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}\right]} f(\boldsymbol{x}) h(\boldsymbol{x})\right]
\end{aligned}
$$

[解析]：理想的$h_t(\boldsymbol{x})$是使得$H_{t}(\boldsymbol{x})$的指数损失函数取得最小值时的$h_t(\boldsymbol{x})$，该式将此转化成某个期望的最大值。第二个式子到第三个式子是因为$\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[\frac{3}{2} e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}\right]$与$h(\boldsymbol{x})$无关，是一个常数。第三个式子到最后一个式子是因为$\frac{1}{\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[\frac{3}{2} e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}\right]}$与$h(\boldsymbol{x})$无关因此可以引入进来。

### 8.16

$$
\begin{aligned} h_{t}(\boldsymbol{x}) &=\underset{h}{\arg \max } \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[\frac{e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}}{\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}\right]} f(\boldsymbol{x}) h(\boldsymbol{x})\right] \\ &=\underset{\boldsymbol{h}}{\arg \max } \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}_{t}}[f(\boldsymbol{x}) h(\boldsymbol{x})] \end{aligned}
$$

[推导]：假设$\boldsymbol{x}$的概率分布是$f(\boldsymbol{x})$
(注:本书中概率分布全都是$\mathcal{D(\boldsymbol{x})}$)
$$
\mathbb{E(g(\boldsymbol{x}))}=\sum_{i=1}^{|D|}f(\boldsymbol{x}_i)g(\boldsymbol{x}_i)
$$
故可得

$$
\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H(\boldsymbol{x})}\right]=\sum_{i=1}^{|D|} \mathcal{D}\left(\boldsymbol{x}_{i}\right) e^{-f\left(\boldsymbol{x}_{i}\right) H\left(\boldsymbol{x}_{i}\right)}
$$
由式(8.15)可知
$$
\mathcal{D}_{t}\left(\boldsymbol{x}_{i}\right)=\mathcal{D}\left(\boldsymbol{x}_{i}\right) \frac{e^{-f\left(\boldsymbol{x}_{i}\right) H_{t-1}\left(\boldsymbol{x}_{i}\right)}}{\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}\right]}
$$

所以式(8.16)可以表示为
$$
\begin{aligned} & \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[\frac{e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}}{\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}\right]} f(\boldsymbol{x}) h(\boldsymbol{x})\right] \\=& \sum_{i=1}^{|D|} \mathcal{D}\left(\boldsymbol{x}_{i}\right) \frac{e^{-f\left(\boldsymbol{x}_{i}\right) H_{t-1}\left(\boldsymbol{x}_{i}\right)}}{\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x}) }]  \right.}f(x_i)h(x_i) \\=& \sum_{i=1}^{|D|} \mathcal{D}_{t}\left(\boldsymbol{x}_{i}\right) f\left(\boldsymbol{x}_{i}\right) h\left(\boldsymbol{x}_{i}\right) \\=& \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}_{t}}[f(\boldsymbol{x}) h(\boldsymbol{x})] \end{aligned}
$$

### 8.17

$$
f(\boldsymbol{x}) h(\boldsymbol{x})=1-2 \mathbb{I}(f(\boldsymbol{x}) \neq h(\boldsymbol{x}))
$$

[解析]：当$f(\boldsymbol{x})=h(\boldsymbol{x})$时，$\mathbb{I}(f(\boldsymbol{x}) \neq h(\boldsymbol{x}))=0$，$f(\boldsymbol{x}) h(\boldsymbol{x})=1$，当$f(\boldsymbol{x})\neq h(\boldsymbol{x})$时，$\mathbb{I}(f(\boldsymbol{x}) \neq h(\boldsymbol{x}))=1$，$f(\boldsymbol{x}) h(\boldsymbol{x})=-1$。

### 8.18

$$
h_{t}(\boldsymbol{x})=\underset{h}{\arg \min } \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}_{t}}[\mathbb{I}(f(\boldsymbol{x}) \neq h(\boldsymbol{x}))]
$$

[解析]：由公式(8.16) 和公式(8.17)有：
$$
\begin{aligned}
h_{t}(\boldsymbol{x}) &=\arg \max _{h} \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}_{t}}[f(\boldsymbol{x}) h(\boldsymbol{x})] \\
&=\arg \max _{h}\left(1-2 \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}_{t}}[\mathbb{I}(f(\boldsymbol{x}) \neq h(\boldsymbol{x}))]\right) \\
&=\underset{h}{\arg \max }\left(-2 \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}_{t}}[\mathbb{I}(f(\boldsymbol{x}) \neq h(\boldsymbol{x}))]\right) \\
&=\arg \min \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}_{t}}[\mathbb{I}(f(\boldsymbol{x}) \neq h(\boldsymbol{x}))]
\end{aligned}
$$

### 8.19

$$
\begin{aligned}
\mathcal{D}_{t+1}(\boldsymbol{x}) &=\frac{\mathcal{D}(\boldsymbol{x}) e^{-f(\boldsymbol{x}) H_{t}(\boldsymbol{x})}}{\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t}(\boldsymbol{x})}\right]} \\
&=\frac{\mathcal{D}(\boldsymbol{x}) e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})} e^{-f(\boldsymbol{x}) \alpha_{t} h_{t}(\boldsymbol{x})}}{\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t}(\boldsymbol{x})}\right]} \\
&=\mathcal{D}_{t}(\boldsymbol{x}) \cdot e^{-f(\boldsymbol{x}) \alpha_{t} h_{t}(\boldsymbol{x})} \frac{\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}\right]}{\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t}(\boldsymbol{x})}\right]}
\end{aligned}
$$

[解析]：boosting算法是根据调整后的样本再去训练下一个基分类器，这就是"重赋权法"的样本分布的调整公式。

## 8.3 Bagging与Random Forest

相比之下，Bagging与随机森林算法就简洁了许多，上面已经提到产生"好而不同"的个体学习器是集成学习研究的核心，即在保证基学习器准确性的同时增加基学习器之间的多样性。而这两种算法的基本思（tao）想（lu）都是通过"自助采样"的方法来增加多样性。

### 8.3.1 Bagging

Bagging是一种并行式的集成学习方法，即基学习器的训练之间没有前后顺序可以同时进行，Bagging使用 **"有放回"** 采样的方式选取训练集，对于包含m个样本的训练集，进行m次有放回的随机采样操作，从而得到m个样本的采样集，这样训练集中有接近36.8%的样本没有被采到。按照相同的方式重复进行，我们就可以采集到T个包含m个样本的数据集，从而训练出T个基学习器，最终对这T个基学习器的输出进行结合。

$$
\lim _{m \rightarrow \infty}\left(1-\frac{1}{m}\right)^{m} \mapsto \frac{1}{e} \approx 0.368
$$
Bagging算法的流程如下所示：

![8.png](media\Chapter8_集成学习\5bc84d0d0e761.png)

可以看出Bagging主要通过**样本的扰动**来增加基学习器之间的多样性，因此Bagging的基学习器应为那些对训练集十分敏感的不稳定学习算法，例如：神经网络与决策树等。从偏差-方差分解来看，Bagging算法主要关注于降低方差，即通过多次重复训练提高稳定性。不同于AdaBoost的是，Bagging可以十分简单地移植到多分类、回归等问题。总的说起来则是：**AdaBoost关注于降低偏差，而Bagging关注于降低方差。**

### 8.20

$$
H^{oob}(\boldsymbol{x})=\underset{y \in \mathcal{Y}}{\arg \max } \sum_{t=1}^{\mathrm{T}} \mathbb{I}\left(h_{t}(\boldsymbol{x})=y\right) \cdot \mathbb{I}\left(\boldsymbol{x} \notin D_{t}\right)
$$

[解析]：$\mathbb{I}\left(h_{t}(\boldsymbol{x})=y\right)$表示对$\mathrm{T}$个基学习器，每一个都判断结果是否与$y$一致，$y$的取值一般是$-1$和$1$，如果基学习器结果与$y$一致，则$\mathbb{I}\left(h_{t}(\boldsymbol{x})=y\right)=1$，如果样本不在训练集内，则$\mathbb{I}\left(\boldsymbol{x} \notin D_{t}\right)=1$，综合起来看就是，对包外的数据，用"投票法"选择包外估计的结果，即1或-1。

### 8.21

$$
\epsilon^{o o b}=\frac{1}{|D|} \sum_{(\boldsymbol{x}, y) \in D} \mathbb{I}\left(H^{o o b}(\boldsymbol{x}) \neq y\right)
$$

[解析]：由8.20知，$H^{oob}(\boldsymbol{x})$是对包外的估计，该式表示估计错误的个数除以总的个数，得到泛化误差的包外估计。

### 8.3.2 随机森林

随机森林（Random Forest）是Bagging的一个拓展体，它的基学习器固定为决策树，多棵树也就组成了森林，而"随机"则在于选择划分属性的随机，随机森林在训练基学习器时，也采用有放回采样的方式添加样本扰动，同时它还引入了一种**属性扰动**，即在基决策树的训练过程中，在选择划分属性时，RF先从候选属性集中随机挑选出一个包含K个属性的子集，再从这个子集中选择最优划分属性，一般推荐K=log2（d）。

这样随机森林中基学习器的多样性不仅来自样本扰动，还来自属性扰动，从而进一步提升了基学习器之间的差异度。相比决策树的Bagging集成，随机森林的起始性能较差（由于属性扰动，基决策树的准确度有所下降），但随着基学习器数目的增多，随机森林往往会收敛到更低的泛化误差。同时不同于Bagging中决策树从所有属性集中选择最优划分属性，随机森林只在属性集的一个子集中选择划分属性，因此训练效率更高。

![9.png](media\Chapter8_集成学习\5bc84d0d7a4fd.png)

## 8.4 结合策略

结合策略指的是在训练好基学习器后，如何将这些基学习器的输出结合起来产生集成模型的最终输出，下面将介绍一些常用的结合策略：

### 8.4.1 平均法（回归问题）

![10.png](media\Chapter8_集成学习\5bc84d0d07983.png)

![11.png](media\Chapter8_集成学习\5bc84de1b74ff.png)

易知简单平均法是加权平均法的一种特例，加权平均法可以认为是集成学习研究的基本出发点。由于各个基学习器的权值在训练中得出，**一般而言，在个体学习器性能相差较大时宜使用加权平均法，在个体学习器性能相差较小时宜使用简单平均法**。

### 8.4.2 投票法（分类问题）

![12.png](media\Chapter8_集成学习\5bc84de2629c4.png)

![13.png](media\Chapter8_集成学习\5bc84de25a74b.png)

![14.png](media\Chapter8_集成学习\5bc84de1bacc4.png)

绝对多数投票法（majority voting）提供了拒绝选项，这在可靠性要求很高的学习任务中是一个很好的机制。同时，对于分类任务，各个基学习器的输出值有两种类型，分别为类标记和类概率。

![15.png](media\Chapter8_集成学习\5bc84de2768c1.png)

一些在产生类别标记的同时也生成置信度的学习器，置信度可转化为类概率使用，**一般基于类概率进行结合往往比基于类标记进行结合的效果更好**，需要注意的是对于异质集成，其类概率不能直接进行比较，此时需要将类概率转化为类标记输出，然后再投票。

### 8.4.3 学习法

学习法是一种更高级的结合策略，即学习出一种"投票"的学习器，Stacking是学习法的典型代表。Stacking的基本思想是：首先训练出T个基学习器，对于一个样本它们会产生T个输出，将这T个基学习器的输出作为新的样本，m个样本就会产生一个m*T的样本集，来训练一个新的对原始训练集的标记进行学习的学习器。投票学习器的输入属性与学习算法对Stacking集成的泛化性能有很大的影响，书中已经提到：**二层学习器采用类概率作为输入属性，选用多响应线性回归（MLR）一般会产生较好的效果**。

![16.png](media\Chapter8_集成学习\5bc84de25cbaf.png)

### 8.22

$$
H(\boldsymbol{x})=\frac{1}{T} \sum_{i=1}^{T} h_{i}(\boldsymbol{x})
$$

[解析]：对基分类器的结果进行简单的平均。

### 8.23

$$
H(\boldsymbol{x})=\sum_{i=1}^{T} w_{i} h_{i}(\boldsymbol{x})
$$

[解析]：对基分类器的结果进行加权平均。

### 8.24

$$
H(\boldsymbol{x})=\left\{\begin{array}{ll}
{c_{j},} & {\text { if } \sum_{i=1}^{T} h_{i}^{j}(\boldsymbol{x})>0.5 \sum_{k=1}^{N} \sum_{i=1}^{T} h_{i}^{k}(\boldsymbol{x})} \\
{\text { reject, }} & {\text { otherwise. }}
\end{array}\right.
$$

[解析]：当某一个类别$j$的基分类器的结果之和，大于所有结果之和的$\frac {1}{2}$，则选择该类别$j$为最终结果。

### 8.25

$$
H(\boldsymbol{x})=c_{\underset{j}{ \arg \max} \sum_{i=1}^{T} h_{i}^{j}(\boldsymbol{x})}
$$

[解析]：相比于其他类别，该类别$j$的基分类器的结果之和最大，则选择类别$j$为最终结果。

### 8.26

$$
H(\boldsymbol{x})=c_{\underset{j}{ \arg \max} \sum_{i=1}^{T} w_i h_{i}^{j}(\boldsymbol{x})}
$$

[解析]：相比于其他类别，该类别$j$的基分类器的结果之和最大，则选择类别$j$为最终结果，与式(8.25)不同的是，该式在基分类器前面乘上一个权重系数，该系数大于等于0，且T个权重之和为1。

## 8.5 多样性（diversity）

在集成学习中，基学习器之间的多样性是影响集成器泛化性能的重要因素。因此增加多样性对于集成学习研究十分重要，一般的思路是在学习过程中引入随机性，常见的做法主要是对数据样本、输入属性、输出表示、算法参数进行扰动。


> **数据样本扰动**，即利用具有差异的数据集来训练不同的基学习器。例如：有放回自助采样法，但此类做法只对那些不稳定学习算法十分有效，例如：决策树和神经网络等，训练集的稍微改变能导致学习器的显著变动。
> **输入属性扰动**，即随机选取原空间的一个子空间来训练基学习器。例如：随机森林，从初始属性集中抽取子集，再基于每个子集来训练基学习器。但若训练集只包含少量属性，则不宜使用属性扰动。
> **输出表示扰动**，此类做法可对训练样本的类标稍作变动，或对基学习器的输出进行转化。
> **算法参数扰动**，通过随机设置不同的参数，例如：神经网络中，随机初始化权重与随机设置隐含层节点数。

### 8.27

$$
A\left(h_{i} | \boldsymbol{x}\right)=\left(h_{i}(\boldsymbol{x})-H(\boldsymbol{x})\right)^{2}
$$

[解析]：该式表示个体学习器结果与预测结果的差值的平方，即为个体学习器的"分歧"。

### 8.28

$$
\begin{aligned}
\bar{A}(h | \boldsymbol{x}) &=\sum_{i=1}^{T} w_{i} A\left(h_{i} | \boldsymbol{x}\right) \\
&=\sum_{i=1}^{T} w_{i}\left(h_{i}(\boldsymbol{x})-H(\boldsymbol{x})\right)^{2}
\end{aligned}
$$

[解析]：该式表示对各个个体学习器的"分歧"加权平均的结果，即集成的"分歧"。

### 8.29

$$
E\left(h_{i} | \boldsymbol{x}\right)=\left(f(\boldsymbol{x})-h_{i}(\boldsymbol{x})\right)^{2}
$$

[解析]：该式表示个体学习器与真实值之间差值的平方，即个体学习器的平方误差。

### 8.30

$$
E(H | \boldsymbol{x})=(f(\boldsymbol{x})-H(\boldsymbol{x}))^{2}
$$

[解析]：该式表示集成与真实值之间差值的平方，即集成的平方误差。

### 8.31

$$
\bar{A}(h | \boldsymbol{x}) =\sum_{i=1}^{T} w_{i} E\left(h_{i} | \boldsymbol{x}\right)-E(H | \boldsymbol{x})
$$

[推导]：由(8.28)知
$$
\begin{aligned}
\bar{A}(h | \boldsymbol{x})&=\sum_{i=1}^{T} w_{i}\left(h_{i}(\boldsymbol{x})-H(\boldsymbol{x})\right)^{2}\\
&=\sum_{i=1}^{T} w_{i}(h_i(\boldsymbol{x})^2-2h_i(\boldsymbol{x})H(\boldsymbol{x})+H(\boldsymbol{x})^2)\\
&=\sum_{i=1}^{T} w_{i}h_i(\boldsymbol{x})^2-H(\boldsymbol{x})^2
\end{aligned}
$$

上面第三个等号来自于把hi sum起来后等于H，所以是H^2^-2H^2^

又因为
$$
\begin{aligned}
& \sum_{i=1}^{T} w_{i} E\left(h_{i} | \boldsymbol{x}\right)-E(H | \boldsymbol{x})\\
&=\sum_{i=1}^{T} w_{i}\left(f(\boldsymbol{x})-h_{i}(\boldsymbol{x})\right)^{2}-(f(\boldsymbol{x})-H(\boldsymbol{x}))^{2}\\
&=\sum_{i=1}^{T} w_{i}h_i(\boldsymbol{x})^2-H(\boldsymbol{x})^{2}
\end{aligned}
$$
上面第三个等号来自于把第二个等号里所有式子拆开再消元。

所以
$$
\bar{A}(h | \boldsymbol{x}) =\sum_{i=1}^{T} w_{i} E\left(h_{i} | \boldsymbol{x}\right)-E(H | \boldsymbol{x})
$$

### 8.32

$$
\sum_{i=1}^{T} w_{i} \int A\left(h_{i} | \boldsymbol{x}\right) p(\boldsymbol{x}) d \boldsymbol{x}=\sum_{i=1}^{T} w_{i} \int E\left(h_{i} | \boldsymbol{x}\right) p(\boldsymbol{x}) d \boldsymbol{x}-\int E(H | \boldsymbol{x}) p(\boldsymbol{x}) d \boldsymbol{x}
$$

[解析]：$\int A\left(h_{i} | \boldsymbol{x}\right) p(\boldsymbol{x}) d \boldsymbol{x}$表示个体学习器在全样本上的"分歧"，$\sum_{i=1}^{T} w_{i} \int A\left(h_{i} | \boldsymbol{x}\right) p(\boldsymbol{x}) d \boldsymbol{x}$表示集成在全样本上的"分歧"，然后根据式(8.31)拆成误差的形式。

### 8.33

$$
E_{i}=\int E\left(h_{i} | \boldsymbol{x}\right) p(\boldsymbol{x}) d \boldsymbol{x}
$$

[解析]：表示个体学习器在全样本上的泛化误差。

### 8.34

$$
A_{i}=\int A\left(h_{i} | \boldsymbol{x}\right) p(\boldsymbol{x}) d \boldsymbol{x}
$$

[解析]：表示个体学习器在全样本上的分歧。

### 8.35

$$
E=\int E(H | \boldsymbol{x}) p(\boldsymbol{x}) d \boldsymbol{x}
$$

[解析]：表示集成在全样本上的泛化误差。

### 8.36

$$
E=\bar{E}-\bar{A}
$$

[解析]：$\bar{E}$表示个体学习器泛化误差的加权均值，$\bar{A}$表示个体学习器分歧项的加权均值，该式称为"误差-分歧分解"。

在此，集成学习就介绍完毕，看到这里，大家也会发现集成学习实质上是一种通用框架，可以使用任何一种基学习器，从而改进单个学习器的泛化性能。数据挖掘竞赛KDDCup历年的冠军几乎都使用了集成学习，看来的确是个大杀器。
