上篇主要介绍了神经网络。首先从生物学神经元出发，引出了它的数学抽象模型--MP神经元以及由两层神经元组成的感知机模型，并基于梯度下降的方法描述了感知机模型的权值调整规则。由于简单的感知机不能处理线性不可分的情形，因此接着引入了含隐层的前馈型神经网络，Back Propagation神经网络则是其中最为成功的一种学习方法，它使用误差逆传播的方法来逐层调节连接权。最后简单介绍了局部/全局最小以及目前十分火热的深度学习的概念。本篇围绕的核心则是曾经一度取代过神经网络的另一种监督学习算法--**支持向量机**（Support Vector Machine），简称**SVM**。

# **6 支持向量机**

支持向量机是一种经典的二分类模型，基本模型定义为特征空间中最大间隔的线性分类器，其学习的优化目标便是间隔最大化，因此支持向量机本身可以转化为一个凸二次规划求解的问题。

## **6.0 函数间隔与几何间隔**

对于二分类学习，假设现在的数据是线性可分的，这时分类学习最基本的想法就是找到一个合适的超平面，该超平面能够将不同类别的样本分开，类似二维平面使用ax+by+c=0来表示，超平面实际上表示的就是高维的平面，如下图所示：

![1.png](media\Chapter6_支持向量机\5bc72f6a2ec8a.png)

对数据点进行划分时，易知：当超平面距离与它最近的数据点的间隔越大，分类的鲁棒性越好，即当新的数据点加入时，超平面对这些点的适应性最强，出错的可能性最小。因此需要让所选择的超平面能够最大化这个间隔（如下图所示）， 常用的间隔定义有两种，一种称之为函数间隔，一种为几何间隔。它们相互间有很微妙的联系。

![2.png](media\Chapter6_支持向量机\5bc72f6a06d5a.png)

### **6.1.1 函数间隔**

（注：这里结合了西瓜书，机器学习，深度之眼，cs229的知识）

在超平面w'x+b=0确定的情况下，$|w^Tx^i+b|$ 能够代表具体的点x^i^距离超平面的远近，我们目标要让这个间隔最大化。但我们不想要abs| |这个符号，因为不好优化。所以引入了函数间隔（functional margin）:
$$
\hat{\gamma}^{i}=y^{i}\left(w^{T}x^{i}+b\right)
$$
假设，当w'x^i^+b>0时，表示预测出的x^i^在超平面的正侧。而实际的标签y^i^=1，那两种乘积就是一个正值，是一个距离值。说明**在正确预测的前提下**上面两个式子是等价的。同理当w'x^i^+b<0时，则表示x^i^在超平面的负侧，如果类别y^i^=-1。两者乘积也是正值。

超平面（w,b）每一个样本点都有一个间隔，我们用所有样本点(X^i^, Y^i^)(i=1,...,m)的函数间隔**最小值**作为超平面在训练数据集T上的函数间隔：

$$
\hat{\gamma}=\min _{i=1, \ldots, m} \hat{\gamma}^{i}
$$
然后，这样定义的函数间隔在处理SVM上会有一个核心的问题，当超平面的两个参数w和b同比例改变时，函数间隔大小也会跟着改变，但是实际上超平面还是原来的超平面，并没有变化。例如：w~1~x~1~+w~2~x~2~+w~3~x~3~+b=0其实等价于2w~1~x~1~+2w~2~x~2~+2w~3~x~3~+2b=0，但计算的函数间隔却翻了一倍。从而我们还需要引出另一个直接度量点到超平面距离的概念-几何间隔（geometrical margin）。

### **6.1.2 几何间隔**

几何间隔代表的则是数据点到超平面的实际距离。怎么得到这个距离呢？分三步走：

1. 证明w是超平面（红线）的法向量（绿线）
2. 求原点到超平面的距离（黑线）
3. 最后再找出样本点x~i~到超平面的距离（蓝虚线）

<img src="media\Chapter6_支持向量机\image-20200226234447841.png" alt="image-20200226234447841" style="zoom: 50%;" />

#### eq. 6.2

根据上图首先我们可以通过在超平面上任取两点来证明w是法向量：点x1减去x2属于超平面又和w垂直。
$$
\left\{\begin{array}{ll}
{\boldsymbol{w}^{T} x_{1}+b=0} \\
{\boldsymbol{w}^{T} x_{2}+b=0} \\
\boldsymbol{w}^{T}\left(x_{1}-x_{2}\right)=0
\end{array}\right.
$$
超平面到原点的距离：我们可以过原点做一条平面垂直线，然后找出当它和超平面相交时它的norm大小$\lambda$，这也就是距离。
$$
\left\{\begin{array}{ll}
{\boldsymbol{w}^{T} x+b=0} \\
{\boldsymbol{w}^{T} \lambda \frac{w}{\|\boldsymbol{w}\|}+b=0} \\
{\lambda=\frac{-b}{\|\boldsymbol{w}\|}}
\end{array}\right.
$$
点到超平面的距离：我们可以把它改写成（点到过原点的平行于超平面的平面的距离）-（超平面到原点的距离）：
$$
\gamma=\left\|\frac{\boldsymbol{w}^{T} \boldsymbol{x}}{\|\boldsymbol{w}\|^{2}} \boldsymbol{w}-\frac{-b}{\|\boldsymbol{w}\|^{2}} \|\boldsymbol{w}\|\right\|
$$
上式左边部分就是x在w上的投影，也就是点到过原点的超平面平行平面的距离。右边部分就是超平面到原点的距离。化简可得：
$$
\begin{aligned}
\gamma&=\left\|\frac{\boldsymbol{w}^{T} \boldsymbol{x}}{\|\boldsymbol{w}\|^{2}}+\frac{b}{\|\boldsymbol{w}\|^{2}}\right\|\|\boldsymbol{w}\|\\
&=\frac{\left\|\boldsymbol{w}^{T} \boldsymbol{x}+b\right\|}{\|\boldsymbol{w}\|}\\
&=\frac{\left|\boldsymbol{w}^{T} \boldsymbol{x}+b\right|}{\|\boldsymbol{w}\|}
\end{aligned}
$$
这个就是书上的公式6.2，也就是任意点到超平面的几何距离。

## **6.1 最大间隔与支持向量**

<img src="media\Chapter6_支持向量机\image-20200227001730191.png" alt="image-20200227001730191" style="zoom:50%;" />

上面的几何间隔里还是有一个绝对值，不好求导，所以我们改写成类似函数间隔的形式：我们乘上一个 $\hat{y}=\{-1, 1\}$ 就可以把绝对值拿掉：
$$
\begin{aligned}
\gamma^{i}=\frac{\widehat{y}^{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}^{i}+b\right)}{\|\boldsymbol{w}\|}
\end{aligned}
$$
比函数间隔更好的是，我们现在是invariant to rescaling of the parameters的。（当||w||=1的时候，functional margin和geometric margin是相等的）最后，我们可以把对几何间隔求最大值的问题写成以下的形式：
$$
\begin{array}{ll}
\max _{\gamma, w, b} \gamma\\
\text { s.t. } & y^{(i)}\left(w^{T} x^{(i)}+b\right) \geq \gamma, \quad i=1, \ldots, m \\
& \|w\|=1
\end{array}
$$
上式意思是我们的优化目标是一个几何间隔，这个间隔要比所有样本的函数间隔都要小。在约束条件里，不等式要成立右侧必须是函数间隔，单现在右侧是几何间隔。要他俩相等，我们必须加上||w||=1的约束。

#### eq. 6.6

但实际上，上面的约束条件||w||=1非常nasty（non-convex），所以我们可以通过$\gamma=\hat{\gamma} / ||w||$的关系把优化目标转换成函数间隔：
$$
\begin{aligned}
\max _{\hat{\gamma}, w, b} & \frac{\hat{\gamma}}{\|w\|} \\
\text { s.t. } & y^{(i)}\left(w^{T} x^{(i)}+b\right) \geq \hat{\gamma}, \quad i=1, \ldots, m
\end{aligned}
$$
这样约束条件不再需要||w||=1。然后回想我们之前说到： we can add an arbitrary scaling constraint on w and b without changing anything. So we will introduce the scaling constraint that the functional margin of w, b with respect to the training set must be 1:  $\hat{\gamma}=1$. 也可以想象我们先优化出一个函数最大值，然后我们再rescale这个超平面，这不会对优化结果产生任何影响，但它可以令$\hat{\gamma}=1$。

于是我们的问题能转化成：
$$
\begin{aligned}
\max _{ w, b} & \frac{1}{\|w\|} \\
\text { s.t. } & y^{(i)}\left(w^{T} x^{(i)}+b\right) \geq 1, \quad i=1, \ldots, m
\end{aligned}
$$
我们再把max改成min，等价下式：
$$
\begin{aligned}
\min _{\gamma, w, b} & \frac{1}{2}\|w\|^{2} \\
\text { s.t. } & y^{(i)}\left(w^{T} x^{(i)}+b\right) \geq 1, \quad i=1, \ldots, m
\end{aligned}
$$
这个上式的解就是我们想要的optimal margin classifier. 我们可以用任何的commercial quadratic programming code去做。这也是所谓的==原始SVM==。  

下面补充证明SVM原始形式的一个重要性质：

#### 最大间隔超平面的存在唯一性

唯一性求解思路，证w相等，再证b相等。

![image-20200227115913670](media\Chapter6_支持向量机\image-20200227115913670.png)
$$
\begin{array}{l}
c \leq\|w\|=\frac{\left\|w_{1}^{*}+w_{2}^{*}\right\|}{2} \leq \frac{\left\|w_{1}^{*}\right\|+\left\|w_{2}^{*}\right\|}{2}=c \Rightarrow\|w\|=\frac{\left\|w_{1}^{*}+w_{2}^{*}\right\|}{2}=\frac{\left\|w_{1}^{*}\right\|+\left\|w_{2}^{*}\right\|}{2} \Rightarrow \underbrace{w_{1}^{*}=k w_{2}^{*}}_{\text {colinear }} \\
\Rightarrow k=\left\{\begin{array}{l}
1 \\
-1
\end{array} \Rightarrow w=\left\{\begin{array}{l}
w_{1}^{*}=w_{2}^{*} \text { if } k=1 \\
0 \text { if } k=-1
\end{array}  \quad \Rightarrow \quad w_{1}^{*}=w_{2}^{*}\right.\right.
\end{array}
$$
假设||w||是两者的平均，那就根据平行四边形定理可以证明w1和w2一定共线。如果它们是反向的话，||w||就为零，就比margin c小了。这样w就成了最优解，与假设相反。所以w是唯一的。

接着我们看b：

![image-20200827004504443](media\Chapter6_支持向量机\image-20200827004504443.png)
$$
\begin{aligned}
&\left\{\begin{array}{l}
b_{1}^{*}=-\frac{1}{2} \boldsymbol{w}^{T}\left(\boldsymbol{x}_{1}^{\prime}+\boldsymbol{x}_{1}^{\prime \prime}\right) \\
b_{2}^{*}=-\frac{1}{2} \boldsymbol{w}^{T}\left(\boldsymbol{x}_{2}^{\prime}+\boldsymbol{x}_{2}^{\prime \prime}\right)
\end{array} \Rightarrow b_{1}^{*}-b_{2}^{*}=-\frac{1}{2}\left[\boldsymbol{w}^{T}\left(\boldsymbol{x}_{1}^{\prime}-\boldsymbol{x}_{2}^{\prime}\right)+\boldsymbol{w}^{T}\left(\boldsymbol{x}_{1}^{\prime \prime}-\boldsymbol{x}_{2}^{\prime \prime}\right)\right]\right.\\
&\left\{\begin{array}{l}
w^{T} x_{2}^{\prime}+b_{1}^{*} \geq 1=w^{T} x_{1}^{\prime}+b_{1}^{*} \\
w^{T} x_{1}^{\prime}+b_{2}^{*} \geq 1=w^{T} x_{2}^{\prime}+b_{2}^{*} \\
w^{T} x_{2}^{\prime \prime}+b_{1}^{\prime} \geq 1=w^{T} x_{1}^{\prime \prime}+b_{1}^{*} \\
w^{T} x_{1}^{\prime \prime}+b_{2}^{\prime} \geq 1=w^{T} x_{2}^{\prime \prime}+b_{2}^{*}
\end{array} \Rightarrow\left\{\begin{array}{l}
w^{T}\left(x_{1}^{\prime}-x_{2}^{\prime}\right)=0 \\
w^{T}\left(x_{1}^{\prime \prime}-x_{2}^{\prime \prime}\right)=0
\end{array} \Rightarrow b_{1}^{*}=b_{2}^{*}\right.\right.
\end{aligned}
$$
为了证明b1，b2相等，我们首先假设x'~1~和x''~1~是第一个超平面的supporting vector。然后我们导出b1-b2的形式。接着第二行，我们证明b1-b2=0。第二行第一个→是因为我们把第一个式子移项可得$w^{T}( x_{2}^{\prime} - x_{1}^{\prime} ) \geq 0$，而第二个式子是$w^{T}( x_{1}^{\prime} - x_{2}^{\prime} ) \geq 0$，所以可得$w^{T}( x_{2}^{\prime} - x_{1}^{\prime} ) = 0$。同理可得第二列第二个式子，然后带入第一行的b1-b2的式子可得b1=b2。

#### AndrewNg课程里对优化目标的几何理解

我们接着再从AndrewNg在机器学习课上提供的角度来理解下SVM：

$$
\begin{array}{ll}
\min _{\theta} \frac{1}{2} \sum_{j=1}^{n} \theta_{j}^{2} \\
\text { s.t. } & \theta^{T} x^{(i)} \geq 1 & \text { if } y^{(i)}=1 \\
&\theta^{T} x^{(i)} \leq-1 & \text { if } y^{(i)}=0
\end{array}
$$
假设我们不考虑b值（$\theta_0=0$），且只限定在2维平面。那其实我们的优化目标就是在最小化参数向量的norm的平方。而它和x的乘积就是x在它方向上的投影。

![image-20200227123129212](media\Chapter6_支持向量机\image-20200227123129212.png)

那怎样才能最小化我们的超平面法向量，同时又能满足样本x到它的投影值要能大于等于1或小于等于-1呢？关键就是找到一个和两类样本连线垂直的平面：

![image-20200227123848232](media\Chapter6_支持向量机\image-20200227123848232.png)

从这样的角度理解，就能知道为什么看似奇怪的目标函数能帮我们找出能划分出最大间隔的超平面了。

## **6.2 对偶问题**

### 引子

为什么通过对偶问题进行求解，有下面两个原因：

* 一是因为使用对偶问题更容易求解；
* 二是因为通过对偶问题求解出现了向量内积的形式，从而能更加自然地引出核函数。

接下来要讲怎么==把原始SVM转化成它的对偶问题，然后用SMO（sequential-minimal-optimization）算法来高效求解，因为原始SVM的一大问题就是求解比较慢==。由于内容本身有难度，所以我先补充一个Victor Lavrenko教授的简易讲解版：

首先已知一个包含正负类样本的文本数据集，我们要找一个linear classifier - w’x把它们分开，然后通过这个classifier去分类未来收到的新样本。我们可以这样去找这个classifier：首先量出正负类样本的centroid。然后在它们两点间连线，这个向量我们记为w。再取一个垂直于w的平面，也就是红色标记的超平面。这个平面就是要找的classifier，它可以表示为w’x。

为什么这个平面能帮我们分类呢？想象现在我们收到有一个新的d样本，为了划分它为正还是为负，我们把他和c+，c-做点乘，再取差值就能看出它更接近谁。而c~+~ - c~-~其实就是向量w，所以这个过程其实就是w’x：

<img src="media\Chapter6_支持向量机\SVM0.JPG" alt="SVM0" style="zoom:80%;" />

类似于上面的思路，支持向量机也可以被写成两个向量相减的形式。只是在SVM里这两个向量不是centroids而是在正负两堆样本里的两个点。（我们最终要找的当然是support点，但它的域是convex hull里的任一点）

这两个点都可以被定义为所属类的样本集的corner向量d~i~的线性组合，所以所有的$\alpha$∈ [0,1]，而且他们的和要是1。然后我们要找到正负样本集里距离最近的两个点。这样连出来的线就是最小的w。它的中点对应的垂直平面也就是SVM classifier。

![SVM1](media\Chapter6_支持向量机\SVM1.JPG)

所以我们成功的把SVM的目标函数argmin||w||^2^变成了找距离最近点的形式，也就是它的对偶形式。然后我们就可以用SMO来求解：

SMO的核心是S，也就是顺序。因为有两个点，我们可以先选定其中一个点，然后把它的线性组合式里的weight拿两个出来调整（下图里拿的是$\alpha_1和\alpha_3$）。这两个由于必须满足sum（$\alpha_1,...,\alpha_v$）=1的这个约束条件，所以点随它们变化而变化的移动轨迹一定是一条直线。所以我们可以用一个变量$\alpha$来parameterize这一条线。然后哦我们会发现，我们就是在解一个非约束条件下的二阶凸函数单变量的最小值问题，这是个有解析解的问题（右下第二行）。然后我们再根据约束条件来看怎么truncate α，然后在rescale得到$\alpha_1和\alpha_3$的值（右下最下面）。下图右下角式子里的a和b分别是左图里的两个绿点（α=0，1）。它的

![SVM2](media\Chapter6_支持向量机\SVM2.JPG)

接着是datawhale的详细版推导。



### 对偶推导

#### 拉格朗日乘子 Lagrange multiplier

拉格朗日乘子法其核心思想就是**把有约束优化问题转变为等价的无约束优化问题(形式上)**。

>In [mathematical optimization](https://en.wikipedia.org/wiki/Mathematical_optimization), the **method of Lagrange multipliers** is a strategy for finding the local [maxima and minima](https://en.wikipedia.org/wiki/Maxima_and_minima) of a [function](https://en.wikipedia.org/wiki/Function_(mathematics)) subject to [equality constraints](https://en.wikipedia.org/wiki/Constraint_(mathematics)) (i.e., subject to the condition that one or more [equations](https://en.wikipedia.org/wiki/Equation) have to be satisfied exactly by the chosen values of the [variables](https://en.wikipedia.org/wiki/Variable_(mathematics))).[[1\]](https://en.wikipedia.org/wiki/Lagrange_multiplier#cite_note-1) It is named for the mathematician [Joseph-Louis Lagrange](https://en.wikipedia.org/wiki/Joseph-Louis_Lagrange). The basic idea is to convert a constrained problem into a form such that the [derivative test](https://en.wikipedia.org/wiki/Derivative_test) of an unconstrained problem can still be applied. The relationship between the gradient of the function and gradients of the constraints rather naturally leads to a reformulation of the original problem, known as the **Lagrangian function

要转换成拉格朗日函数，我们首先需要把原始问题写成标准形式：
$$
\begin{array}{cl}
\min _{x} & f_{0}(x) \\
\text { s.t. } & f_{i}(x) \leq 0, \quad i=1, \ldots, m \\
& h_{i}(x)=0, \quad i=1, \ldots, p
\end{array}
$$
如果原始问题是max f，那就在函数前加个负号，转换成min -f。如果不等式约束是大于等于，那就移项。

接着我们就能把标准形式的原始问题转换成拉格朗日乘子了：
$$
\begin{array}{c}
\mathcal{L}(x, \lambda, \nu)=f_{0}(x)+\sum_{i=1}^{m} \lambda_{i} f_{i}(x)+\sum_{i=1}^{p} \nu_{i} h_{i}(x) \\
\text { s.t. } \quad \lambda_{i} \geq 0, \quad i=1, \ldots, m
\end{array}
$$
原始问题就能写成：
$$
\begin{array}{c}
\min _{x} \max _{\lambda, v} \mathcal{L}(x, \lambda, \nu) \\
\text { s.t. } \quad \lambda_{i} \geq 0, \quad i=1, \ldots, m
\end{array}
$$


其实就是把原始的min f问题，转化成了求min-max L的问题。且如果原始问题满足一定条件，则这个问题的最优解又等价于 max-min L问题的。也就是min max = max min。==所以我们就可以通过高效求解max-min L来获得原问题的解。高效求解的方法通常是利用KKT条件。==

* 等价证明：

![image-20200827004746064](media\Chapter6_支持向量机\image-20200827004746064.png)

因为$\theta_{P}(x)=\max _{\lambda, v} \mathcal{L}(x, \lambda, \nu)$只有两种取值可能：当满足原问题约束就取f(x)，当不满足约束就取+∞。上图下半部分解释了为什么只有这两种可能。

#### 原始SVM的拉格朗日转换

原始形式（式6.6）
$$
\begin{array}{ll}
\min _{\boldsymbol{w}, b} & \frac{1}{2}\|\boldsymbol{w}\|^{2} \\
\text { s.t. } & \boldsymbol{y}_{i}\left(\boldsymbol{w}^{T} \boldsymbol{x}_{i}+b\right) \geq 1, \quad i=1,2, \ldots, m
\end{array}
$$
标准形式
$$
\begin{array}{ll}
\min _{\boldsymbol{w}, b} & \frac{1}{2}\|\boldsymbol{w}\|^{2} \\
\text { s.t. } & 1-\boldsymbol{y}_{i}\left(\boldsymbol{w}^{T} \boldsymbol{x}_{i}+b\right) \leq 0, \quad i=1,2, \ldots, m
\end{array}
$$
拉格朗日乘子（式6.8）
$$
\begin{aligned}
\mathcal{L}(\boldsymbol{w}, b, \alpha) &=\frac{1}{2}\|\boldsymbol{w}\|^{2}+\sum_{i=1}^{m} \alpha_{i}\left(1-\boldsymbol{y}_{i}\left(\boldsymbol{w}^{T} \boldsymbol{x}_{i}+b\right)\right) \\
\text { s.t. } & \alpha_{i} \geq 0 \text { for } i=1, \ldots, m
\end{aligned}
$$
等价问题
$$
\begin{aligned}
\min _{\boldsymbol{w}, b} \max _{\boldsymbol{\alpha}} & \mathcal{L}(\boldsymbol{w}, b, \boldsymbol{\alpha}) \\
\text { s.t. } & \alpha_{i} \geq 0 \text { for } i=1, \ldots, m
\end{aligned}
$$

#### 对偶形式

==对偶形式（min-max顺序变成了max-min，变成凸函数了）==
$$
\begin{array}{cl}
\max _{\alpha} \min _{\boldsymbol{w}, b} & \frac{1}{2}\|\boldsymbol{w}\|^{2}+\sum_{i=1}^{m} \alpha_{i}\left(1-\boldsymbol{y}_{i}\left(\boldsymbol{w}^{T} \boldsymbol{x}_{i}+b\right)\right) \\
\text { s.t. } & \alpha_{i} \geq 0 \quad \text { for } i=1, \ldots, m
\end{array}		\tag1
$$
对w和b求导令为零（式6.9和6.10）
$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial w}=0 &=\frac{\partial}{\partial w}\left(\frac{1}{2} w^{T} w+\sum_{i=1}^{m} \alpha_{i}-\sum_{i=1}^{m} \alpha_{i} y_{i}\left(w^{T} x_{i}+b\right)\right) \\
&=\frac{\partial}{\partial w}\left(\frac{1}{2} w^{T} w+\sum_{i=1}^{m} \alpha_{i}-\sum_{i=1}^{m} a_{i} y_{i} w^{T} x_{i}-\sum_{i=1}^{m} \alpha_{i} y_{i} x_{i} b\right) \\
&=\frac{\partial}{\partial w}\left(\frac{1}{2} w^{T} w-\sum_{i=1}^{m} \alpha_{i} y_{i} w^{T} x_{i}\right) \\
&=\frac{\partial}{\partial w}\left(\frac{1}{2} w^{T} w-w^{T} \sum_{i=1}^{m} \alpha_{i} y_{i} x_{i}\right) \\
&=w-\sum_{i=1}^{m} \alpha_{i} y_{i} x_{i}
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial b}=0 &=\frac{\partial}{\partial b}\left(\frac{1}{2} w^{T} w+\sum_{i=1}^{m} \alpha_{i}-\sum_{i=1}^{m} \alpha_{i} y_{i}\left(\boldsymbol{w}^{T} \boldsymbol{x}_{i}+b\right)\right) \\
&=\frac{\partial}{\partial b}\left(\frac{1}{2} \boldsymbol{w}^{T} \boldsymbol{w}+\sum_{i=1}^{m} \alpha_{i}-\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i} \boldsymbol{w}^{T} \boldsymbol{x}_{i}-\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i} b\right) \\
&=\frac{\partial}{\partial b}\left(-\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i} b\right) \\
&=\frac{\partial}{\partial b}\left(-b \sum_{i=1}^{m} \alpha_{i} y_{i}\right) \\
&=-\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i}
\end{aligned}
$$

再把这两个结论带回到对偶公式(1)
$$
\begin{aligned}
\min _{\boldsymbol{w}, b} \mathcal{L}(\boldsymbol{w}, b, \alpha) &=\frac{1}{2}\|\boldsymbol{w}\|^{2}+\sum_{i=1}^{m} \alpha_{i}\left(1-\boldsymbol{y}_{i}\left(\boldsymbol{w}^{T} \boldsymbol{x}_{i}+b\right)\right) \\
&=\frac{1}{2} \boldsymbol{w}^{T} \boldsymbol{w}-\boldsymbol{w}^{T}\left(\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i} \boldsymbol{x}_{i}\right)-b\left(\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i}\right)+\left(\sum_{i=1}^{m} \alpha_{i}\right) \\
&=\frac{1}{2} \boldsymbol{w}^{T}\left(\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i} \boldsymbol{x}_{i}\right)-\boldsymbol{w}^{T}\left(\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i} \boldsymbol{x}_{i}\right)-b \cdot 0+\left(\sum_{i=1}^{m} \alpha_{i}\right) \\
&=\left(\sum_{i=1}^{m} \alpha_{i}\right)+\left(\frac{1}{2} \boldsymbol{w}^{T}-\boldsymbol{w}^{T}\right)\left(\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i} \boldsymbol{x}_{i}\right) \\
&=\left(\sum_{i=1}^{m} \alpha_{i}\right)-\frac{1}{2}\left(\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i} \boldsymbol{x}_{i}\right)^{T}\left(\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i} \boldsymbol{x}_{i}\right) \\
&=\left(\sum_{i=1}^{m} \alpha_{i}\right)-\frac{1}{2}\left(\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i} \boldsymbol{x}_{i}^{T}\right)\left(\sum_{j=1}^{m} \alpha_{j} \boldsymbol{y}_{j} \boldsymbol{x}_{j}\right) \\
&=\left(\sum_{i=1}^{m} \alpha_{i}\right)-\frac{1}{2}\left(\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i} \boldsymbol{x}_{i}^{T}\left(\sum_{j=1}^{m} \alpha_{j} \boldsymbol{y}_{j} \boldsymbol{x}_{j}\right)\right) \\
&=\sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} \boldsymbol{y}_{i} \boldsymbol{y}_{i}^{T} \boldsymbol{x}_{j}
\end{aligned}
$$
这就是式6.11
$$
\begin{aligned}
\max _{\alpha} & \sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j} x_{i}^{T} x_{j} \\
\text { s.t. } & \sum_{i=1}^{m} \alpha_{i} y_{i}=0 \\
& \alpha_{i} \geq 0, \quad i=1, \ldots, m
\end{aligned}
$$
现在假设我们已经求出α值使得目标函数最大。然后我们可以通过下式获得w和b
$$
\begin{aligned}
\boldsymbol{w}^{*} &=\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i} \boldsymbol{x}_{i} \\
b^{*} &=\boldsymbol{y}_{j}-\sum_{i=1}^{m} \alpha_{i} y_{i}\left(\boldsymbol{x}_{i}^{T} \boldsymbol{x}_{j}\right)
\end{aligned}
$$
理论上b*只有一个值。故，在求得α之后，只需要任选一个支持向量（α!=0）就能获得b\*的值，因为支持向量满足
$$
y_{j}\left(\boldsymbol{w}^{* T} \boldsymbol{x}_{j}+b^{*}\right)-1=0 \\ \Rightarrow  y_{j} y_{j}\left(\boldsymbol{w}^{* T} \boldsymbol{x}_{j}+b^{*}\right)-y_{j}=0\\
\Rightarrow b^{*}=y_{j}-w^{* T} x_{j}=y_{j}-\sum_{i=1}^{m} \alpha_{i} y_{i}\left(x_{i}^{T} x_{j}\right)
$$

#### KKT条件

KKT条件就是一组方程，满足此组方程的解等价于一部分最优化问题的解。下图是标准形式问题的KKT：

![image-20200827004832164](media\Chapter6_支持向量机\image-20200827004832164.png)

**使用方法** - 在情况1里，如果x*是解则满足KKT，但我们不能反推如果x满足KKT就是最优解（必要性）。情况2做了更多限定，我们也因此获得了充要条件，这对应的问题就是强对偶问题。

![image-20200827004903501](media\Chapter6_支持向量机\image-20200827004903501.png)

上图的slater条件不对，应该是有一个约束域内点能让约束不等号严格成立。

![image-20200827005055285](media\Chapter6_支持向量机\image-20200827005055285.png)

我们来看看我们的原始SVM问题是否能用KKT条件，抄写式6.6
$$
\begin{array}{ll}
\min _{\boldsymbol{w}, b} & \frac{1}{2}\|\boldsymbol{w}\|^{2} \\
\text { s.t. } & \boldsymbol{y}_{i}\left(\boldsymbol{w}^{T} \boldsymbol{x}_{i}+b\right) \geq 1, \quad i=1,2, \ldots, m
\end{array}
$$
目标函数和不等式函数都是凸的，不存在等式约束，且不等式约束满足slater condition也满足。

证明slater条件：我们可以假设已经找出了一个超平面参数w，b，能满足约束。然后我们再把左边乘上一个大于零的数，这样就能让所有$\boldsymbol{y}_{i}\left(\boldsymbol{w}^{T} \boldsymbol{x}_{i}+b\right)$都大于1，也就是$1 - \boldsymbol{y}_{i}\left(\boldsymbol{w}^{T} \boldsymbol{x}_{i}+b\right) < 0, \quad i=1,2, \ldots, m$。由于对w，b没约束，所以他们也一定是内点。

==所以，我们的原始问题的最优解就等价于满足KKT条件的解==

#### KKT条件的理解

![image-20200827005208447](media\Chapter6_支持向量机\image-20200827005208447.png)

对于不等式约束优化问题，最优解就只有两种情况：解在或不在边界上。

1. 在边界上，这意味着我们从不等式回到了等式约束，f和g的gradient应该共线反向（注意上图的g其实是wiki里的f~i~或者本文中的h~i~）。因为是最小化问题，所以$\nabla f$是指向圈内的，而f是想向圈外走的。但是g的约束是<=0，所以$\nabla g$应该是指向圈外的，表示我们不能再往外面走了。于是我们得到下式

$$
\left\{\begin{array}{r}
\nabla_{x} f\left(x^{*}\right)+\lambda_{i} \nabla_{\boldsymbol{x}} h\left(x^{*}\right)=0 \\
h_{k}(x)=0 \\
\lambda_{k}>0
\end{array}\right.
$$

2. 不在边界上，这意味着我们能取到原函数的极值。我们可以用右边的形式来等价表示

$$
\nabla_{x} f\left(x^{*}\right)=0 \quad \Rightarrow \quad\left\{\begin{array}{r}
\nabla_{x f}\left(x^{*}\right)+\lambda_{i} \nabla_{x} h\left(x^{*}\right)=0 \\
h_{k}(x)<0 \\
\lambda_{k}=0
\end{array}\right.
$$

==注意，我们上面的推导用的是1个不等式约束的例子，因为这样更直观，但结论对多个约束情况也是适用的==。最后，当我们把这两种情况加上等式约束的情况组合在一起就形成了KKT条件。

![image-20200827005345226](media\Chapter6_支持向量机\image-20200827005345226.png)

具体来说，根据SVM的KKT条件可以得到

![image-20200827005444566](media\Chapter6_支持向量机\image-20200827005444566.png)

ok，我们来回顾一下目前为止的求解流程：

1. 把原始问题转换成标准形式
2. 写出拉格朗日函数
3. 把原始问题改写成min-max形式
4. 获得对偶问题 max-min
5. 根据KKT条件的stationary equation求min部分，简化得到最终SVM对偶表达式
6. 再根据剩下的KKT条件求解问题（SMO算法）

### SMO求解

SMO的全称是Sequential minimal optimization。它的特点是：

1. 是一个迭代求解。每次迭代仅优化总变量里的两个参数，且有闭式解。
2. 启发式的寻找每次优化的参数，有效减少迭代次数。

思路：

1. 设置$\alpha$列表，并设其处置为0。（每个数据点对应一个$\alpha_i$）
2. 选取两个待优化变量（记为$\alpha_1$, $\alpha_2$）
3. 求出两个变量的最优解（闭式解），并更新至$\alpha$列表。
4. 检查更新后的$\alpha$列表是否在某个精度范围内满足KKT条件。若不满足，返回step2。

这里要指出的是$\alpha_1$, $\alpha_2$不是某个特定顺序的样本点，每次都可能更新为别的点，这样标记只是为了方便。因为是计算机就会有精度误差，所以第四点需要一个误差tolerance。

我们把对偶SVM复制下来
$$
\begin{aligned}
\max _{\alpha} & \sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} \boldsymbol{y}_{i} \boldsymbol{y}_{j} \boldsymbol{x}_{i}^{T} \boldsymbol{x}_{j} \\
\text { s.t. } & \sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i}=0 \\
& \alpha_{i} \geq 0, \boldsymbol{i}=1, \ldots, m
\end{aligned}
$$
然后把两个参数项拧出来，得到下式第一个等号后面的内容，这里第2，3项是i=j={1,2}的情况，第4项是ij=12+21的情况，后面5，6则分别是以α1和α2对应α3,4,5...考虑的情况。由于这是个对α12的优化问题，所以可以把sum(α345...)这项约掉在第二个等号里。
$$
\begin{aligned}
\max _{a_{1}, a_{2}} W\left(\alpha_{1}, \alpha_{2}\right) &=\left(\alpha_{1}+\alpha_{2}+\sum_{i=3}^{m} \alpha_{i}\right)-\frac{1}{2} K_{11} \alpha_{1}^{2}-\frac{1}{2} K_{22} \alpha_{2}^{2}-y_{1} y_{2} K_{12} \alpha_{1} \alpha_{2}-y_{1} \alpha_{1} \sum_{i=3}^{m} y_{i} \alpha_{i} K_{i 1}-y_{2} \alpha_{2} \sum_{i=3}^{m} y_{i} \alpha_{i} K_{i 2} \\
&=\left(\alpha_{1}+\alpha_{2}\right)-\frac{1}{2} K_{11} \alpha_{1}^{2}-\frac{1}{2} K_{22} \alpha_{2}^{2}-y_{1} y_{2} K_{12} \alpha_{1} \alpha_{2}-y_{1} \alpha_{1} \sum_{i=3}^{m} y_{i} \alpha_{i} K_{i 1}-y_{2} \alpha_{2} \sum_{i=3}^{m} y_{i} \alpha_{i} K_{i 2}
\end{aligned}\\

\begin{aligned}
\text { s.t. } \quad \alpha_{1} y_{1}+\alpha_{2} y_{2} &=-\sum_{i=3}^{m} y_{i} \alpha_{i}=\xi \\
\alpha_{i} & \geq 0, \quad i=1,2	\quad其中，K_{ij}=x^T_i x_j
\end{aligned}
$$
观察上式可知，这是一个二元二次问题。留意到约束的存在，我们可以将问题再次转换成带约束的一元二次问题。对于一元二次问题，我们可以通过比较最终指点和可行域之间关系，获得在可行域中的最值。

令 $v_{i}=\sum_{j=3}^{N} \alpha_{j} y_{j} K_{i j}=f\left(x_{i}\right)-\sum_{j=1}^{2} \alpha_{j} y_{j} K_{i j}-b, \quad i=1,2$。然后给原W乘一个负号把maxW转化成min(-W)，并记-W为新的W。则有

$$
\begin{aligned}
W\left(\alpha_{1}, \alpha_{2}\right)=& \frac{1}{2} K_{11} \alpha_{1}^{2}+\frac{1}{2} K_{22} \alpha_{2}^{2}+y_{1} y_{2} K_{12} \alpha_{1} \alpha_{2} \\
&-\left(\alpha_{1}+\alpha_{2}\right)+y_{1} v_{1} \alpha_{1}+y_{2} v_{2} \alpha_{2}
\end{aligned}
$$
回代 $\alpha_{1} y_{1}+\alpha_{2} y_{2}=-\sum_{i=3}^{m} y_{i} \alpha_{i}=\xi \Leftrightarrow \alpha_{1}=\left(\xi-y_{2} \alpha_{2}\right) y_{1}$（要得最后一个等式首先两边同乘y1），得一元二次式
$$
\begin{array}{c}
W\left(\alpha_{2}\right)=\frac{1}{2} K_{11}\left(\xi-\alpha_{2} y_{2}\right)^{2}+\frac{1}{2} K_{22} \alpha_{2}^{2}+y_{2} K_{12}\left(\xi-\alpha_{2} y_{2}\right) \alpha_{2} \\
-\left(\xi-\alpha_{2} y_{2}\right) y_{1}-\alpha_{2}+v_{1}\left(\xi-\alpha_{2} y_{2}\right)+y_{2} v_{2} \alpha_{2}
\end{array}
$$
接着对α2求偏导令其=0
$$
\frac{\partial W}{\partial \alpha_{2}}=K_{11} \alpha_{2}+K_{22} \alpha_{2}-2 K_{12} \alpha_{2}-K_{11} \xi y_{2}+K_{12} \xi y_{2}+y_{1} y_{2}-1-v_{1} y_{2}+y_{2} v_{2}=0
$$
整理可得（其中1=y2y2）
$$
\begin{aligned}
\left(K_{11}+K_{22}-2 K_{12}\right) \alpha_{2}=& y_{2}\left(y_{2}-y_{1}+\xi K_{11}-\xi K_{12}+v_{1}-v_{2}\right) \\
=& y_{2}\left[y_{2}-y_{1}+\xi K_{11}-\xi K_{12}+\left(f\left(x_{1}\right)-\sum_{j=1}^{2} y_{j} \alpha_{j} K_{1 j}-b\right)-\left(f\left(x_{2}\right)-\sum_{j=1}^{2} y_{j} \alpha_{j} K_{2 j}-b\right)\right]
\end{aligned}
$$
将$\xi=\alpha_{1}^{o l d} y_{1}+\alpha_{2}^{o l d} y_{2}$代入（这里old上标是为了和更新后的new α做区分），并设$E_{i}=f\left(x_{i}\right)-y_{i}$，可得
$$
\begin{aligned}
\left(K_{11}+K_{22}-2 K_{12}\right) \alpha_{2}^{\text {new},\text {unc}} &=y_2[y_2-y_1+\alpha_1y_1K_{11}+\alpha_2y_2K_{11}-\alpha_1y_1K_{12}-\alpha_2y_2K_{12}+(f(x_1)-f(x_2))-\alpha_1y_1K_{11}-\alpha_2y_2K_{12}+\alpha_1y_1K_{21}+\alpha_2y_2K_{22}]
\\&=y_{2}\left(\left(K_{11}+K_{22}-2 K_{12}\right) \alpha_{2}^{\text {old }} y_{2}+y_{2}-y_{1}+f\left(x_{1}\right)-f\left(x_{2}\right)\right) \\
&=\left(K_{11}+K_{22}-2 K_{12}\right) \alpha_{2}^{\text {old }}+y_{2}\left(E_{1}-E_{2}\right)
\end{aligned}
$$
α2的上标new是说这是更新值，unc是说这是没有根据可行域clip的。注意，上面$v_{i}=\sum_{j=3}^{N} \alpha_{j} y_{j} K_{i j}=f\left(x_{i}\right)-\sum_{j=1}^{2} \alpha_{j} y_{j} K_{i j}-b$ 的第二个等号是由$w =\sum_{j=1}^{N} \alpha_{j} y_{j} \boldsymbol{x}_{j},\quad f(x_i)=wx_i+b$推导得来的。

设$\eta=K_{11}+K_{22}-2 K_{12}$可得
$$
\alpha_{2}^{\text {new},\text {unc}}=\alpha_{2}^{\text {old }}+\frac{y_{2}\left(E_{1}-E_{2}\right)}{\eta}
$$
然后由于我们对α有定义域的约束，所以它的最优解还需要通过边界条件去判定再做clip。现在假设我们已经获得α2最优解，α1即可根据$y_{1} \alpha_{1}^{\text {new }}+y_{2} \alpha_{2}^{\text {new }}=y_{1} \alpha_{1}^{\text {old }}+y_{2} \alpha_{2}^{\text {old }}$求得
$$
\alpha_{1}^{\text {new }}=\alpha_{1}^{\text {old }}+y_{1} y_{2}\left(\alpha_{2}^{\text {old }}-\alpha_{2}^{\text {new }}\right)
$$

#### α2的可行域判定

α2的可行域必须满足下式条件
$$
\begin{array}{l}
L \leqslant \alpha_{2}^{\text {new }} \leqslant H \\
\alpha_{1} y_{1}+\alpha_{2} y_{2}=\xi
\end{array}
$$
<img src="media\Chapter6_支持向量机\image-20200827113051060.png" alt="image-20200827113051060" style="zoom: 50%;" />

如果y1和y2同号（右图），则α2的最小值只能取max()。因为这种情况下可能α2取不到零，如果α1-α2=k这条线在第一象限的话。
$$
\begin{array}{l}
L=\max \left(0, \alpha_{2}^{\text {old }}-\alpha_{1}^{\text {old }}\right) \\
H=+\infty
\end{array}
$$
异号的话
$$
\begin{array}{l}
L=0 \\
H=+\infty
\end{array}
$$
则可得
$$
\alpha_{2}^{n e w}=\left\{\begin{array}{ll}
H & \alpha_{2}^{n e w, u n c}>H \\
\alpha_{2}^{n e w, u n c} & L \leq \alpha_{2}^{n e w, u n c} \leq H \\
L & \alpha_{2}^{n e w, u n c} \leq L
\end{array}\right.
$$

#### 怎样启发式选取两个变量

![image-20200827010905937](media\Chapter6_支持向量机\image-20200827010905937.png)

==关于细节代码的实现，可以看附录==

### 整体流程

我们再来回忆一遍整体的求解过程

1. 首先求L对w和b的极小，分别求L关于w和b的偏导，可以得出：

![14.png](media\Chapter6_支持向量机\5bc72f9333e66.png)

将上述结果代入L得到：

![15.png](media\Chapter6_支持向量机\5bc72f935ae21.png)

2. 接着L关于α极大求解α（通过SMO算法求解，此处不做深入）。

![16.png](media\Chapter6_支持向量机\5bc72f9338a9d.png)

3. 最后便可以根据求解出的α，计算出w和b，从而得到分类超平面函数。

![17.png](media\Chapter6_支持向量机\5bc72f93419ca.png)

4. 在对新的点进行预测时，将数据点x*代入分类函数f(x)=w'x+b中，若f(x)>0，则为正类，f(x)<0，则为负类，根据前面推导得出的w与b，分类函数如下所示，此时便出现了上面所提到的内积形式。

![18.png](media\Chapter6_支持向量机\5bc72f9353166.png)

这里实际上只需计算新样本与支持向量的内积，因为对于非支持向量的数据点，其对应的拉格朗日乘子一定为0。因为根据最优化理论（KKT条件）对于不等式约束y(w'x+b)-1≥0，满足：

![19.png](media\Chapter6_支持向量机\5bc72f933c947.png)



## **6.3 核函数**

由于上述的超平面只能解决线性可分的问题，对于线性不可分的问题，例如：异或问题，我们需要使用核函数将其进行推广。一般地，解决线性不可分问题时，常常采用**映射**的方式，将低维原始空间映射到高维特征空间，使得数据集在高维空间中变得线性可分，从而再使用线性学习器分类。如果原始空间为有限维，即属性数有限，那么总是存在一个高维特征空间使得样本线性可分。若∅代表一个映射，则在特征空间中的划分函数变为：

![20.png](media\Chapter6_支持向量机\5bc72f934303e.png)

按照同样的方法，先写出新目标函数的拉格朗日函数，接着写出其对偶问题，求L关于w和b的极大，最后运用SOM求解α。可以得出：

（1）原对偶问题变为：

![21.png](media\Chapter6_支持向量机\5bc730cc68b3b.png)

（2）原分类函数变为：​  

![22.png](media\Chapter6_支持向量机\5bc730cc1b673.png)

求解的过程中，只涉及到了高维特征空间中的内积运算，由于特征空间的维数可能会非常大，例如：若原始空间为二维，映射后的特征空间为5维，若原始空间为三维，映射后的特征空间将是19维，之后甚至可能出现无穷维，根本无法进行内积运算了，此时便引出了**核函数**（Kernel）的概念。

![23.png](media\Chapter6_支持向量机\5bc730cc49adc.png)

因此，核函数可以直接计算隐式映射到高维特征空间后的向量内积，而不需要显式地写出映射后的结果，它虽然完成了将特征从低维到高维的转换，但最终却是在低维空间中完成向量内积计算，与高维特征空间中的计算等效**（低维计算，高维表现）**，从而避免了直接在高维空间无法计算的问题。引入核函数后，原来的对偶问题与分类函数则变为：

（1）对偶问题：

![24.png](media\Chapter6_支持向量机\5bc730cc173b2.png)

（2）分类函数：

![25.png](media\Chapter6_支持向量机\5bc730cc05959.png)

因此，在线性不可分问题中，核函数的选择成了支持向量机的最大变数，若选择了不合适的核函数，则意味着将样本映射到了一个不合适的特征空间，则极可能导致性能不佳。同时，核函数需要满足以下这个必要条件：

![26.png](media\Chapter6_支持向量机\5bc730ccc468c.png)

由于核函数的构造十分困难，通常我们都是从一些常用的核函数中选择，下面列出了几种常用的核函数：

![27.png](media\Chapter6_支持向量机\5bc730ccc541a.png)

## **6.4 软间隔SVM**

前面的讨论中，我们主要解决了两个问题：当数据线性可分时，直接使用最大间隔的超平面划分；当数据线性不可分时，则通过核函数将数据映射到高维特征空间，使之线性可分。然而在现实问题中，对于某些情形还是很难处理，例如数据中有**噪声**的情形，噪声数据（**outlier**）本身就偏离了正常位置，但是在前面的SVM模型中，我们要求所有的样本数据都必须满足约束，如果不要这些噪声数据还好，当加入这些outlier后导致划分超平面被挤歪了，如下图所示，对支持向量机的泛化性能造成很大的影响。

![28.png](media\Chapter6_支持向量机\5bc730ccce68e.png)

为了解决这一问题，我们需要允许某一些数据点不满足约束，即可以在一定程度上偏移超平面，同时使得不满足约束的数据点尽可能少，这便引出了**“软间隔”支持向量机**的概念。

hard margin 式6.6
$$
\begin{array}{ll}
\min _{\boldsymbol{w}, b} & \frac{1}{2}\|\boldsymbol{w}\|^{2} \\
\text { s.t. } & \boldsymbol{y}_{i}\left(\boldsymbol{w}^{T} \boldsymbol{x}_{i}+b\right) \geq 1, \quad i=1,2, \ldots, m
\end{array}
$$
naive-soft-margin 式6.29，6.30

$$
\begin{array}{l}
\min _{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \ell_{0 / 1}\left(y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)-1\right) \\
\ell_{0 / 1}(z)=\left\{\begin{array}{l}
1, \text { if } z<0 \\
0, \text { otherwise }
\end{array}\right.
\end{array}
$$
如同阶跃函数，0/1损失函数虽然表示效果最好，但是数学性质不佳，不容易被微分。因此常用其它函数作为“替代损失函数surrogate loss function”，他们都是凸的连续函数，且是0/1损失函数的上界。

<img src="media\Chapter6_支持向量机\5bc730cc5e5a9.png" alt="30.png" style="zoom: 80%;" />

![image-20200829191800395](media\Chapter6_支持向量机\image-20200829191800395.png)

soft-margin 式**6.35**。支持向量机中的损失函数为**hinge损失**，引入**“松弛变量”**，目标函数与约束条件可以改写为 

![31.png](media\Chapter6_支持向量机\5bc7317aa3411.png)

其中C为一个参数，控制着目标函数与新引入正则项之间的权重。上式等价于hinge loss 6.34
$$
\min _{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \underbrace{\max \left(0,1-y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)\right)}_{\xi_{i}}
$$
==证明两者等价性==：因为式6.35里的$\xi_i$一定是大于等于0的，同时把6.35的第一个不等式移项也能得到$\xi_{i} \geq 1-y_{i}\left(w^{T} x_{i}+b\right)$。又由于6.35目标函数minimize的操作，所以$\xi_i$能对这两者中大的一个取到等号。
$$
\xi_{i}=\max \left(0,1-y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)\right)\left\{\begin{array}{l}
\geq 0 \\
\geq 1-\boldsymbol{y}_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)
\end{array}\right.
$$
==hinge loss其实是L1regulation正则，所以有很好的参数稀疏性==。

这样显然每个样本数据都有一个对应的松弛变量，用以表示该样本不满足约束的程度，将新的目标函数转化为拉格朗日函数得到 6.36
$$
\mathcal{L}(w, b, \xi, \alpha, r)=\frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{n} \xi_{i}-\sum_{i=1}^{n} \alpha_{i}\left(y_{i}\left(w^{T} x_{i}+b\right)-1+\xi_{i}\right)-\sum_{i=1}^{n} r_{i} \xi_{i}
$$
按照与之前相同的方法，先让L求关于w，b以及松弛变量的极小，再使用SMO求出α

$$
\begin{array}{l}
\frac{\partial L}{\partial w}=0 \Rightarrow w=\sum_{i=1}^{n} \alpha_{i} y_{i} x_{i} \\
\frac{\partial L}{\partial b}=0 \Rightarrow \sum_{i=1}^{n} \alpha_{i} y_{i}=0 \\
\frac{\partial L}{\partial \xi_{i}}=0 \Rightarrow C-\alpha_{i}-r_{i}=0, \quad i=1, \ldots, m
\end{array}
$$
偏导求解
$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial w}=0 &=\frac{\partial}{\partial \boldsymbol{w}}\left(\frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \xi_{i}+\sum_{i=1}^{m} \alpha_{i}\left(1-\xi_{i}-y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)\right)-\sum_{i=1}^{m} \mu_{i} \xi_{i}\right) \\
&=\frac{\partial}{\partial \boldsymbol{w}}\left(\frac{1}{2} \boldsymbol{w}^{T} \boldsymbol{w}-\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i} \boldsymbol{w}^{T} \boldsymbol{x}_{i}\right) \\
&=\frac{\partial}{\partial \boldsymbol{w}}\left(\frac{1}{2} \boldsymbol{w}^{T} \boldsymbol{w}-\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i} \boldsymbol{w}^{T} \boldsymbol{x}_{i}\right) \\
&=\frac{\partial}{\partial \boldsymbol{w}}\left(\frac{1}{2} \boldsymbol{w}^{T} \boldsymbol{w}-\boldsymbol{w}^{T} \sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i} \boldsymbol{x}_{i}\right) \\
&=\boldsymbol{w}-\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i} \boldsymbol{x}_{i}
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial b}=0 &=\frac{\partial}{\partial b}\left(\frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{m} \xi_{i}+\sum_{i=1}^{m} \alpha_{i}\left(1-\xi_{i}-y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)\right)-\sum_{i=1}^{m} \mu_{i} \xi_{i}\right) \\
&=\frac{\partial}{\partial b}\left(-\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i} b\right) \\
&=\frac{\partial}{\partial \boldsymbol{b}}\left(-b \sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i}\right) \\
&=-\sum_{i=1}^{m} \alpha_{i} y_{i}
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \xi_{i}}=0 &=\frac{\partial}{\partial \xi_{i}}\left(\frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \xi_{i}+\sum_{i=1}^{m} \alpha_{i}\left(1-\xi_{i}-y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)\right)-\sum_{i=1}^{m} \mu_{i} \xi_{i}\right) \\
&=\frac{\partial}{\partial \xi_{i}}\left(C \sum_{i=1}^{m} \xi_{i}+\sum_{i=1}^{m} \alpha_{i}\left(-\xi_{i}\right)-\sum_{i=1}^{m} \mu_{i} \xi_{i}\right) \\
&=\frac{\partial}{\partial \xi_{i}}\left(C \xi_{i}-\alpha_{i} \xi_{i}-\mu_{i} \xi_{i}\right) \\
&=C-\alpha_{i}-\mu_{i}
\end{aligned}
$$

将w代入L化简
$$
\begin{aligned}
\min _{\boldsymbol{w}, b, \xi} \mathcal{L}(\boldsymbol{w}, b, \xi, \alpha) &=\left(\frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \xi_{i}+\sum_{i=1}^{m} \alpha_{i}\left(1-\xi_{i}-y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)\right)-\sum_{i=1}^{m} \mu_{i} \xi_{i}\right) \\
&=\frac{1}{2} \boldsymbol{w}^{T} \boldsymbol{w}-\boldsymbol{w}^{T}\left(\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i} \boldsymbol{x}_{i}\right)+\left(\sum_{i=1}^{m} \alpha_{i}\right)-b\left(\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i}\right)+C \sum_{i=1}^{m} \xi_{i}-\sum_{i=1}^{m} \alpha_{i} \xi_{i}-\sum_{i=1}^{m} \mu_{i} \xi_{i} \\
&=\frac{1}{2} \boldsymbol{w}^{T}\left(\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i} \boldsymbol{x}_{i}\right)-\boldsymbol{w}^{T}\left(\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i} \boldsymbol{x}_{i}\right)+\left(\sum_{i=1}^{m} \alpha_{i}\right)-b \cdot 0+ \sum_{i=1}^{m} \left(C-\alpha_{i}-\mu_{i}\right)  \xi_{i} \\
&=\left(\sum_{i=1}^{m} \alpha_{i}\right)+\left(\frac{1}{2} \boldsymbol{w}^{T}-\boldsymbol{w}^{T}\right)\left(\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i} \boldsymbol{x}_{i}\right) \\
&=\left(\sum_{i=1}^{m} \alpha_{i}\right)-\frac{1}{2}\left(\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i} \boldsymbol{x}_{i}\right)^{T}\left(\sum_{i=1}^{m} \alpha_{i} y_{i} \boldsymbol{x}_{i}\right)\\
&=\left(\sum_{i=1}^{m} \alpha_{i}\right)-\frac{1}{2}\left(\sum_{i=1}^{m} \alpha_{i} y_{i} x_{i}^{T}\right)\left(\sum_{j=1}^{m} \alpha_{j} y_{j} x_{j}\right) \\
&=\left(\sum_{i=1}^{m} \alpha_{i}\right)-\frac{1}{2}\left(\sum_{i=1}^{m} \alpha_{i} y_{i} x_{i}^{T}\left(\sum_{j=1}^{m} \alpha_{j} y_{j} x_{j}\right)\right) \\
&=\sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j} x_{i}^{T} x_{j} \\
&=\mathcal{L}(\alpha)
\end{aligned}
$$
约束条件
$$
\left\{\begin{array}{l}
\alpha_{i} \geq 0 \\
\mu_{i} \geq 0 \\
\sum_{i=1}^{m} \alpha_{i} \boldsymbol{y}_{i}=0
\end{array} \Rightarrow\left\{\begin{array}{l}
\alpha_{i} \geq 0 \\
C-\alpha_{i} \geq 0 \\
\sum_{i=1}^{m} \alpha_{i} y_{i}=0
\end{array} \Rightarrow\left\{\begin{array}{l}
0 \leq \alpha_{i} \leq C \\
\sum_{i=1}^{m} \alpha_{i} y_{i}=0
\end{array}\right.\right.\right.
$$
下面是一个软SVM对偶推导的总览

![image-20200829175626594](media\Chapter6_支持向量机\image-20200829175626594.png)

对偶软间隔SVM的KKT条件
$$
\left\{\begin{array}{l}
\alpha_{i} \geqslant 0, \quad \mu_{i} \geqslant 0 \\
y_{i} f\left(\boldsymbol{x}_{i}\right)-1+\xi_{i} \geqslant 0， \xi_{i} \geqslant 0\\
\mu_{i} \xi_{i}=0， \alpha_{i}\left(y_{i} f\left(\boldsymbol{x}_{i}\right)-1+\xi_{i}\right)=0 \\
\end{array}\right.
$$
头一行是dual feasibility，第二行是primal feasibility，第三行是complimentary slackness。下面第一点，因为α=0的项没有被用来计算w，所以对分类平面没有贡献。第二点，由于$\mu$等于零，所以$\xi$不等于零，所以该点是要么落在margin里，要么被误分类了。第三行就是边界上的support点。

![image-20200829181728380](media\Chapter6_支持向量机\image-20200829181728380.png)

将“软间隔”下产生的对偶问题与原对偶问题对比可以发现：新的对偶问题只是约束条件中的α多出了一个上限C，其它的完全相同，因此在引入核函数处理线性不可分问题时，便能使用与“硬间隔”支持向量机完全相同的方法。

## 6.5 SVR回归

![image-20200829191657422](media\Chapter6_支持向量机\image-20200829191657422.png)

![image-20200829193918701](media\Chapter6_支持向量机\image-20200829193918701.png)

SVR的一般形式，式6.43和6.44，很直观的意思就是只有和y取值大到超过了epsilon才会有损失。
$$
\begin{array}{l}
\min _{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \ell_{c}\left(f\left(\boldsymbol{x}_{i}\right)-y_{i}\right) \\
\ell_{\epsilon}(z)=\left\{\begin{array}{l}
0, \quad \text { if }|z| \leqslant \epsilon \\
|z|-\epsilon, \text { otherwise }
\end{array}\right.
\end{array}
$$
经过修改，得到上式的等价形式6.45，观察发现上下式明显等价。
$$
\begin{aligned}
\min _{\boldsymbol{w}, b, \xi_{i}, \hat{\xi}_{i}} & \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m}\left(\xi_{i}+\hat{\xi}_{i}\right) \\
\text { s.t. } & f\left(\boldsymbol{x}_{i}\right)-y_{i} \leqslant \epsilon+\xi_{i} \\
& y_{i}-f\left(\boldsymbol{x}_{i}\right) \leqslant \epsilon+\hat{\xi}_{i} \\
& \xi_{i} \geq 0, \hat{\xi}_{i} \geq 0, i=1,2, \ldots, m
\end{aligned}
$$
![image-20200829194132523](media\Chapter6_支持向量机\image-20200829194132523.png)

下面是和软硬间隔SVM同样流程的对偶形式推导

![image-20200829194411439](media\Chapter6_支持向量机\image-20200829194411439.png)

首先是拉格朗日函数
$$
L(\boldsymbol{w}, b, \boldsymbol{\alpha}, \hat{\boldsymbol{\alpha}}, \boldsymbol{\xi}, \dot{\boldsymbol{\xi}}, \boldsymbol{\mu}, \hat{\boldsymbol{\mu}})=\frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{m}\left(\xi_{i}+\hat{\xi}_{i}\right)-\sum_{i=1}^{m} \mu_{i} \xi_{i}-\sum_{i=1}^{m} \hat{\mu}_{i} \hat{\xi}_{i}+\sum_{i=1}^{m} \alpha_{i}\left(f\left(x_{i}\right)-y_{i}-\epsilon-\xi_{i}\right)+\sum_{i=1}^{m} \hat{a}_{i}\left(y_{i}-f\left(x_{i}\right)-\epsilon-\hat{\xi}_{i}\right)
$$
然后是求偏导
$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial w}=0 &=\frac{\partial}{\partial w}\left(\frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{m}\left(\xi_{i}+\hat{\xi}_{i}\right)-\sum_{i=1}^{m} \mu_{i} \xi_{i}-\sum_{i=1}^{m} \hat{\mu}_{i} \hat{\xi}_{i}+\sum_{i=1}^{m} \alpha_{i}\left(f\left(x_{i}\right)-y_{i}-\epsilon-\xi_{i}\right)+\sum_{i=1}^{m} \hat{a}_{i}\left(y_{i}-f\left(x_{i}\right)-\epsilon-\hat{\xi}_{i}\right)\right) \\
&=\frac{\partial}{\partial w}\left(\frac{1}{2}\|w\|^{2}+\sum_{i=1}^{m} \alpha_{i} f\left(x_{i}\right)-\sum_{i=1}^{m} \hat{a}_{i} f\left(x_{i}\right)\right) \\
&=\frac{\partial}{\partial w}\left(\frac{1}{2}\|w\|^{2}+\sum_{i=1}^{m} \alpha_{i}\left(w^{T} x_{i}+b\right)-\sum_{i=1}^{m} \hat{a}_{i}\left(w^{T} x_{i}+b\right)\right) \\
&=w+\sum_{i=1}^{m}\left(\alpha_{i}-\hat{a}_{i}\right) x_{i}
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial b}=0 &=\frac{\partial}{\partial b}\left(\frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{m}\left(\xi_{i}+\hat{\xi}_{i}\right)-\sum_{i=1}^{m} \mu_{i} \xi_{i}-\sum_{i=1}^{m} \hat{\mu}_{i} \hat{\xi}_{i}+\sum_{i=1}^{m} \alpha_{i}\left(f\left(\boldsymbol{x}_{i}\right)-y_{i}-\epsilon-\xi_{i}\right)+\sum_{i=1}^{m} \hat{a}_{i}\left(y_{i}-f\left(\boldsymbol{x}_{i}\right)-\epsilon-\hat{\xi}_{i}\right)\right) \\
&=\frac{\partial}{\partial b}\left(\sum_{i=1}^{m} \alpha_{i} f\left(\boldsymbol{x}_{i}\right)-\sum_{i=1}^{m} \hat{a}_{i} f\left(\boldsymbol{x}_{i}\right)\right) \\
&=\frac{\partial}{\partial b}\left(\sum_{i=1}^{m} \alpha_{i}\left(\boldsymbol{w}^{T} \boldsymbol{x}_{i}+b\right)-\sum_{i=1}^{m} \hat{a}_{i}\left(\boldsymbol{w}^{T} x_{i}+b\right)\right) \\
&=\sum_{i=1}^{m}\left(\alpha_{i}-\hat{\alpha}_{i}\right)
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \xi_{i}}=0 &=\frac{\partial}{\partial \xi_{i}}\left(\frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{m}\left(\xi_{i}+\hat{\xi}_{i}\right)-\sum_{i=1}^{m} \mu_{i} \xi_{i}-\sum_{i=1}^{m} \hat{\mu}_{i} \hat{\xi}_{i}+\sum_{i=1}^{m} \alpha_{i}\left(f\left(\boldsymbol{x}_{i}\right)-y_{i}-\epsilon-\xi_{i}\right)+\sum_{i=1}^{m} \hat{\alpha}_{i}\left(y_{i}-f\left(\boldsymbol{x}_{i}\right)-\epsilon-\hat{\xi}_{i}\right)\right) \\
&=\frac{\partial}{\partial \xi_{i}}\left(C \sum_{i=1}^{m}\left(\xi_{i}\right)-\sum_{i=1}^{m} \mu_{i} \xi_{i}+\sum_{i=1}^{m} \alpha_{i}\left(-\xi_{i}\right)\right) \\
&=\frac{\partial}{\partial \mathcal{E}_{i}}\left(C \xi_{i}-\mu_{i} \xi_{i}-\alpha_{i} \xi_{i}\right) \\
&=C-\mu_{i}-\alpha_{i}
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \hat{\xi}_{i}}=0 &=\frac{\partial}{\partial \xi_{i}}\left(\frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{m}\left(\xi_{i}+\xi_{i}\right)-\sum_{i=1}^{m} \mu_{i} \xi_{i}-\sum_{i=1}^{m} \hat{\mu}_{i} \hat{\xi}_{i}+\sum_{i=1}^{m} \alpha_{i}\left(f\left(x_{i}\right)-y_{i}-\epsilon-\xi_{i}\right)+\sum_{i=1}^{m} \hat{a}_{i}\left(y_{i}-f\left(\boldsymbol{x}_{i}\right)-\epsilon-\hat{\xi}_{i}\right)\right) \\
&=\frac{\partial}{\partial \xi_{i}}\left(C \sum_{i=1}^{m}\left(\xi_{i}\right)-\sum_{i=1}^{m} \hat{\mu}_{i} \hat{\xi}_{i}+\sum_{i=1}^{m} \alpha_{i}\left(-\xi_{i}\right)\right) \\
&=\frac{\partial}{\partial \xi_{i}}\left(C \xi_{i}-\hat{\mu}_{i} \xi_{i}-\hat{\alpha}_{i} \xi_{i}\right) \\
&=C-\hat{\mu}_{i}-\hat{a}_{i}
\end{aligned}
$$

代入拉格朗日函数得，第二个等号是把第一行的移项，然后利用上面的第三四个偏导结论可得后面的一坨为零。接着倒数第三个等号利用了第一二个偏导的结果。最后一个等号就是移项，然后把w换成了sum α。

![image-20200829195528573](media\Chapter6_支持向量机\image-20200829195528573.png)

接着是约束的推导
$$
\left\{\begin{array}{l}
\alpha_{i}, \hat{\alpha}_{i} \geq 0 \\
\mu_{i}, \hat{\mu}_{i} \geq 0 \\
\sum_{i=1}^{m} \hat{\alpha}_{i}-\alpha_{i}=0
\end{array} \Rightarrow\left\{\begin{array}{l}
\alpha_{i}, \hat{\alpha}_{i} \geq 0 \\
C-\alpha_{i}, C-\hat{\alpha}_{i} \geq 0 \Rightarrow\left\{\begin{array}{l}
0 \leq \alpha_{i}, \hat{\alpha}_{i} \leq C \\
\sum_{i=1}^{m} \hat{\alpha}_{i}-\alpha_{i}=0
\end{array}\right. \\
\sum_{i=1}^{m} \hat{\alpha}_{i}-\alpha_{i}=0
\end{array}\right.\right.
$$
合在一起就是对偶问题的标准形式了
$$
\begin{aligned}
&\begin{array}{c}
\max _{\alpha, \alpha} \sum_{i=1}^{m} y_{i}\left(\hat{\alpha}_{i}-\alpha_{i}\right)-\epsilon\left(\hat{\alpha}_{i}+\alpha_{i}\right)-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m}\left(\hat{\alpha}_{i}-\alpha_{i}\right)\left(\hat{\alpha}_{j}-\alpha_{j}\right) x_{i}^{T} x_{j}
\end{array}\\
&\text { s.t. } \sum_{i=1}^{m}\left(\bar{a}_{i}-\alpha_{i}\right)=0\\
& \quad 0 \leqslant \alpha_{i}, \bar{a}_{i} \leqslant C
\end{aligned}
$$
最后，我们来分析下KKT条件 ==6.52==
$$
\left\{\begin{array}{l}
\alpha_{i}\left(f\left(\boldsymbol{x}_{i}\right)-y_{i}-\epsilon-\xi_{i}\right)=0 \\
\hat{\alpha}_{i}\left(y_{i}-f\left(\boldsymbol{x}_{i}\right)-\epsilon-\hat{\xi}_{i}\right)=0 \\
\alpha_{i} \hat{\alpha}_{i}=0, \xi_{i} \hat{\xi}_{i}=0 \\
\left(C-\alpha_{i}\right) \xi_{i}=0,\left(C-\hat{\alpha}_{i}\right) \hat{\xi}_{i}=0
\end{array}\right.
$$
可以看到，第1，2和最后一行是complementary slackness equations。最后一行只是做了$\mu_i=C-\alpha_i$的改写。

那为什么$\alpha_{i} \hat{\alpha}_{i}=0$呢？因为如果假设αi!=0，则$y_{i}-f\left(\boldsymbol{x}_{i}\right)=-\epsilon-\xi_{i}$，把这个代入第二个式子得$-\epsilon-\xi_{i}-\epsilon-\hat{\xi}_{i}$，这个恒小于零，所以$\hat{\alpha}_i=0$。

进一步推导，如果$\hat{\alpha}_i=0$，则又由于$\left(C-\hat{\alpha}_{i}\right) \hat{\xi}_{i}=0$且C大于零，那$\hat{\xi}_{i}$必须为零。所以就得到了上面KKT条件的第三行。

和SVM不同的是，这里当α不等于零时，我们获得的支持向量不仅是边界上的，同时也是落在边界以外的。

![image-20200829202146092](media\Chapter6_支持向量机\image-20200829202146092.png)

式6.47就是$\frac{\partial \mathcal{L}}{\partial w}=0 \Rightarrow w=\sum_{i=1}^{m}\left(\hat{a}_{i}-\alpha_{i}\right) x_{i}$。

![image-20200829221546344](media\Chapter6_支持向量机\image-20200829221546344.png)

![image-20200829222637457](media\Chapter6_支持向量机\image-20200829222637457.png)

## 6.6 核方法

### 6.59

$$h(\boldsymbol{x})=\boldsymbol{w}^{\mathrm{T}}\phi(\boldsymbol{x})$$
[解析]：由于书上已经交代公式(6.60)是公式(3.35)引入核函数后的形式，而公式(3.35)是二分类LDA的损失函数，并且此式为直线方程，所以此时讨论的KLDA应当也是二分类KLDA。那么此公式就类似于第3章图3.3里的$y=\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x}$，表示的是二分类KLDA中所要求解的那条投影直线。

复习一下3.35

![image-20200829223645354](media\Chapter6_支持向量机\image-20200829223645354.png)

![image-20200829232416087](media\Chapter6_支持向量机\image-20200829232416087.png)

### 6.60

$$
\max _{\boldsymbol{w}} J(\boldsymbol{w})=\frac{\boldsymbol{w}^{\mathrm{T}} \mathbf{S}_{b}^{\phi} \boldsymbol{w}}{\boldsymbol{w}^{\mathrm{T}} \mathbf{S}_{w}^{\phi} \boldsymbol{w}}
$$
[解析]：类似于第3章的公式(3.35)。

### 6.62

$$
\mathbf{S}_{b}^{\phi}=\left(\boldsymbol{\mu}_{1}^{\phi}-\boldsymbol{\mu}_{0}^{\phi}\right)\left(\boldsymbol{\mu}_{1}^{\phi}-\boldsymbol{\mu}_{0}^{\phi}\right)^{\mathrm{T}}
$$
[解析]：类似于第3章的公式(3.34)。

### 6.63

$$
\mathbf{S}_{w}^{\phi}=\sum_{i=0}^{1} \sum_{\boldsymbol{x} \in X_{i}}\left(\phi(\boldsymbol{x})-\boldsymbol{\mu}_{i}^{\phi}\right)\left(\phi(\boldsymbol{x})-\boldsymbol{\mu}_{i}^{\phi}\right)^{\mathrm{T}}
$$
[解析]：类似于第3章的公式(3.33)。

### 6.65

$$
\boldsymbol{w}=\sum_{i=1}^{m} \alpha_{i} \phi\left(\boldsymbol{x}_{i}\right)
$$
[推导]：由表示定理可知，此时二分类KLDA最终求得的投影直线方程总可以写成如下形式
$$
h(\boldsymbol{x})=\sum_{i=1}^{m} \alpha_{i} \kappa\left(\boldsymbol{x}, \boldsymbol{x}_{i}\right)
$$
又因为直线方程的固定形式为
$$
h(\boldsymbol{x})=\boldsymbol{w}^{\mathrm{T}}\phi(\boldsymbol{x})
$$
所以
$$
\boldsymbol{w}^{\mathrm{T}}\phi(\boldsymbol{x})=\sum_{i=1}^{m} \alpha_{i} \kappa\left(\boldsymbol{x}, \boldsymbol{x}_{i}\right)
$$
将$\kappa\left(\boldsymbol{x}, \boldsymbol{x}_{i}\right)=\phi(\boldsymbol{x})^{\mathrm{T}}\phi(\boldsymbol{x}_i)$代入可得
$$
\boldsymbol{w}^{\mathrm{T}}\phi(\boldsymbol{x})=\sum_{i=1}^{m} \alpha_{i} \phi(\boldsymbol{x})^{\mathrm{T}}\phi(\boldsymbol{x}_i)
$$
由于$\boldsymbol{w}^{\mathrm{T}}\phi(\boldsymbol{x})$的计算结果为标量，而标量的转置等于其本身，所以
$$
\boldsymbol{w}^{\mathrm{T}}\phi(\boldsymbol{x})=\left(\boldsymbol{w}^{\mathrm{T}}\phi(\boldsymbol{x})\right)^{\mathrm{T}}=\phi(\boldsymbol{x})^{\mathrm{T}}\boldsymbol{w} \\
\sum_{i=1}^{m} \alpha_{i} \phi(\boldsymbol{x})^{\mathrm{T}}\phi(\boldsymbol{x}_i)=\phi(\boldsymbol{x})^{\mathrm{T}}\sum_{i=1}^{m} \alpha_{i} \phi(\boldsymbol{x}_i)
$$
$$
\boldsymbol{w}=\sum_{i=1}^{m} \alpha_{i} \phi(\boldsymbol{x}_i)
$$

### 6.66

$$
\hat{\boldsymbol{\mu}}_{0}=\frac{1}{m_{0}} \mathbf{K} \mathbf{1}_{0}
$$
[解析]：为了详细地说明此公式的计算原理，下面首先先举例说明，然后再在例子的基础上延展出其一般形式。假设此时仅有4个样本，其中第1和第3个样本的标记为0，第2和第4个样本的标记为1，那么此时：$m=4$，$m_0=2,m_1=2$
$$
X_0=\{\boldsymbol{x}_1,\boldsymbol{x}_3\},X_1=\{\boldsymbol{x}_2,\boldsymbol{x}_4\}
$$

$$
\mathbf{K}=\left[ \begin{array}{cccc}
\kappa\left(\boldsymbol{x}_1, \boldsymbol{x}_1\right) & \kappa\left(\boldsymbol{x}_1, \boldsymbol{x}_2\right) & \kappa\left(\boldsymbol{x}_1, \boldsymbol{x}_3\right) & \kappa\left(\boldsymbol{x}_1, \boldsymbol{x}_4\right)\\ 
\kappa\left(\boldsymbol{x}_2, \boldsymbol{x}_1\right) & \kappa\left(\boldsymbol{x}_2, \boldsymbol{x}_2\right) & \kappa\left(\boldsymbol{x}_2, \boldsymbol{x}_3\right) & \kappa\left(\boldsymbol{x}_2, \boldsymbol{x}_4\right)\\ 
\kappa\left(\boldsymbol{x}_3, \boldsymbol{x}_1\right) & \kappa\left(\boldsymbol{x}_3, \boldsymbol{x}_2\right) & \kappa\left(\boldsymbol{x}_3, \boldsymbol{x}_3\right) & \kappa\left(\boldsymbol{x}_3, \boldsymbol{x}_4\right)\\ 
\kappa\left(\boldsymbol{x}_4, \boldsymbol{x}_1\right) & \kappa\left(\boldsymbol{x}_4, \boldsymbol{x}_2\right) & \kappa\left(\boldsymbol{x}_4, \boldsymbol{x}_3\right) & \kappa\left(\boldsymbol{x}_4, \boldsymbol{x}_4\right)\\ 
\end{array} \right]\in \mathbb{R}^{4\times 4}
$$

$$
\mathbf{1}_{0}=\left[ \begin{array}{c}
1\\ 
0\\ 
1\\ 
0\\ 
\end{array} \right]\in \mathbb{R}^{4\times 1}
$$

$$
\mathbf{1}_{1}=\left[ \begin{array}{c}
0\\ 
1\\ 
0\\ 
1\\ 
\end{array} \right]\in \mathbb{R}^{4\times 1}
$$


所以
$$
\hat{\boldsymbol{\mu}}_{0}=\frac{1}{m_{0}} \mathbf{K} \mathbf{1}_{0}=\frac{1}{2}\left[ \begin{array}{c}
\kappa\left(\boldsymbol{x}_1, \boldsymbol{x}_1\right)+\kappa\left(\boldsymbol{x}_1, \boldsymbol{x}_3\right)\\ 
\kappa\left(\boldsymbol{x}_2, \boldsymbol{x}_1\right)+\kappa\left(\boldsymbol{x}_2, \boldsymbol{x}_3\right)\\ 
\kappa\left(\boldsymbol{x}_3, \boldsymbol{x}_1\right)+\kappa\left(\boldsymbol{x}_3, \boldsymbol{x}_3\right)\\ 
\kappa\left(\boldsymbol{x}_4, \boldsymbol{x}_1\right)+\kappa\left(\boldsymbol{x}_4, \boldsymbol{x}_3\right)\\ 
\end{array} \right]\in \mathbb{R}^{4\times 1}
$$

$$
\hat{\boldsymbol{\mu}}_{1}=\frac{1}{m_{1}} \mathbf{K} \mathbf{1}_{1}=\frac{1}{2}\left[ \begin{array}{c}
\kappa\left(\boldsymbol{x}_1, \boldsymbol{x}_2\right)+\kappa\left(\boldsymbol{x}_1, \boldsymbol{x}_4\right)\\ 
\kappa\left(\boldsymbol{x}_2, \boldsymbol{x}_2\right)+\kappa\left(\boldsymbol{x}_2, \boldsymbol{x}_4\right)\\ 
\kappa\left(\boldsymbol{x}_3, \boldsymbol{x}_2\right)+\kappa\left(\boldsymbol{x}_3, \boldsymbol{x}_4\right)\\ 
\kappa\left(\boldsymbol{x}_4, \boldsymbol{x}_2\right)+\kappa\left(\boldsymbol{x}_4, \boldsymbol{x}_4\right)\\ 
\end{array} \right]\in \mathbb{R}^{4\times 1}
$$

根据此结果易得$\hat{\boldsymbol{\mu}}_{0},\hat{\boldsymbol{\mu}}_{1}$的一般形式为
$$
\hat{\boldsymbol{\mu}}_{0}=\frac{1}{m_{0}} \mathbf{K} \mathbf{1}_{0}=\frac{1}{m_{0}}\left[ \begin{array}{c}
\sum_{\boldsymbol{x} \in X_{0}}\kappa\left(\boldsymbol{x}_1, \boldsymbol{x}\right)\\ 
\sum_{\boldsymbol{x} \in X_{0}}\kappa\left(\boldsymbol{x}_2, \boldsymbol{x}\right)\\ 
\vdots\\ 
\sum_{\boldsymbol{x} \in X_{0}}\kappa\left(\boldsymbol{x}_m, \boldsymbol{x}\right)\\ 
\end{array} \right]\in \mathbb{R}^{m\times 1}
$$

$$
\hat{\boldsymbol{\mu}}_{1}=\frac{1}{m_{1}} \mathbf{K} \mathbf{1}_{1}=\frac{1}{m_{1}}\left[ \begin{array}{c}
\sum_{\boldsymbol{x} \in X_{1}}\kappa\left(\boldsymbol{x}_1, \boldsymbol{x}\right)\\ 
\sum_{\boldsymbol{x} \in X_{1}}\kappa\left(\boldsymbol{x}_2, \boldsymbol{x}\right)\\ 
\vdots\\ 
\sum_{\boldsymbol{x} \in X_{1}}\kappa\left(\boldsymbol{x}_m, \boldsymbol{x}\right)\\ 
\end{array} \right]\in \mathbb{R}^{m\times 1}
$$

### 6.67

$$
\hat{\boldsymbol{\mu}}_{1}=\frac{1}{m_{1}} \mathbf{K} \mathbf{1}_{1}
$$

[解析]：参见公式(6.66)的解析。

### 6.70

==如果是LDA，没有核，直接用6.60就可以解了，但现在为了使用核函数来高效求解，就转化成了下面的这个求解形式。==
$$
\max _{\boldsymbol{\alpha}} J(\boldsymbol{\alpha})=\frac{\boldsymbol{\alpha}^{\mathrm{T}} \mathbf{M} \boldsymbol{\alpha}}{\boldsymbol{\alpha}^{\mathrm{T}} \mathbf{N} \boldsymbol{\alpha}}
$$
[推导]：此公式是将公式(6.65)代入公式(6.60)后推得而来的，下面给出详细地推导过程。首先将公式(6.65)代入公式(6.60)的分子可得：
$$
\begin{aligned}
\boldsymbol{w}^{\mathrm{T}} \mathbf{S}_{b}^{\phi} \boldsymbol{w}&=\left(\sum_{i=1}^{m} \alpha_{i} \phi\left(\boldsymbol{x}_{i}\right)\right)^{\mathrm{T}}\cdot\mathbf{S}_{b}^{\phi}\cdot \sum_{i=1}^{m} \alpha_{i} \phi\left(\boldsymbol{x}_{i}\right) \\
&=\sum_{i=1}^{m} \alpha_{i} \phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}}\cdot\mathbf{S}_{b}^{\phi}\cdot \sum_{i=1}^{m} \alpha_{i} \phi\left(\boldsymbol{x}_{i}\right) \\
\end{aligned}
$$
其中
$$
\begin{aligned}
\mathbf{S}_{b}^{\phi} &=\left(\boldsymbol{\mu}_{1}^{\phi}-\boldsymbol{\mu}_{0}^{\phi}\right)\left(\boldsymbol{\mu}_{1}^{\phi}-\boldsymbol{\mu}_{0}^{\phi}\right)^{\mathrm{T}} \\
&=\left(\frac{1}{m_{1}} \sum_{\boldsymbol{x} \in X_{1}} \phi(\boldsymbol{x})-\frac{1}{m_{0}} \sum_{\boldsymbol{x} \in X_{0}} \phi(\boldsymbol{x})\right)\left(\frac{1}{m_{1}} \sum_{\boldsymbol{x} \in X_{1}} \phi(\boldsymbol{x})-\frac{1}{m_{0}} \sum_{\boldsymbol{x} \in X_{0}} \phi(\boldsymbol{x})\right)^{\mathrm{T}} \\
&=\left(\frac{1}{m_{1}} \sum_{\boldsymbol{x} \in X_{1}} \phi(\boldsymbol{x})-\frac{1}{m_{0}} \sum_{\boldsymbol{x} \in X_{0}} \phi(\boldsymbol{x})\right)\left(\frac{1}{m_{1}} \sum_{\boldsymbol{x} \in X_{1}} \phi(\boldsymbol{x})^{\mathrm{T}}-\frac{1}{m_{0}} \sum_{\boldsymbol{x} \in X_{0}} \phi(\boldsymbol{x})^{\mathrm{T}}\right) \\
\end{aligned}
$$
将其代入上式可得
$$
\begin{aligned}
\boldsymbol{w}^{\mathrm{T}} \mathbf{S}_{b}^{\phi} \boldsymbol{w}=&\sum_{i=1}^{m} \alpha_{i} \phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}}\cdot\left(\frac{1}{m_{1}} \sum_{\boldsymbol{x} \in X_{1}} \phi(\boldsymbol{x})-\frac{1}{m_{0}} \sum_{\boldsymbol{x} \in X_{0}} \phi(\boldsymbol{x})\right)\cdot\left(\frac{1}{m_{1}} \sum_{\boldsymbol{x} \in X_{1}} \phi(\boldsymbol{x})^{\mathrm{T}}-\frac{1}{m_{0}} \sum_{\boldsymbol{x} \in X_{0}} \phi(\boldsymbol{x})^{\mathrm{T}}\right)\cdot \sum_{i=1}^{m} \alpha_{i} \phi\left(\boldsymbol{x}_{i}\right) \\
=&\left(\frac{1}{m_{1}} \sum_{\boldsymbol{x} \in X_{1}}\sum_{i=1}^{m} \alpha_{i} \phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}} \phi(\boldsymbol{x})-\frac{1}{m_{0}} \sum_{\boldsymbol{x} \in X_{0}} \sum_{i=1}^{m} \alpha_{i} \phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}}\phi(\boldsymbol{x})\right)\\
&\cdot\left(\frac{1}{m_{1}} \sum_{\boldsymbol{x} \in X_{1}} \sum_{i=1}^{m} \alpha_{i} \phi(\boldsymbol{x})^{\mathrm{T}}\phi\left(\boldsymbol{x}_{i}\right)-\frac{1}{m_{0}} \sum_{\boldsymbol{x} \in X_{0}} \sum_{i=1}^{m} \alpha_{i} \phi(\boldsymbol{x})^{\mathrm{T}}\phi\left(\boldsymbol{x}_{i}\right)\right) \\
\end{aligned}
$$
由于$\kappa\left(\boldsymbol{x}_i, \boldsymbol{x}\right)=\phi(\boldsymbol{x}_i)^{\mathrm{T}}\phi(\boldsymbol{x})$为标量，所以其转置等于本身，也即$\kappa\left(\boldsymbol{x}_i, \boldsymbol{x}\right)=\phi(\boldsymbol{x}_i)^{\mathrm{T}}\phi(\boldsymbol{x})=\left(\phi(\boldsymbol{x}_i)^{\mathrm{T}}\phi(\boldsymbol{x})\right)^{\mathrm{T}}=\phi(\boldsymbol{x})^{\mathrm{T}}\phi(\boldsymbol{x}_i)=\kappa\left(\boldsymbol{x}_i, \boldsymbol{x}\right)^{\mathrm{T}}$，将其代入上式可得
$$
\begin{aligned}
\boldsymbol{w}^{\mathrm{T}} \mathbf{S}_{b}^{\phi} \boldsymbol{w}=&\left(\frac{1}{m_{1}} \sum_{i=1}^{m}\sum_{\boldsymbol{x} \in X_{1}}\alpha_{i} \kappa\left(\boldsymbol{x}_i, \boldsymbol{x}\right)-\frac{1}{m_{0}} \sum_{i=1}^{m} \sum_{\boldsymbol{x} \in X_{0}}  \alpha_{i} \kappa\left(\boldsymbol{x}_i, \boldsymbol{x}\right)\right)\\
&\cdot\left(\frac{1}{m_{1}} \sum_{i=1}^{m}\sum_{\boldsymbol{x} \in X_{1}} \alpha_{i} \kappa\left(\boldsymbol{x}_i, \boldsymbol{x}\right)-\frac{1}{m_{0}}\sum_{i=1}^{m}  \sum_{\boldsymbol{x} \in X_{0}} \alpha_{i} \kappa\left(\boldsymbol{x}_i, \boldsymbol{x}\right)\right)
\end{aligned}
$$
令$\boldsymbol{\alpha}=(\alpha_1;\alpha_2;...;\alpha_m)^{\mathrm{T}}\in \mathbb{R}^{m\times 1}$，同时结合公式(6.66)的解析中得到的$\hat{\boldsymbol{\mu}}_{0},\hat{\boldsymbol{\mu}}_{1}$的一般形式，上式可以化简为
$$
\begin{aligned}
\boldsymbol{w}^{\mathrm{T}} \mathbf{S}_{b}^{\phi} \boldsymbol{w}&=\left(\boldsymbol{\alpha}^{\mathrm{T}}\hat{\boldsymbol{\mu}}_{1}-\boldsymbol{\alpha}^{\mathrm{T}}\hat{\boldsymbol{\mu}}_{0}\right)\cdot\left(\hat{\boldsymbol{\mu}}_{1}^{\mathrm{T}}\boldsymbol{\alpha}-\hat{\boldsymbol{\mu}}_{0}^{\mathrm{T}}\boldsymbol{\alpha}\right)\\
&=\boldsymbol{\alpha}^{\mathrm{T}}\cdot\left(\hat{\boldsymbol{\mu}}_{1}-\hat{\boldsymbol{\mu}}_{0}\right)\cdot\left(\hat{\boldsymbol{\mu}}_{1}^{\mathrm{T}}-\hat{\boldsymbol{\mu}}_{0}^{\mathrm{T}}\right)\cdot\boldsymbol{\alpha}\\
&=\boldsymbol{\alpha}^{\mathrm{T}}\cdot\left(\hat{\boldsymbol{\mu}}_{1}-\hat{\boldsymbol{\mu}}_{0}\right)\cdot\left(\hat{\boldsymbol{\mu}}_{1}-\hat{\boldsymbol{\mu}}_{0}\right)^{\mathrm{T}}\cdot\boldsymbol{\alpha}\\
&=\boldsymbol{\alpha}^{\mathrm{T}} \mathbf{M} \boldsymbol{\alpha}\\
\end{aligned}
$$
以上便是公式(6.70)分子部分的推导，下面继续推导公式(6.70)的分母部分。将公式(6.65)代入公式(6.60)的分母可得：
$$
\begin{aligned}
\boldsymbol{w}^{\mathrm{T}} \mathbf{S}_{w}^{\phi} \boldsymbol{w}&=\left(\sum_{i=1}^{m} \alpha_{i} \phi\left(\boldsymbol{x}_{i}\right)\right)^{\mathrm{T}}\cdot\mathbf{S}_{w}^{\phi}\cdot \sum_{i=1}^{m} \alpha_{i} \phi\left(\boldsymbol{x}_{i}\right) \\
&=\sum_{i=1}^{m} \alpha_{i} \phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}}\cdot\mathbf{S}_{w}^{\phi}\cdot \sum_{i=1}^{m} \alpha_{i} \phi\left(\boldsymbol{x}_{i}\right) \\
\end{aligned}
$$
其中
$$
\begin{aligned}
\mathbf{S}_{w}^{\phi}&=\sum_{i=0}^{1} \sum_{\boldsymbol{x} \in X_{i}}\left(\phi(\boldsymbol{x})-\boldsymbol{\mu}_{i}^{\phi}\right)\left(\phi(\boldsymbol{x})-\boldsymbol{\mu}_{i}^{\phi}\right)^{\mathrm{T}} \\
&=\sum_{i=0}^{1} \sum_{\boldsymbol{x} \in X_{i}}\left(\phi(\boldsymbol{x})-\boldsymbol{\mu}_{i}^{\phi}\right)\left(\phi(\boldsymbol{x})^{\mathrm{T}}-\left(\boldsymbol{\mu}_{i}^{\phi}\right)^{\mathrm{T}}\right) \\
&=\sum_{i=0}^{1} \sum_{\boldsymbol{x} \in X_{i}}\left(\phi(\boldsymbol{x})\phi(\boldsymbol{x})^{\mathrm{T}}-\phi(\boldsymbol{x})\left(\boldsymbol{\mu}_{i}^{\phi}\right)^{\mathrm{T}}-\boldsymbol{\mu}_{i}^{\phi}\phi(\boldsymbol{x})^{\mathrm{T}}+\boldsymbol{\mu}_{i}^{\phi}\left(\boldsymbol{\mu}_{i}^{\phi}\right)^{\mathrm{T}}\right) \\
\end{aligned}
$$
由于$\phi(\boldsymbol{x})\left(\boldsymbol{\mu}_{i}^{\phi}\right)^{\mathrm{T}}$的外面再配上sum的结果是一个对称矩阵，所以$\phi(\boldsymbol{x})\left(\boldsymbol{\mu}_{i}^{\phi}\right)^{\mathrm{T}}=\left[\phi(\boldsymbol{x})\left(\boldsymbol{\mu}_{i}^{\phi}\right)^{\mathrm{T}}\right]^{\mathrm{T}}=\boldsymbol{\mu}_{i}^{\phi}\phi(\boldsymbol{x})^{\mathrm{T}}$，将其代回上式可得
$$
\begin{aligned}
\mathbf{S}_{w}^{\phi}&=\sum_{i=0}^{1} \sum_{\boldsymbol{x} \in X_{i}}\left(\phi(\boldsymbol{x})\phi(\boldsymbol{x})^{\mathrm{T}}-2\boldsymbol{\mu}_{i}^{\phi}\phi(\boldsymbol{x})^{\mathrm{T}}+\boldsymbol{\mu}_{i}^{\phi}\left(\boldsymbol{\mu}_{i}^{\phi}\right)^{\mathrm{T}}\right) \\
&=\sum_{i=0}^{1} \sum_{\boldsymbol{x} \in X_{i}}\phi(\boldsymbol{x})\phi(\boldsymbol{x})^{\mathrm{T}}-\sum_{i=0}^{1} \sum_{\boldsymbol{x} \in X_{i}}2\boldsymbol{\mu}_{i}^{\phi}\phi(\boldsymbol{x})^{\mathrm{T}}+\sum_{i=0}^{1} \sum_{\boldsymbol{x} \in X_{i}}\boldsymbol{\mu}_{i}^{\phi}\left(\boldsymbol{\mu}_{i}^{\phi}\right)^{\mathrm{T}} \\
&=\sum_{\boldsymbol{x} \in  D}\phi(\boldsymbol{x})\phi(\boldsymbol{x})^{\mathrm{T}}-2\boldsymbol{\mu}_{0}^{\phi}\sum_{\boldsymbol{x} \in X_{0}}\phi(\boldsymbol{x})^{\mathrm{T}}-2\boldsymbol{\mu}_{1}^{\phi}\sum_{\boldsymbol{x} \in X_{1}}\phi(\boldsymbol{x})^{\mathrm{T}}+\sum_{\boldsymbol{x} \in X_{0}}\boldsymbol{\mu}_{0}^{\phi}\left(\boldsymbol{\mu}_{0}^{\phi}\right)^{\mathrm{T}}+\sum_{\boldsymbol{x} \in X_{1}}\boldsymbol{\mu}_{1}^{\phi}\left(\boldsymbol{\mu}_{1}^{\phi}\right)^{\mathrm{T}} \\
&=\sum_{\boldsymbol{x} \in  D}\phi(\boldsymbol{x})\phi(\boldsymbol{x})^{\mathrm{T}}-2m_0\boldsymbol{\mu}_{0}^{\phi}\left(\boldsymbol{\mu}_{0}^{\phi}\right)^{\mathrm{T}}-2m_1\boldsymbol{\mu}_{1}^{\phi}\left(\boldsymbol{\mu}_{1}^{\phi}\right)^{\mathrm{T}}+m_0 \boldsymbol{\mu}_{0}^{\phi}\left(\boldsymbol{\mu}_{0}^{\phi}\right)^{\mathrm{T}}+m_1 \boldsymbol{\mu}_{1}^{\phi}\left(\boldsymbol{\mu}_{1}^{\phi}\right)^{\mathrm{T}} \\
&=\sum_{\boldsymbol{x} \in  D}\phi(\boldsymbol{x})\phi(\boldsymbol{x})^{\mathrm{T}}-m_0\boldsymbol{\mu}_{0}^{\phi}\left(\boldsymbol{\mu}_{0}^{\phi}\right)^{\mathrm{T}}-m_1\boldsymbol{\mu}_{1}^{\phi}\left(\boldsymbol{\mu}_{1}^{\phi}\right)^{\mathrm{T}}\\
\end{aligned}
$$
再将此式代回$\boldsymbol{w}^{\mathrm{T}} \mathbf{S}_{b}^{\phi} \boldsymbol{w}$可得
$$
\begin{aligned}
\boldsymbol{w}^{\mathrm{T}} \mathbf{S}_{w}^{\phi} \boldsymbol{w}=&\sum_{i=1}^{m} \alpha_{i} \phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}}\cdot\mathbf{S}_{w}^{\phi}\cdot \sum_{i=1}^{m} \alpha_{i} \phi\left(\boldsymbol{x}_{i}\right) \\
=&\sum_{i=1}^{m} \alpha_{i} \phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}}\cdot\left(\sum_{\boldsymbol{x} \in  D}\phi(\boldsymbol{x})\phi(\boldsymbol{x})^{\mathrm{T}}-m_0\boldsymbol{\mu}_{0}^{\phi}\left(\boldsymbol{\mu}_{0}^{\phi}\right)^{\mathrm{T}}-m_1\boldsymbol{\mu}_{1}^{\phi}\left(\boldsymbol{\mu}_{1}^{\phi}\right)^{\mathrm{T}}\right)\cdot \sum_{i=1}^{m} \alpha_{i} \phi\left(\boldsymbol{x}_{i}\right) \\
=&\sum_{i=1}^{m}\sum_{j=1}^{m}\sum_{\boldsymbol{x} \in  D}\alpha_{i} \phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}}\phi(\boldsymbol{x})\phi(\boldsymbol{x})^{\mathrm{T}}\alpha_{j} \phi\left(\boldsymbol{x}_{j}\right)-\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i} \phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}}m_0\boldsymbol{\mu}_{0}^{\phi}\left(\boldsymbol{\mu}_{0}^{\phi}\right)^{\mathrm{T}}\alpha_{j} \phi\left(\boldsymbol{x}_{j}\right)\\
&-\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i} \phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}}m_1\boldsymbol{\mu}_{1}^{\phi}\left(\boldsymbol{\mu}_{1}^{\phi}\right)^{\mathrm{T}}\alpha_{j} \phi\left(\boldsymbol{x}_{j}\right) \\
\end{aligned}
$$
其中，第1项可化简为
$$
\begin{aligned}
\sum_{i=1}^{m}\sum_{j=1}^{m}\sum_{\boldsymbol{x} \in  D}\alpha_{i} \phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}}\phi(\boldsymbol{x})\phi(\boldsymbol{x})^{\mathrm{T}}\alpha_{j} \phi\left(\boldsymbol{x}_{j}\right)&=\sum_{i=1}^{m}\sum_{j=1}^{m}\sum_{\boldsymbol{x} \in  D}\alpha_{i} \alpha_{j}\kappa\left(\boldsymbol{x}_i, \boldsymbol{x}\right)\kappa\left(\boldsymbol{x}_j, \boldsymbol{x}\right)\\
&=\boldsymbol{\alpha}^{\mathrm{T}} \mathbf{K} \mathbf{K}^{\mathrm{T}} \boldsymbol{\alpha}
\end{aligned}
$$
第2项可化简为
$$
\begin{aligned}
\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i} \phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}}m_0\boldsymbol{\mu}_{0}^{\phi}\left(\boldsymbol{\mu}_{0}^{\phi}\right)^{\mathrm{T}}\alpha_{j} \phi\left(\boldsymbol{x}_{j}\right)&=m_0\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i}\alpha_{j}\phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}}\boldsymbol{\mu}_{0}^{\phi}\left(\boldsymbol{\mu}_{0}^{\phi}\right)^{\mathrm{T}} \phi\left(\boldsymbol{x}_{j}\right)\\
&=m_0\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i}\alpha_{j}\phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}}\left[\frac{1}{m_{0}} \sum_{\boldsymbol{x} \in X_{0}} \phi(\boldsymbol{x})\right]\left[\frac{1}{m_{0}} \sum_{\boldsymbol{x} \in X_{0}} \phi(\boldsymbol{x})\right]^{\mathrm{T}} \phi\left(\boldsymbol{x}_{j}\right)\\
&=m_0\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i}\alpha_{j}\left[\frac{1}{m_{0}} \sum_{\boldsymbol{x} \in X_{0}} \phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}}\phi(\boldsymbol{x})\right]\left[\frac{1}{m_{0}} \sum_{\boldsymbol{x} \in X_{0}} \phi(\boldsymbol{x})^{\mathrm{T}}\phi\left(\boldsymbol{x}_{j}\right)\right] \\
&=m_0\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i}\alpha_{j}\left[\frac{1}{m_{0}} \sum_{\boldsymbol{x} \in X_{0}} \kappa\left(\boldsymbol{x}_i, \boldsymbol{x}\right)\right]\left[\frac{1}{m_{0}} \sum_{\boldsymbol{x} \in X_{0}} \kappa\left(\boldsymbol{x}_j, \boldsymbol{x}\right)\right] \\
&=m_0\boldsymbol{\alpha}^{\mathrm{T}} \hat{\boldsymbol{\mu}}_{0} \hat{\boldsymbol{\mu}}_{0}^{\mathrm{T}} \boldsymbol{\alpha}
\end{aligned}
$$
同理可得，第3项可化简为
$$
\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i} \phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}}m_1\boldsymbol{\mu}_{1}^{\phi}\left(\boldsymbol{\mu}_{1}^{\phi}\right)^{\mathrm{T}}\alpha_{j} \phi\left(\boldsymbol{x}_{j}\right)=m_1\boldsymbol{\alpha}^{\mathrm{T}} \hat{\boldsymbol{\mu}}_{1} \hat{\boldsymbol{\mu}}_{1}^{\mathrm{T}} \boldsymbol{\alpha}
$$
将上述三项的化简结果代回再将此式代回$\boldsymbol{w}^{\mathrm{T}} \mathbf{S}_{b}^{\phi} \boldsymbol{w}$可得
$$
\begin{aligned}
\boldsymbol{w}^{\mathrm{T}} \mathbf{S}_{b}^{\phi} \boldsymbol{w}&=\boldsymbol{\alpha}^{\mathrm{T}} \mathbf{K} \mathbf{K}^{\mathrm{T}} \boldsymbol{\alpha}-m_0\boldsymbol{\alpha}^{\mathrm{T}} \hat{\boldsymbol{\mu}}_{0} \hat{\boldsymbol{\mu}}_{0}^{\mathrm{T}} \boldsymbol{\alpha}-m_1\boldsymbol{\alpha}^{\mathrm{T}} \hat{\boldsymbol{\mu}}_{1} \hat{\boldsymbol{\mu}}_{1}^{\mathrm{T}} \boldsymbol{\alpha}\\
&=\boldsymbol{\alpha}^{\mathrm{T}} \cdot\left(\mathbf{K} \mathbf{K}^{\mathrm{T}} -m_0\hat{\boldsymbol{\mu}}_{0} \hat{\boldsymbol{\mu}}_{0}^{\mathrm{T}} -m_1\hat{\boldsymbol{\mu}}_{1} \hat{\boldsymbol{\mu}}_{1}^{\mathrm{T}} \right)\cdot\boldsymbol{\alpha}\\
&=\boldsymbol{\alpha}^{\mathrm{T}} \cdot\left(\mathbf{K} \mathbf{K}^{\mathrm{T}}-\sum_{i=0}^{1} m_{i} \hat{\boldsymbol{\mu}}_{i} \hat{\boldsymbol{\mu}}_{i}^{\mathrm{T}} \right)\cdot\boldsymbol{\alpha}\\
&=\boldsymbol{\alpha}^{\mathrm{T}} \mathbf{N}\boldsymbol{\alpha}\\
\end{aligned}
$$






