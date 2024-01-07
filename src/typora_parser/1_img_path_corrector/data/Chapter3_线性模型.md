笔记的前一部分主要是对机器学习预备知识的概括，包括机器学习的定义/术语、学习器性能的评估/度量以及比较，本篇之后将主要对具体的学习算法进行理解总结，本篇则主要是第3章的内容--线性模型。

# 3 线性模型

谈及线性模型，其实我们很早就已经与它打过交道，还记得高中数学必修3课本中那个顽皮的“最小二乘法”吗？这就是线性模型的经典算法之一：根据给定的（x，y）点对，求出一条与这些点拟合效果最好的直线y=ax+b，之前我们利用下面的公式便可以计算出拟合直线的系数a,b（3.1中给出了具体的计算过程），从而对于一个新的x，可以预测它所对应的y值。前面我们提到：在机器学习的术语中，当预测值为连续值时，称为“回归问题”，离散值时为“分类问题”。本篇先从线性回归任务开始，接着讨论分类和多分类问题。

![1.png](media\Chapter3_线性模型\5bc722b068e48.png)

## **3.1 线性回归**

线性回归问题就是试图学到一个线性模型尽可能准确地预测新样本的输出值，例如：通过历年的人口数据预测2017年人口数量。在这类问题中，往往我们会先得到一系列的有标记数据，例如：2000-->13亿...2016-->15亿，这时输入的属性只有一个，即年份；也有输入多属性的情形，假设我们预测一个人的收入，这时输入的属性值就不止一个了，例如：（学历，年龄，性别，颜值，身高，体重）-->15k。

有时这些输入的属性值并不能直接被我们的学习模型所用，需要进行相应的处理，对于连续值的属性，一般都可以被学习器所用，有时会根据具体的情形作相应的预处理，例如：归一化等；对于离散值的属性，可作下面的处理：

- 若属性值之间存在“序关系”，则可以将其转化为连续值，例如：身高属性分为“高”“中等”“矮”，可转化为数值：{1， 0.5， 0}。

- 若属性值之间不存在“序关系”，则通常将其转化为向量的形式，例如：性别属性分为“男”“女”，可转化为二维向量：{（1，0），（0，1）}。

### 3.1.1

当输入属性只有一个的时候，就是最简单的情形，也就是我们高中时最熟悉的“最小二乘法”（Euclidean distance），首先计算出每个样本预测值与真实值之间的误差并求和，通过最小化均方误差MSE，使用求偏导等于零的方法计算出拟合直线y=wx+b的两个参数w和b，计算过程如下图所示：

![2.png](media\Chapter3_线性模型\5bc722b0ccec4.png)

注意这上面的证明的前提条件是：$E_{w,b}$是关于w和b的凸函数。判断函数凹凸特性有两种方式，第一种是书上写的：

1. ![image-20200225200942164](media\Chapter3_线性模型\image-20200225200942164.png)
2. ![image-20200225200955523](media\Chapter3_线性模型\image-20200225200955523.png)
3. 下面这个是更为严格的定义，面对非实数集也可进行判断：

![vlcsnap-2020-02-25-20h02m47s688](media\Chapter3_线性模型\vlcsnap-2020-02-25-20h02m47s688.png)

要看上面的这个凸函数条件是否满足，要求我们求二阶偏导，我们首先可以对一阶导数也就是公式3.5和3.6进行推导：

#### eq. 3.5

$$
\cfrac{\partial E_{(w, b)}}{\partial w}=2\left(w \sum_{i=1}^{m} x_{i}^{2}-\sum_{i=1}^{m}\left(y_{i}-b\right) x_{i}\right)
$$
[推导]：已知$E_{(w, b)}=\sum\limits_{i=1}^{m}\left(y_{i}-w x_{i}-b\right)^{2}$，所以
$$
\begin{aligned}
\cfrac{\partial E_{(w, b)}}{\partial w}&=\cfrac{\partial}{\partial w} \left[\sum_{i=1}^{m}\left(y_{i}-w x_{i}-b\right)^{2}\right] \\
&= \sum_{i=1}^{m}\cfrac{\partial}{\partial w} \left[\left(y_{i}-w x_{i}-b\right)^{2}\right] \\
&= \sum_{i=1}^{m}\left[2\cdot\left(y_{i}-w x_{i}-b\right)\cdot (-x_i)\right] \\
&= \sum_{i=1}^{m}\left[2\cdot\left(w x_{i}^2-y_i x_i +bx_i\right)\right] \\
&= 2\cdot\left(w\sum_{i=1}^{m} x_{i}^2-\sum_{i=1}^{m}y_i x_i +b\sum_{i=1}^{m}x_i\right) \\
&=2\left(w \sum_{i=1}^{m} x_{i}^{2}-\sum_{i=1}^{m}\left(y_{i}-b\right) x_{i}\right)
\end{aligned}
$$

#### eq. 3.6

$$\cfrac{\partial E_{(w, b)}}{\partial b}=2\left(m b-\sum_{i=1}^{m}\left(y_{i}-w x_{i}\right)\right)$$
[推导]：已知$E_{(w, b)}=\sum\limits_{i=1}^{m}\left(y_{i}-w x_{i}-b\right)^{2}$，所以
$$
\begin{aligned}
\cfrac{\partial E_{(w, b)}}{\partial b}&=\cfrac{\partial}{\partial b} \left[\sum_{i=1}^{m}\left(y_{i}-w x_{i}-b\right)^{2}\right] \\
&=\sum_{i=1}^{m}\cfrac{\partial}{\partial b} \left[\left(y_{i}-w x_{i}-b\right)^{2}\right] \\
&=\sum_{i=1}^{m}\left[2\cdot\left(y_{i}-w x_{i}-b\right)\cdot (-1)\right] \\
&=\sum_{i=1}^{m}\left[2\cdot\left(b-y_{i}+w x_{i}\right)\right] \\
&=2\cdot\left[\sum_{i=1}^{m}b-\sum_{i=1}^{m}y_{i}+\sum_{i=1}^{m}w x_{i}\right] \\
&=2\left(m b-\sum_{i=1}^{m}\left(y_{i}-w x_{i}\right)\right)
\end{aligned}
$$
==我们再带入二阶导数来看是否为凸函数==：
$$
\begin{aligned}
\frac{\partial^{2} E_{(w, b)}}{\partial w^{2}} &=\frac{\partial}{\partial w}\left(\frac{\partial E_{(w, b)}}{\partial w}\right) \\
&=\frac{\partial}{\partial w}\left[2\left(w \sum_{i=1}^{m} x_{i}^{2}-\sum_{i=1}^{m}\left(y_{i}-b\right) x_{i}\right)\right] \\
&=\frac{\partial}{\partial w}\left[2 w \sum_{i=1}^{m} x_{i}^{2}\right] \\
&=2 \sum_{i=1}^{m} x_{i}^{2}=A \\
\end{aligned}
$$
然后看B：
$$
\begin{aligned}
\frac{\partial^{2} E_{(w, b)}}{\partial w \partial b} &=\frac{\partial}{\partial b}\left(\frac{\partial E_{(w, b)}}{\partial w}\right) \\
&=\frac{\partial}{\partial b}\left[2\left(w \sum_{i=1}^{m} x_{i}^{2}-\sum_{i=1}^{m}\left(y_{i}-b\right) x_{i}\right)\right] \\
&=\frac{\partial}{\partial b}\left[-2 \sum_{i=1}^{m}\left(y_{i}-b\right) x_{i}\right] \\
&=\frac{\partial}{\partial b}\left(-2 \sum_{i=1}^{m} y_{i} x_{i}+2 \sum_{i=1}^{m} b x_{i}\right) \\
&=\frac{\partial}{\partial b}\left(2 \sum_{i=1}^{m} b x_{i}\right)=2 \sum_{i=1}^{m} x_{i}=B
\end{aligned}
$$
最后再来看C：
$$
\begin{aligned}
\frac{\partial^{2} E_{(w, b)}}{\partial b^{2}} &=\frac{\partial}{\partial b}\left(\frac{\partial E_{(w, b)}}{\partial b}\right) \\
&=\frac{\partial}{\partial b}\left[2\left(m b-\sum_{i=1}^{m}\left(y_{i}-w x_{i}\right)\right)\right] \\
&=\frac{\partial}{\partial b}(2 m b) \\
&=2m
\end{aligned}
$$
然后我们就能来看是否满足严格凸函数的条件(二阶导矩阵正定 - Schur complement)：A>0, AC-B^2^>=0。
$$
\begin{aligned}
{A=2 \sum_{i=1}^{m} x_{i}^{2} \quad B=2 \sum_{i=1}^{m} x_{i} \quad C=2 m} \\
\end{aligned}
$$
$$
\begin{aligned}
{A C-B^{2}=2 m \cdot 2 \sum_{i=1}^{m} x_{i}^{2}-\left(2 \sum_{i=1}^{m} x_{i}\right)^{2}=4 m \sum_{i=1}^{m} x_{i}^{2}-4\left(\sum_{i=1}^{m} x_{i}\right)^{2}=4 m \sum_{i=1}^{m} x_{i}^{2}-4 \cdot m \cdot \frac{1}{m} \cdot\left(\sum_{i=1}^{m} x_{i}\right)^{2}} \\
{\quad=4 m \sum_{i=1}^{m} x_{i}^{2}-4 m \cdot \bar{x} \cdot \sum_{i=1}^{m} x_{i}=4 m\left(\sum_{i=1}^{m} x_{i}^{2}-\sum_{i=1}^{m} x_{i} \bar{x}\right)=4 m \sum_{i=1}^{m}\left(x_{i}^{2}-x_{i} \bar{x}\right)}
\end{aligned}
$$
$$
又因为：\\
\sum_{i=1}^{m} x_{i} \bar{x}=\bar{x} \sum_{i=1}^{m} x_{i}=\bar{x} \cdot m \cdot \frac{1}{m} \cdot \sum_{i=1}^{m} x_{i}=m \bar{x}^{2}=\sum_{i=1}^{m} \bar{x}^{2}\\
所以:\\
=4m\sum_{i=1}^{m}\left(x_{i}^{2}-x_{i} \bar{x}-x_{i} \bar{x}+x_{i} \bar{x}\right)=4 m \sum_{i=1}^{m}\left(x_{i}^{2}-x_{i} \bar{x}-x_{i} \bar{x}+\bar{x}^{2}\right)=4 m \sum_{i=1}^{m}\left(x_{i}-\bar{x}\right)^{2}\\
\square
$$

所以上式恒大于等于零，凸函数得证。

#### eq. 3.7

$$ w=\cfrac{\sum_{i=1}^{m}y_i(x_i-\bar{x})}{\sum_{i=1}^{m}x_i^2-\cfrac{1}{m}(\sum_{i=1}^{m}x_i)^2} $$
[推导]：令式（3.6）等于0：
$$
\begin{aligned}
&\frac{\partial E_{(w, b)}}{\partial b}=2\left(m b-\sum_{i=1}^{m}\left(y_{i}-w x_{i}\right)\right)=0\\
&m b-\sum_{i=1}^{m}\left(y_{i}-w x_{i}\right)=0\\
&b=\frac{1}{m} \sum_{i=1}^{m}\left(y_{i}-w x_{i}\right)
=\bar{y}-w \bar{x}
\end{aligned}
$$
令式（3.5）等于0，讲上面的结果带入3.5可得：
$$
\begin{aligned}	 
    w\sum_{i=1}^{m}x_i^2 & = \sum_{i=1}^{m}y_ix_i-\sum_{i=1}^{m}(\bar{y}-w\bar{x})x_i \\
    w\sum_{i=1}^{m}x_i^2 & = \sum_{i=1}^{m}y_ix_i-\bar{y}\sum_{i=1}^{m}x_i+w\bar{x}\sum_{i=1}^{m}x_i \\
    w(\sum_{i=1}^{m}x_i^2-\bar{x}\sum_{i=1}^{m}x_i) & = \sum_{i=1}^{m}y_ix_i-\bar{y}\sum_{i=1}^{m}x_i \\
    w & = \cfrac{\sum_{i=1}^{m}y_ix_i-\bar{y}\sum_{i=1}^{m}x_i}{\sum_{i=1}^{m}x_i^2-\bar{x}\sum_{i=1}^{m}x_i}
\end{aligned}
$$
又因为：
$$
\bar{y}\sum_{i=1}^{m}x_i=\cfrac{1}{m}\sum_{i=1}^{m}y_i\sum_{i=1}^{m}x_i=\bar{x}\sum_{i=1}^{m}y_i \\ \bar{x}\sum_{i=1}^{m}x_i=\cfrac{1}{m}\sum_{i=1}^{m}x_i\sum_{i=1}^{m}x_i=\cfrac{1}{m}(\sum_{i=1}^{m}x_i)^2
$$
代入上式即可得式（3.7）。$\square$

【注】：式（3.7）还可以进一步化简为能用向量表达的形式，将$ \cfrac{1}{m}(\sum_{i=1}^{m}x_i)^2=\bar{x}\sum_{i=1}^{m}x_i $代入分母可得：
$$
\begin{aligned}	  
     w & = \cfrac{\sum_{i=1}^{m}y_i(x_i-\bar{x})}{\sum_{i=1}^{m}x_i^2-\bar{x}\sum_{i=1}^{m}x_i} \\
     & = \cfrac{\sum_{i=1}^{m}(y_ix_i-y_i\bar{x})}{\sum_{i=1}^{m}(x_i^2-x_i\bar{x})}
\end{aligned}
$$
我们可以利用以下三个变形对上式继续简化：
$$
\begin{aligned}
&\sum_{i=1}^{m} y_{i} \bar{x}=\bar{x} \sum_{i=1}^{m} y_{i}=\frac{1}{m} \sum_{i=1}^{m} x_{i} \sum_{i=1}^{m} y_{i}=\sum_{i=1}^{m} x_{i} \cdot \frac{1}{m} \cdot \sum_{i=1}^{m} y_{i}=\sum_{i=1}^{m} x_{i} \bar{y}\\
&\sum_{i=1}^{m} y_{i} \bar{x}=\bar{x} \sum_{i=1}^{m} y_{i}=\bar{x} \cdot m \cdot \frac{1}{m} \cdot \sum_{i=1}^{m} y_{i}=m \bar{x} \bar{y}=\sum_{i=1}^{m} \bar{x} \bar{y}\\
&\sum_{i=1}^{m} x_{i} \bar{x}=\bar{x} \sum_{i=1}^{m} x_{i}=\bar{x} \cdot m \cdot \frac{1}{m} \cdot \sum_{i=1}^{m} x_{i}=m \bar{x}^{2}=\sum_{i=1}^{m} \bar{x}^{2}
\end{aligned}
$$
则上式可化为：
$$
\begin{aligned}
w &=\frac{\sum_{i=1}^{m}\left(y_{i} x_{i}-y_{i} \bar{x}\right)}{\sum_{i=1}^{m}\left(x_{i}^{2}-x_{i} \bar{x}\right)}=\frac{\sum_{i=1}^{m}\left(y_{i} x_{i}-y_{i} \bar{x}-y_{i} \bar{x}+y_{i} \bar{x}\right)}{\sum_{i=1}^{m}\left(x_{i}^{2}-x_{i} \bar{x}-x_{i} \bar{x}+x_{i} \bar{x}\right)} \\
&=\frac{\sum_{i=1}^{m}\left(y_{i} x_{i}-y_{i} \bar{x}-x_{i} \bar{y}+\bar{x} \bar{y}\right)}{\sum_{i=1}^{m}\left(x_{i}^{2}-x_{i} \bar{x}-x_{i} \bar{x}+\bar{x}^{2}\right)}=\frac{\sum_{i=1}^{m}\left(x_{i}-\bar{x}\right)\left(y_{i}-\bar{y}\right)}{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}}
\end{aligned}
$$
若令$ \boldsymbol{x}=(x_1,x_2,...,x_m)^T $，$ \boldsymbol{x}_{d}=(x_1-\bar{x},x_2-\bar{x},...,x_m-\bar{x})^T $为去均值后的$ \boldsymbol{x} $，$ \boldsymbol{y}=(y_1,y_2,...,y_m)^T $，$ \boldsymbol{y}_{d}=(y_1-\bar{y},y_2-\bar{y},...,y_m-\bar{y})^T $为去均值后的$ \boldsymbol{y} $，其中$ \boldsymbol{x} $、$ \boldsymbol{x}_{d} $、$ \boldsymbol{y} $、$ \boldsymbol{y}_{d} $均为m行1列的列向量，代入上式可得：
$$
w=\cfrac{\boldsymbol{x}_{d}^T\boldsymbol{y}_{d}}{\boldsymbol{x}_d^T\boldsymbol{x}_{d}}
$$

### 3.1.2

当输入属性有多个的时候，例如对于一个样本有d个属性{（x1,x2...xd）,y}，则y=wx+b需要写成：

![0.png](media\Chapter3_线性模型\5bc72567b8bcd.png)

通常对于多元问题，常常使用矩阵的形式来表示数据。在本问题中，将具有m个样本的数据集表示成矩阵X，将系数w与b合并成一个列向量，这样每个样本的预测值以及所有样本的均方误差最小化就可以写成下面的形式：

![3.png](media\Chapter3_线性模型\5bc722b0ad8f7.png)

![4.png](media\Chapter3_线性模型\5bc722b0af652.png)

然后我们继续把损失函数用最小二乘法的形式表达出来：
$$
\begin{aligned}
\boldsymbol{y} &=\left(y_{1}, y_{2}, \ldots, y_{m}\right)^{T} \\
E_{\hat{\boldsymbol{w}}} &=\sum_{i=1}^{m}\left(y_{i}-\hat{\boldsymbol{w}}^{T} \hat{\boldsymbol{x}}_{i}\right)^{2} \\
&=\left(y_{1}-\hat{\boldsymbol{w}}^{T} \hat{\boldsymbol{x}}_{1}\right)^{2}+\left(y_{2}-\hat{\boldsymbol{w}}^{T} \hat{\boldsymbol{x}}_{2}\right)^{2}+\ldots+\left(y_{m}-\hat{\boldsymbol{w}}^{T} \hat{\boldsymbol{x}}_{m}\right)^{2}
\end{aligned}
$$

$$
E_{\hat{\boldsymbol{w}}}=\left(\begin{array}{c}
{y_{1}-\hat{\boldsymbol{w}}^{T} \hat{\boldsymbol{x}}_{1}} & {y_{2}-\hat{\boldsymbol{w}}^{T} \hat{\boldsymbol{x}}_{2}} & {\cdots} & {y_{m}-\hat{\boldsymbol{w}}^{T} \hat{\boldsymbol{x}}_{m}}
\end{array}\right)\left(\begin{array}{c}
{y_{1}-\hat{\boldsymbol{w}}^{T} \hat{\boldsymbol{x}}_{1}} \\
{y_{2}-\hat{\boldsymbol{w}}^{T} \hat{\boldsymbol{x}}_{2}} \\
{\vdots} \\
{y_{m}-\hat{\boldsymbol{w}}^{T} \hat{\boldsymbol{x}}_{m}}
\end{array}\right)
$$

$$
\begin{aligned}
&\left(\begin{array}{c}
{y_{1}-\hat{\boldsymbol{w}}^{T} \hat{\boldsymbol{x}}_{1}} \\
{y_{2}-\hat{\boldsymbol{w}}^{T} \hat{\boldsymbol{x}}_{2}} \\
{\vdots} \\
{y_{m}-\hat{\boldsymbol{w}}^{T} \hat{\boldsymbol{x}}_{m}}
\end{array}\right)=\left(\begin{array}{c}
{y_1}\\{y_2}\\{\vdots}\\{y_m}
\end{array}\right)-\left(\begin{array}{c}
{\boldsymbol{\hat { x }}_{1}^{T} \hat{\boldsymbol{w}}} \\
{\hat{\boldsymbol{x}}_{2}^{T} \hat{\boldsymbol{w}}} \\
{\vdots} \\
{\boldsymbol{\hat { x }}_{m}^{T} \hat{\boldsymbol{w}}}
\end{array}\right)\\
&\left(\begin{array}{c}
{\hat{\boldsymbol{x}}_{1}^{T} \hat{\boldsymbol{w}}} \\
{\hat{\boldsymbol{x}}_{2}^{T} \hat{\boldsymbol{w}}} \\
{\vdots} \\
{\hat{\boldsymbol{x}}_{m}^{T} \hat{\boldsymbol{w}}}
\end{array}\right)=\left(\begin{array}{c}
{\hat{\boldsymbol{x}}_{1}^{T}} \\
{\hat{\boldsymbol{x}}_{2}^{T}} \\
{\vdots} \\
{\hat{\boldsymbol{x}}_{m}^{T}}
\end{array}\right) \cdot \hat{\boldsymbol{w}}=\mathbf{X} \cdot \hat{\boldsymbol{w}}
\end{aligned}
$$

然后我们就把上式写成inner product的形式了：

![5.png](media\Chapter3_线性模型\5bc722b090543.png)

同样地，要对上式求最小值，我们首先证明它是凸函数，然后令均方误差的求导等于0：

![image-20200225212940194](media\Chapter3_线性模型\image-20200225212940194.png)

那怎么证明这个函数是凸函数呢？由下面两条定理推导：

![vlcsnap-2020-02-25-21h57m04s885](media\Chapter3_线性模型\vlcsnap-2020-02-25-21h57m04s885.png)

证明H正定就可以了。然后由于是凸函数，所以当一阶导数为零，我们就有了全局最小解。首先证明H正定：
$$
\begin{aligned}
\frac{\partial E_{\hat{w}}}{\partial \hat{\boldsymbol{w}}} &=\frac{\partial}{\partial \hat{\boldsymbol{w}}}\left[(\boldsymbol{y}-\mathbf{X} \hat{\boldsymbol{w}})^{T}(\boldsymbol{y}-\mathbf{X} \hat{\boldsymbol{w}})\right] \\
&=\frac{\partial}{\partial \hat{\boldsymbol{w}}}\left[\left(\boldsymbol{y}^{T}-\hat{\boldsymbol{w}}^{T} \mathbf{X}^{T}\right)(\boldsymbol{y}-\mathbf{X} \hat{\boldsymbol{w}})\right] \\
&=\frac{\partial}{\partial \hat{\boldsymbol{w}}}\left[\boldsymbol{y}^{T} \boldsymbol{y}-\boldsymbol{y}^{T} \mathbf{X} \hat{\boldsymbol{w}}-\hat{\boldsymbol{w}}^{T} \mathbf{X}^{T} \boldsymbol{y}+\hat{\boldsymbol{w}}^{T} \mathbf{X}^{T} \mathbf{X} \hat{\boldsymbol{w}}\right] \\
&=\frac{\partial}{\partial \hat{\boldsymbol{w}}}\left[-\boldsymbol{y}^{T} \mathbf{X} \hat{\boldsymbol{w}}-\hat{\boldsymbol{w}}^{T} \mathbf{X}^{T} \boldsymbol{y}+\hat{\boldsymbol{w}}^{T} \mathbf{X}^{T} \mathbf{X} \hat{\boldsymbol{w}}\right]
\end{aligned}
$$
这里补充一下矩阵微分的两个公式：
$$
\frac{\partial \boldsymbol{x}^{T} \boldsymbol{a}}{\partial \boldsymbol{x}}=\frac{\partial \boldsymbol{a}^{T} \boldsymbol{x}}{\partial \boldsymbol{x}}=\left(\begin{array}{c}
{\frac{\partial\left(a_{1} x_{1}+a_{2} x_{2}+\ldots+a_{n} x_{n}\right)}{\partial x_{1}}} \\
{\frac{\partial\left(a_{1} x_{1}+a_{2} x_{2}+\ldots+a_{n} x_{n}\right)}{\partial x_{2}}} \\
{\vdots} \\
{\frac{\partial\left(a_{1} x_{1}+a_{2} x_{2}+\ldots+a_{n} x_{n}\right)}{\partial x_{n}}}
\end{array}\right)=\left(\begin{array}{c}
{a_{1}} \\
{a_{2}} \\
{\vdots} \\
{a_{n}}
\end{array}\right)=\boldsymbol{a}
$$

$$
\frac{\partial \boldsymbol{x}^{T} \mathbf{B} \boldsymbol{x}}{\partial \boldsymbol{x}}=\left(\mathbf{B}+\mathbf{B}^{T}\right) \boldsymbol{x}
$$

#### eq. 3.10

然后利用这两个公式，我们对上面的损失函数进行微分：
$$
\begin{aligned}
\frac{\partial E_{\hat{\boldsymbol{w}}}}{\partial \hat{\boldsymbol{w}}} 
&=-\frac{\partial \boldsymbol{y}^{T} \mathbf{X} \hat{\boldsymbol{w}}}{\partial \hat{\boldsymbol{w}}}-\frac{\partial \hat{\boldsymbol{w}}^{T} \mathbf{X}^{T} \boldsymbol{y}}{\partial \hat{\boldsymbol{w}}}+\frac{\partial \hat{\boldsymbol{w}}^{T} \mathbf{X}^{T} \mathbf{X} \hat{\boldsymbol{w}}}{\partial \hat{\boldsymbol{w}}}
=-\mathbf{X}^{T} \boldsymbol{y}-\mathbf{X}^{T} \boldsymbol{y}+\left(\mathbf{X}^{T} \mathbf{X}+\mathbf{X}^{T} \mathbf{X}\right) \hat{w}=2 \mathbf{X}^{T}(\mathbf{X} \hat{w}-\boldsymbol{y})
\end{aligned}
$$
这是书上的==eq. 3.10==一阶偏导式，接着求二阶：
$$
\begin{aligned}
\frac{\partial^{2} E_{\hat{w}}}{\partial \hat{w} \partial \hat{w}^{T}} &=\frac{\partial}{\partial \hat{w}}\left(\frac{\partial E_{\hat{w}}}{\partial \hat{w}}\right) \\
&=\frac{\partial}{\partial \hat{w}}\left[2 \mathbf{X}^{T}(\mathbf{X} \hat{w}-\boldsymbol{y})\right] \\
&=\frac{\partial}{\partial \hat{w}}\left(2 \mathbf{X}^{T} \mathbf{X} \hat{w}-2 \mathbf{X}^{T} \boldsymbol{y}\right) \\
&=2 \mathbf{X}^{T} \mathbf{X}
\end{aligned}
$$
这里的$\mathbf{X}^{T} \mathbf{X}$必须是满秩且正定的，我们才能说我们的损失函数是凸函数的。而这个在真实场景中，往往是不一定的。我们采取的办法是正则化：

![image-20200225221824212](media\Chapter3_线性模型\image-20200225221824212.png)

对于具体怎样正则化，暂时不去深入。我们开始求最优解：

#### eq. 3.11

$$
\begin{aligned}
&\frac{\partial E_{\hat{w}}}{\partial \hat{w}}=2 \mathbf{X}^{T}(\mathbf{X} \hat{w}-\boldsymbol{y})=0\\
&2 \mathbf{X}^{T} \mathbf{X} \hat{w}-2 \mathbf{X}^{T} \boldsymbol{y}=0\\
&2 \mathbf{X}^{T} \mathbf{X} \hat{w}=2 \mathbf{X}^{T} \boldsymbol{y}\\
&\hat{w}=\left(\mathbf{X}^{T} \mathbf{X}\right)^{-1} \mathbf{X}^{T} \boldsymbol{y}
\end{aligned}
$$

#### 对数线性回归

有时像上面这种原始的线性回归可能并不能满足需求，例如：y值并不是线性变化，而是在指数尺度上变化。这时我们可以采用线性模型来逼近y的衍生物，例如lny，这时衍生的线性模型如下所示，实际上就是相当于将指数曲线投影在一条直线上，如下图所示：

![7.png](media\Chapter3_线性模型\5bc722b103cbf.png)

#### 广义线性回归

更一般地，考虑所有y的衍生物的情形，就得到了“广义的线性模型”（generalized linear model），其中，g（*）称为联系函数（link function）。

![8.png](media\Chapter3_线性模型\5bc722b0a2841.png)

## **3.2 线性几率回归**

回归就是通过输入的属性值得到一个预测值，利用上述广义线性模型的特征，是否可以通过一个联系函数，将预测值转化为离散值从而进行分类呢？线性几率回归正是研究这样的问题。对数几率引入了一个对数几率函数（logistic function）,将预测值投影到0-1之间，从而将线性回归问题转化为二分类问题。

![9.png](media\Chapter3_线性模型\5bc722b0c7748.png)

![10.png](media\Chapter3_线性模型\5bc722b0a655d.png)

若将y看做样本为正例的概率，（1-y）看做样本为反例的概率，则上式实际上使用线性回归模型的预测结果器逼近真实标记的对数几率。因此这个模型称为“对数几率回归”（logistic regression），也有一些书籍称之为“逻辑回归”。下面使用最大似然估计的方法来计算出w和b两个参数的取值，下面只列出求解的思路，不列出具体的计算过程。

![11.png](media\Chapter3_线性模型\5bc723b824f0c.png)

然后我们通过最大化对数似然函数来估计w和b。

![12.png](media\Chapter3_线性模型\5bc723b817961.png)

要推导上面这个式子3.25和3.26还有3.27，我们需要首先补充两个相关概念：

1. exponential family；
2. 广义线性模型的三条假设。

### 指数族分布

#### 定义

首先，指数族必须满足以下的一般形式：

![vlcsnap-2020-02-26-01h43m55s744](media\Chapter3_线性模型\vlcsnap-2020-02-26-01h43m55s744.png)

normal distribution的公式：$\frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$。可知符合上图刻画。

![image-20200226021503145](media\Chapter3_线性模型\image-20200226021503145.png)

#### Bernoulli

那我们首先来看最基本的example - 证明Bernoulli distribution属于指数族分布。下式是Bernoulli的pdf。
$$
p(y)=\phi^{y}(1-\phi)^{1-y}
$$
其中y属于{0，1}，$\phi$是y=1的概率。

对上式恒等变形可得：
$$
\begin{aligned}
p(y) &=\phi^{y}(1-\phi)^{1-y} \\
&=\exp \left(\ln \left(\phi^{y}(1-\phi)^{1-y}\right)\right) \\
&=\exp \left(\ln \phi^{y}+\ln (1-\phi)^{1-y}\right)
\end{aligned}
$$

$$
\begin{aligned}
p(y) &=\exp (y \ln \phi+(1-y) \ln (1-\phi)) \\
&=\exp (y \ln \phi+\ln (1-\phi)-y \ln (1-\phi)) \\
&=\exp (y(\ln \phi-\ln (1-\phi))+\ln (1-\phi)) \\
&=\exp \left(y \ln \left(\frac{\phi}{1-\phi}\right)+\ln (1-\phi)\right)
\end{aligned}
$$

得到上面的式子之后，我们只需要和一般表达形式对比一下，就能对应各个参数了。这和wiki的定义也能吻合。https://en.wikipedia.org/wiki/Exponential_family
$$
\begin{aligned}
b(y) &=1 \\
\eta &=\ln \left(\frac{\phi}{1-\phi}\right) \\
T(y) &=y \\
a(\eta) &=-\ln (1-\phi)=\ln \left(1+e^{\eta}\right)
\end{aligned}
$$

### 广义线性模型

判定模型是广义线性需要满足三条假设：

![vlcsnap-2020-02-26-02h28m34s177](media\Chapter3_线性模型\vlcsnap-2020-02-26-02h28m34s177.png)

### eq. 3.23

我们来利用这三点用广义线性模型对求输入为x，输出为y的二分类{0, 1}概率问题进行建模（推出该问题h(x)的线性表达式）。

首先，利用第一条假设我们会很自然的想到使用**伯努利分布**，因为它的输出y就是0，1两类。然后利用第二条假设进行推导，可以得出h(x)=$\phi$：

![vlcsnap-2020-02-26-03h15m20s663](media\Chapter3_线性模型\vlcsnap-2020-02-26-03h15m20s663.png)

然后我们要利用第三条假设，首先把$\phi$和$\eta$的关系进行等价变换：
$$
\begin{aligned}
\eta&=\ln \left(\frac{\phi}{1-\phi}\right) \\
e^{\eta}&=\frac{\phi}{1-\phi} \\
e^{-\eta}&=\frac{1-\phi}{\phi}\\
e^{-\eta}&=\frac{1}{\phi}-1\\
\frac{1}{1+e^{-\eta}}&=\phi
\end{aligned}
$$
然后我们把W^T^X带入进$\eta$：
$$
h(\boldsymbol{x})=\phi=\frac{1}{1+e^{-\boldsymbol{w}^{T} \boldsymbol{x}}}=p(y=1 \mid \boldsymbol{x})
$$
这个公式就等价于logistic regression模型：$p(y=1 \mid \boldsymbol{x})=\frac{e^{\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+b}}{1+e^{\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+b}}$

------------

接着我们看看怎么求解参数值w，b。

我们可以通过极大似然估计法来确定W的值：首先我们定义概率密度函数probability density function，然后把抽取样本的联合概率joint probability写成下面的形式。我们要做的就是调整参数使得==抽样发生的联合概率最大==。

![vlcsnap-2020-02-26-03h36m12s316](media\Chapter3_线性模型\vlcsnap-2020-02-26-03h36m12s316.png)

由于求（凸）$\Pi(x)$函数的最大值要这个函数对x微分，这不好操作，所以乘上一个单调递增monotonic increasing的log来化乘为加，简化微分。

### eq. 3.25

![vlcsnap-2020-02-26-03h36m57s627](media\Chapter3_线性模型\vlcsnap-2020-02-26-03h36m57s627.png)

那首先我们简化一下标记方式，

![vlcsnap-2020-02-26-03h37m43s674](media\Chapter3_线性模型\vlcsnap-2020-02-26-03h37m43s674.png)

### eq. 3.26

所以对于分布律我们用下面两种形式来组合都可以：
$$
\begin{aligned}
&p(y | \boldsymbol{x} ; \boldsymbol{w}, b)=y \cdot p_{1}(\hat{\boldsymbol{x}} ; \boldsymbol{\beta})+(1-y) \cdot p_{0}(\hat{\boldsymbol{x}} ; \boldsymbol{\beta})\\
&p(y | \boldsymbol{x} ; \boldsymbol{w}, b)=\left[p_{1}(\hat{\boldsymbol{x}} ; \boldsymbol{\beta})\right]^{y}\left[p_{0}(\hat{\boldsymbol{x}} ; \boldsymbol{\beta})\right]^{1-y}
\end{aligned}
$$
我们先配合书上的选择，用第一个来推导极大似然估计，将它代入对数似然函数可得：
$$
\ell(\boldsymbol{\beta})=\sum_{i=1}^{m} \ln \left(y_{i} p_{1}\left(\hat{\boldsymbol{x}}_{i} ; \boldsymbol{\beta}\right)+\left(1-y_{i}\right) p_{0}\left(\hat{\boldsymbol{x}}_{i} ; \boldsymbol{\beta}\right)\right)
$$
由于$p_{1}\left(\hat{\boldsymbol{x}}_{i} ; \boldsymbol{\beta}\right)=\frac{e^{\boldsymbol{\beta}^{T} \hat{x}_{i}}}{1+e^{\boldsymbol{\beta}^{T} \hat{\boldsymbol{x}}_{i}}}, \quad p_{0}\left(\hat{\boldsymbol{x}}_{i} ; \boldsymbol{\beta}\right)=\frac{1}{1+e^{\boldsymbol{\beta}^{T} \hat{\boldsymbol{x}}_{i}}}$，上式可化为：
$$
\begin{aligned}
\ell(\boldsymbol{\beta}) &=\sum_{i=1}^{m} \ln \left(\frac{y_{i} e^{\boldsymbol{\beta}^{T} \hat{\boldsymbol{x}}_{i}}}{1+e^{\boldsymbol{\beta}^{T} \hat{\boldsymbol{x}}_{i}}}+\frac{1-y_{i}}{1+e^{\boldsymbol{\beta}^{T} \hat{\boldsymbol{x}}_{i}}}\right) \\
&=\sum_{i=1}^{m} \ln \left(\frac{y_{i} e^{\boldsymbol{\beta}^{T} \hat{\boldsymbol{x}}_{i}}+1-y_{i}}{1+e^{\boldsymbol{\beta}^{T} \hat{\boldsymbol{x}}_{i}}}\right) \\
&=\sum_{i=1}^{m}\left(\ln \left(y_{i} e^{\boldsymbol{\beta}^{T} \hat{\boldsymbol{x}}_{i}}+1-y_{i}\right)-\ln \left(1+e^{\boldsymbol{\beta}^{T} \hat{\boldsymbol{x}}_{i}}\right)\right)
\end{aligned}
$$
由于$y_{i} \in\{0,1\}$，而且我们只有两种情况，所以把y=0或1的情况带入上式可得：

* 当y~i~=0时：

$$
\ell(\boldsymbol{\beta})=\sum_{i=1}^{m}\left(\ln \left(0 \cdot e^{\boldsymbol{\beta}^{T} \boldsymbol{x}_{i}}+1-0\right)-\ln \left(1+e^{\boldsymbol{\beta}^{T} \boldsymbol{x}_{i}}\right)\right)=\sum_{i=1}^{m}\left(\ln 1-\ln \left(1+e^{\boldsymbol{\beta}^{T} \boldsymbol{x}_{i}}\right)\right)=\sum_{i=1}^{m}\left(-\ln \left(1+e^{\boldsymbol{\beta}^{T} \boldsymbol{x}_{i}}\right)\right)
$$
* 当y~i~=1时：

$$
\ell(\beta)=\sum_{i=1}^{m}\left(\ln \left(1 \cdot e^{\beta^{T} x_{i}}+1-1\right)-\ln \left(1+e^{\beta^{T}{x_{i}}}\right)\right)=\sum_{i=1}^{m}\left(\ln e^{\beta^{T} x_{i}}-\ln \left(1+e^{\beta^{T} x_{i}}\right)\right)=\sum_{i=1}^{m}\left(\beta^{T}{{x}_{i}}-\ln \left(1+e^{\beta^{T} x_{i}}\right)\right)
$$

### eq. 3.27

综合上面两种情况可得最大似然估计：
$$
\ell(\boldsymbol{\beta})=\sum_{i=1}^{m}\left(y_{i} \boldsymbol{\beta}^{T} \hat{\boldsymbol{x}}_{i}-\ln \left(1+e^{\boldsymbol{\beta}^{T} \hat{\boldsymbol{x}}_{i}}\right)\right)
$$
如果在前面加上个负号，就能得到损失函数eq. 3.27。max()就变成了min()，这种最优化是没法直接求出解析解的，只能用数值算法去迭代逼近（Gradient Descent，Newton）。

其实用另一种分布律更容易推导出最大似然估计的表达式：
$$
\begin{aligned}
\ell(\boldsymbol{\beta}) &=\sum_{i=1}^{m} \ln \left(\left[p_{1}\left(\hat{\boldsymbol{x}}_{i} ; \boldsymbol{\beta}\right)\right]^{y_{i}}\left[p_{0}\left(\hat{\boldsymbol{x}}_{i} ; \boldsymbol{\beta}\right)\right]^{1-y_{i}}\right) \\
&=\sum_{i=1}^{m}\left[\ln \left(\left[p_{1}\left(\hat{\boldsymbol{x}}_{i} ; \boldsymbol{\beta}\right)\right]^{y_{i}}\right)+\ln \left(\left[p_{0}\left(\hat{\boldsymbol{x}}_{i} ; \boldsymbol{\beta}\right)\right]^{1-y_{i}}\right)\right] \\
&=\sum_{i=1}^{m}\left[y_{i} \ln \left(p_{1}\left(\hat{\boldsymbol{x}}_{i} ; \boldsymbol{\beta}\right)\right)+\left(1-y_{i}\right) \ln \left(p_{0}\left(\hat{\boldsymbol{x}}_{i} ; \boldsymbol{\beta}\right)\right)\right] \\
&=\sum_{i=1}^{m}\left\{y_{i}\left[\ln \left(p_{1}\left(\hat{\boldsymbol{x}}_{i} ; \boldsymbol{\beta}\right)\right)-\ln \left(p_{0}\left(\hat{\boldsymbol{x}}_{i} ; \boldsymbol{\beta}\right)\right)\right]+\ln \left(p_{0}\left(\hat{\boldsymbol{x}}_{i} ; \boldsymbol{\beta}\right)\right)\right\}
\end{aligned}
$$

$$
\ell(\boldsymbol{\beta})=\sum_{i=1}^{m}\left[y_{i} \ln \left(\frac{p_{1}\left(\hat{\boldsymbol{x}}_{i} ; \boldsymbol{\beta}\right)}{p_{0}\left(\hat{\boldsymbol{x}}_{i} ; \boldsymbol{\beta}\right)}\right)+\ln \left(p_{0}\left(\hat{\boldsymbol{x}}_{i} ; \boldsymbol{\beta}\right)\right)\right]
$$

我们在把p1和p0的具体表达式代入进去，我们就可求得：
$$
\begin{aligned}
\ell(\boldsymbol{\beta}) &=\sum_{i=1}^{m}\left[y_{i} \ln \left(e^{\boldsymbol{\beta}^{T} \hat{\boldsymbol{x}}_{i}}\right)+\ln \left(\frac{1}{1+e^{\boldsymbol{\beta}^{T} \hat{\boldsymbol{x}}_{i}}}\right)\right]\\
&=\sum_{i=1}^{m}\left[y_{i} \boldsymbol{\beta}^{T} \hat{\boldsymbol{x}}_{i}-\ln \left(1+e^{\boldsymbol{\beta}^{T} \hat{x}_{i}}\right)\right]
\end{aligned}
$$

### eq. 3.30

上式$\ell(\boldsymbol{\beta})$是最大似然估计，而我们通常求解时会取负，当作最小化处理：$l(\beta)=-\ell(\boldsymbol{\beta})$
$$
\frac{\partial l(\beta)}{\partial \beta}=-\sum_{i=1}^{m}\hat{\boldsymbol x}_i(y_i-p_1(\hat{\boldsymbol x}_i;\beta))
$$
[解析]：此式可以进行向量化，令$p_1(\hat{\boldsymbol x}_i;\beta)=\hat{y}_i$，代入上式得：
$$
\begin{aligned}
	\frac{\partial l(\beta)}{\partial \beta} &= -\sum_{i=1}^{m}\hat{\boldsymbol x}_i(y_i-\hat{y}_i) \\
	& =\sum_{i=1}^{m}\hat{\boldsymbol x}_i(\hat{y}_i-y_i) \\
	& ={\boldsymbol X^T}(\hat{\boldsymbol y}-\boldsymbol{y}) \\
	& ={\boldsymbol X^T}(p_1(\boldsymbol X;\beta)-\boldsymbol{y}) \\
\end{aligned}
$$

## **3.3 线性判别分析**

线性判别分析（Linear Discriminant Analysis，简称LDA）,其基本思想是：将训练样本投影到一条直线上，使得同类的样例尽可能近，不同类的样例尽可能远。如图所示：

![13.png](media\Chapter3_线性模型\5bc723b863ebb.png)

![14.png](media\Chapter3_线性模型\5bc723b85bfa9.png)

想让同类样本点的投影点尽可能接近，不同类样本点投影之间尽可能远，即：让各类的协方差之和尽可能小，不用类之间中心的距离尽可能大。基于这样的考虑，LDA定义了两个散度矩阵。

+ 类内散度矩阵（within-class scatter matrix）

![15.png](media\Chapter3_线性模型\5bc723b8156e1.png)

+ 类间散度矩阵(between-class scaltter matrix)

![16.png](media\Chapter3_线性模型\5bc723b7e9db3.png)

因此得到了LDA的最大化目标：“广义瑞利商”（generalized Rayleigh quotient）。

![17.png](media\Chapter3_线性模型\5bc723b7e8a61.png)

从而分类问题转化为最优化求解w的问题，当求解出w后，对新的样本进行分类时，只需将该样本点投影到这条直线上，根据与各个类别的中心值进行比较，从而判定出新样本与哪个类别距离最近。求解w的方法如下所示，使用的方法为λ乘子。

![18.png](media\Chapter3_线性模型\5bc723b83d5e0.png)

若将w看做一个投影矩阵，类似PCA的思想，则LDA可将样本投影到N-1维空间（N为类簇数），投影的过程使用了类别信息（标记信息），因此LDA也常被视为一种经典的监督降维技术。    
​             

## **3.4 多分类学习**

现实中我们经常遇到不只两个类别的分类问题，即多分类问题，在这种情形下，我们常常运用“拆分”的策略，通过多个二分类学习器来解决多分类问题，即将多分类问题拆解为多个二分类问题，训练出多个二分类学习器，最后将多个分类结果进行集成得出结论。最为经典的拆分策略有三种：“一对一”（OvO）、“一对其余”（OvR）和“多对多”（MvM），核心思想与示意图如下所示。

+ OvO：给定数据集D，假定其中有N个真实类别，将这N个类别进行两两配对（一个正类/一个反类），从而产生N（N-1）/2个二分类学习器，在测试阶段，将新样本放入所有的二分类学习器中测试，得出N（N-1）个结果，最终通过投票产生最终的分类结果。

+ OvM：给定数据集D，假定其中有N个真实类别，每次取出一个类作为正类，剩余的所有类别作为一个新的反类，从而产生N个二分类学习器，在测试阶段，得出N个结果，若仅有一个学习器预测为正类，则对应的类标作为最终分类结果。

+ MvM：给定数据集D，假定其中有N个真实类别，每次取若干个类作为正类，若干个类作为反类（通过ECOC码给出，编码），若进行了M次划分，则生成了M个二分类学习器，在测试阶段（解码），得出M个结果组成一个新的码，最终通过计算海明/欧式距离选择距离最小的类别作为最终分类结果。

![19.png](media\Chapter3_线性模型\5bc723b862bfb.png)

![20.png](media\Chapter3_线性模型\5bc723b8300d5.png)

## **3.5 类别不平衡问题**

类别不平衡（class-imbanlance）就是指分类问题中不同类别的训练样本相差悬殊的情况，例如正例有900个，而反例只有100个，这个时候我们就需要进行相应的处理来平衡这个问题。常见的做法有三种：

1.  在训练样本较多的类别中进行“欠采样”（undersampling）,比如从正例中采出100个，常见的算法有：EasyEnsemble。
2.  在训练样本较少的类别中进行“过采样”（oversampling）,例如通过对反例中的数据进行插值，来产生额外的反例，常见的算法有SMOTE。
3.  直接基于原数据集进行学习，对预测值进行“再缩放”处理。其中再缩放也是代价敏感学习的基础。
4.  ![21.png](media\Chapter3_线性模型\5bc726fe87ae2.png)      

----------

# 南瓜书

## 3.32

$$J=\cfrac{\boldsymbol w^T(\mu_0-\mu_1)(\mu_0-\mu_1)^T\boldsymbol w}{\boldsymbol w^T(\Sigma_0+\Sigma_1)\boldsymbol w}$$
[推导]：
$$\begin{aligned}
	J &= \cfrac{\big|\big|\boldsymbol w^T\mu_0-\boldsymbol w^T\mu_1\big|\big|_2^2}{\boldsymbol w^T(\Sigma_0+\Sigma_1)\boldsymbol w} \\
	&= \cfrac{\big|\big|(\boldsymbol w^T\mu_0-\boldsymbol w^T\mu_1)^T\big|\big|_2^2}{\boldsymbol w^T(\Sigma_0+\Sigma_1)\boldsymbol w} \\
	&= \cfrac{\big|\big|(\mu_0-\mu_1)^T\boldsymbol w\big|\big|_2^2}{\boldsymbol w^T(\Sigma_0+\Sigma_1)\boldsymbol w} \\
	&= \cfrac{[(\mu_0-\mu_1)^T\boldsymbol w]^T(\mu_0-\mu_1)^T\boldsymbol w}{\boldsymbol w^T(\Sigma_0+\Sigma_1)\boldsymbol w} \\
	&= \cfrac{\boldsymbol w^T(\mu_0-\mu_1)(\mu_0-\mu_1)^T\boldsymbol w}{\boldsymbol w^T(\Sigma_0+\Sigma_1)\boldsymbol w}
\end{aligned}$$

## 3.37

$$\boldsymbol S_b\boldsymbol w=\lambda\boldsymbol S_w\boldsymbol w$$
[推导]：由3.36可列拉格朗日函数：$l(\boldsymbol w)=-\boldsymbol w^T\boldsymbol S_b\boldsymbol w+\lambda(\boldsymbol w^T\boldsymbol S_w\boldsymbol w-1)$

对$\boldsymbol w$求偏导可得：
$$
\begin{aligned}
\cfrac{\partial l(\boldsymbol w)}{\partial \boldsymbol w} &= -\cfrac{\partial(\boldsymbol w^T\boldsymbol S_b\boldsymbol w)}{\partial \boldsymbol w}+\lambda \cfrac{\partial(\boldsymbol w^T\boldsymbol S_w\boldsymbol w-1)}{\partial \boldsymbol w} \\
	&= -(\boldsymbol S_b+\boldsymbol S_b^T)\boldsymbol w+\lambda(\boldsymbol S_w+\boldsymbol S_w^T)\boldsymbol w
\end{aligned}
$$
又$\boldsymbol S_b=\boldsymbol S_b^T,\boldsymbol S_w=\boldsymbol S_w^T$，则：
$$
\cfrac{\partial l(\boldsymbol w)}{\partial \boldsymbol w} = -2\boldsymbol S_b\boldsymbol w+2\lambda\boldsymbol S_w\boldsymbol w
$$
令导函数等于0即可得式3.37。

## 3.43

$$
\begin{aligned}
\boldsymbol S_b &= \boldsymbol S_t - \boldsymbol S_w \\
&= \sum_{i=1}^N m_i(\boldsymbol\mu_i-\boldsymbol\mu)(\boldsymbol\mu_i-\boldsymbol\mu)^T
\end{aligned}
$$
[推导]：由式3.40、3.41、3.42可得：
$$
\begin{aligned}
\boldsymbol S_b &= \boldsymbol S_t - \boldsymbol S_w \\
&= \sum_{i=1}^m(\boldsymbol x_i-\boldsymbol\mu)(\boldsymbol x_i-\boldsymbol\mu)^T-\sum_{i=1}^N\sum_{\boldsymbol x\in X_i}(\boldsymbol x-\boldsymbol\mu_i)(\boldsymbol x-\boldsymbol\mu_i)^T \\
&= \sum_{i=1}^N\left(\sum_{\boldsymbol x\in X_i}\left((\boldsymbol x-\boldsymbol\mu)(\boldsymbol x-\boldsymbol\mu)^T-(\boldsymbol x-\boldsymbol\mu_i)(\boldsymbol x-\boldsymbol\mu_i)^T\right)\right) \\
&= \sum_{i=1}^N\left(\sum_{\boldsymbol x\in X_i}\left((\boldsymbol x-\boldsymbol\mu)(\boldsymbol x^T-\boldsymbol\mu^T)-(\boldsymbol x-\boldsymbol\mu_i)(\boldsymbol x^T-\boldsymbol\mu_i^T)\right)\right) \\
&= \sum_{i=1}^N\left(\sum_{\boldsymbol x\in X_i}\left(\boldsymbol x\boldsymbol x^T - \boldsymbol x\boldsymbol\mu^T-\boldsymbol\mu\boldsymbol x^T+\boldsymbol\mu\boldsymbol\mu^T-\boldsymbol x\boldsymbol x^T+\boldsymbol x\boldsymbol\mu_i^T+\boldsymbol\mu_i\boldsymbol x^T-\boldsymbol\mu_i\boldsymbol\mu_i^T\right)\right) \\
&= \sum_{i=1}^N\left(\sum_{\boldsymbol x\in X_i}\left(- \boldsymbol x\boldsymbol\mu^T-\boldsymbol\mu\boldsymbol x^T+\boldsymbol\mu\boldsymbol\mu^T+\boldsymbol x\boldsymbol\mu_i^T+\boldsymbol\mu_i\boldsymbol x^T-\boldsymbol\mu_i\boldsymbol\mu_i^T\right)\right) \\
&= \sum_{i=1}^N\left(-\sum_{\boldsymbol x\in X_i}\boldsymbol x\boldsymbol\mu^T-\sum_{\boldsymbol x\in X_i}\boldsymbol\mu\boldsymbol x^T+\sum_{\boldsymbol x\in X_i}\boldsymbol\mu\boldsymbol\mu^T+\sum_{\boldsymbol x\in X_i}\boldsymbol x\boldsymbol\mu_i^T+\sum_{\boldsymbol x\in X_i}\boldsymbol\mu_i\boldsymbol x^T-\sum_{\boldsymbol x\in X_i}\boldsymbol\mu_i\boldsymbol\mu_i^T\right) \\
&= \sum_{i=1}^N\left(-m_i\boldsymbol\mu_i\boldsymbol\mu^T-m_i\boldsymbol\mu\boldsymbol\mu_i^T+m_i\boldsymbol\mu\boldsymbol\mu^T+m_i\boldsymbol\mu_i\boldsymbol\mu_i^T+m_i\boldsymbol\mu_i\boldsymbol\mu_i^T-m_i\boldsymbol\mu_i\boldsymbol\mu_i^T\right) \\
&= \sum_{i=1}^N\left(-m_i\boldsymbol\mu_i\boldsymbol\mu^T-m_i\boldsymbol\mu\boldsymbol\mu_i^T+m_i\boldsymbol\mu\boldsymbol\mu^T+m_i\boldsymbol\mu_i\boldsymbol\mu_i^T\right) \\
&= \sum_{i=1}^Nm_i\left(-\boldsymbol\mu_i\boldsymbol\mu^T-\boldsymbol\mu\boldsymbol\mu_i^T+\boldsymbol\mu\boldsymbol\mu^T+\boldsymbol\mu_i\boldsymbol\mu_i^T\right) \\
&= \sum_{i=1}^N m_i(\boldsymbol\mu_i-\boldsymbol\mu)(\boldsymbol\mu_i-\boldsymbol\mu)^T
\end{aligned}
$$

## 3.44

$$
\max\limits_{\mathbf{W}}\cfrac{
tr(\mathbf{W}^T\boldsymbol S_b \mathbf{W})}{tr(\mathbf{W}^T\boldsymbol S_w \mathbf{W})}
$$

[解析]：此式是式3.35的推广形式，证明如下：
设$\mathbf{W}=[\boldsymbol w_1,\boldsymbol w_2,...,\boldsymbol w_i,...,\boldsymbol w_{N-1}]$，其中$\boldsymbol w_i$为$d$行1列的列向量，则：
$$
\left\{
\begin{aligned}
tr(\mathbf{W}^T\boldsymbol S_b \mathbf{W})&=\sum_{i=1}^{N-1}\boldsymbol w_i^T\boldsymbol S_b \boldsymbol w_i \\
tr(\mathbf{W}^T\boldsymbol S_w \mathbf{W})&=\sum_{i=1}^{N-1}\boldsymbol w_i^T\boldsymbol S_w \boldsymbol w_i
\end{aligned}
\right.
$$
所以式3.44可变形为：
$$
\max\limits_{\mathbf{W}}\cfrac{
\sum_{i=1}^{N-1}\boldsymbol w_i^T\boldsymbol S_b \boldsymbol w_i}{\sum_{i=1}^{N-1}\boldsymbol w_i^T\boldsymbol S_w \boldsymbol w_i}
$$
对比式3.35易知上式即为式3.35的推广形式。
