## 5.2

$$
\Delta w_i=\eta(y-\hat{y})x_i
$$

[解析]：此公式是感知机学习算法中的参数更新公式，下面依次给出感知机模型、学习策略和学习算法的具体介绍<sup>[1]</sup>：

### 感知机模型

已知感知机由两层神经元组成，故感知机模型的公式可表示为
$$
y=f(\sum\limits_{i=1}^{n}w_ix_i-\theta)=f(\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x}-\theta)
$$
其中，$\boldsymbol{x} \in \mathbb{R}^n$为样本的特征向量，是感知机模型的输入；$\boldsymbol{w},\theta$是感知机模型的参数，$\boldsymbol{w} \in \mathbb{R}^n$为权重，$\theta$为阈值。假定$f$为阶跃函数，那么感知机模型的公式可进一步表示为
$$
 y=\operatorname{sgn}(\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x}-\theta)=\left\{\begin{array}{rcl}
1,& {\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x} -\theta\geq 0}\\
0,& {\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x} -\theta < 0}\\
\end{array} \right.
$$
由于$n$维空间中的超平面方程为
$$
w_1x_1+w_2x_2+\cdots+w_nx_n+b  =\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x} +b=0
$$
所以此时感知机模型公式中的$\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x}-\theta$可以看作是$n$维空间中的一个超平面，通过它将$n$维空间划分为$\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x}-\theta\geq 0$和$\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x}-\theta<0$两个子空间，落在前一个子空间的样本对应的模型输出值为1，落在后一个子空间的样本对应的模型输出值为0，以此来实现分类功能。

### 感知机学习策略

给定一个线性可分的数据集$T$（参见附录①），感知机的学习目标是求得能对数据集$T$中的正负样本完全正确划分的分离超平面：
$$
\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x}-\theta=0
$$
假设此时误分类样本集合为$M\subseteq T$，对任意一个误分类样本$(\boldsymbol{x},y)\in M$来说，当$\boldsymbol{w}^\mathrm{T}\boldsymbol{x}-\theta \geq 0$时，模型输出值为$\hat{y}=1$，样本真实标记为$y=0$；反之，当$\boldsymbol{w}^\mathrm{T}\boldsymbol{x}-\theta<0$时，模型输出值为$\hat{y}=0$，样本真实标记为$y=1$。综合两种情形可知，以下公式恒成立
$$
(\hat{y}-y)(\boldsymbol{w}^\mathrm{T}\boldsymbol{x}-\theta)\geq0
$$
所以，给定数据集$T$，其损失函数可以定义为：
$$
L(\boldsymbol{w},\theta)=\sum_{\boldsymbol{x}\in M}(\hat{y}-y)(\boldsymbol{w}^\mathrm{T}\boldsymbol{x}-\theta)
$$
显然，此损失函数是非负的。如果没有误分类点，损失函数值是0。而且，误分类点越少，误分类点离超平面越近，损失函数值就越小。因此，给定数据集$T$，损失函数$L(\boldsymbol{w},\theta)$是关于$\boldsymbol{w},\theta$的连续可导函数。

### 感知机学习算法

感知机模型的学习问题可以转化为求解损失函数的最优化问题，具体地，给定数据集
$$
T=\{(\boldsymbol{x}_1,y_1),(\boldsymbol{x}_2,y_2),\dots,(\boldsymbol{x}_N,y_N)\}
$$
其中$\boldsymbol{x}_i \in \mathbb{R}^n,y_i \in \{0,1\}$，求参数$\boldsymbol{w},\theta$，使其为极小化损失函数的解：
$$
\min\limits_{\boldsymbol{w},\theta}L(\boldsymbol{w},\theta)=\min\limits_{\boldsymbol{w},\theta}\sum_{\boldsymbol{x_i}\in M}(\hat{y}_i-y_i)(\boldsymbol{w}^\mathrm{T}\boldsymbol{x}_i-\theta)
$$
其中$M\subseteq T$为误分类样本集合。若将阈值$\theta$看作一个固定输入为$-1$的“哑节点”，即
$$
-\theta=-1\cdot w_{n+1}=x_{n+1}\cdot w_{n+1}
$$
那么$\boldsymbol{w}^\mathrm{T}\boldsymbol{x}_i-\theta$可化简为
$$
\begin{aligned}
\boldsymbol{w}^\mathrm{T}\boldsymbol{x_i}-\theta&=\sum \limits_{j=1}^n w_jx_j+x_{n+1}\cdot w_{n+1}\\
&=\sum \limits_{j=1}^{n+1}w_jx_j\\
&=\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x_i}
 \end{aligned}
$$
其中$\boldsymbol{x_i} \in \mathbb{R}^{n+1},\boldsymbol{w} \in \mathbb{R}^{n+1}$。根据该式，可将要求解的极小化问题进一步简化为
$$
\min\limits_{\boldsymbol{w}}L(\boldsymbol{w})=\min\limits_{\boldsymbol{w}}\sum_{\boldsymbol{x_i}\in M}(\hat{y}_i-y_i)\boldsymbol{w}^\mathrm{T}\boldsymbol{x_i}
$$
假设误分类样本集合$M$固定，那么可以求得损失函数$L(\boldsymbol{w})$的梯度为：
$$
\nabla_{\boldsymbol{w}}L(\boldsymbol{w})=\sum_{\boldsymbol{x_i}\in M}(\hat{y}_i-y_i)\boldsymbol{x_i}
$$
感知机的学习算法具体采用的是随机梯度下降法，也就是极小化过程中不是一次使$M$中所有误分类点的梯度下降，而是一次随机选取一个误分类点使其梯度下降。所以权重$\boldsymbol{w}$的更新公式为
$$
\boldsymbol w \leftarrow \boldsymbol w+\Delta \boldsymbol w
$$

$$
\Delta \boldsymbol w=-\eta(\hat{y}_i-y_i)\boldsymbol x_i=\eta(y_i-\hat{y}_i)\boldsymbol x_i
$$

相应地，$\boldsymbol{w}$中的某个分量$w_i$的更新公式即为公式(5.2)。

## 5.10

$$
\begin{aligned}
g_j&=-\frac{\partial {E_k}}{\partial{\hat{y}_j^k}} \cdot \frac{\partial{\hat{y}_j^k}}{\partial{\beta_j}}
\\&=-( \hat{y}_j^k-y_j^k ) f ^{\prime} (\beta_j-\theta_j)
\\&=\hat{y}_j^k(1-\hat{y}_j^k)(y_j^k-\hat{y}_j^k)
\end{aligned}
$$

[推导]：参见公式(5.12)

## 5.12

$$
\Delta \theta_j = -\eta g_j
$$

[推导]：因为
$$
\Delta \theta_j = -\eta \cfrac{\partial E_k}{\partial \theta_j}
$$
又
$$
\begin{aligned}	
\cfrac{\partial E_k}{\partial \theta_j} &= \cfrac{\partial E_k}{\partial \hat{y}_j^k} \cdot\cfrac{\partial \hat{y}_j^k}{\partial \theta_j} \\
&= \cfrac{\partial E_k}{\partial \hat{y}_j^k} \cdot\cfrac{\partial [f(\beta_j-\theta_j)]}{\partial \theta_j} \\
&=\cfrac{\partial E_k}{\partial \hat{y}_j^k} \cdot f^{\prime}(\beta_j-\theta_j) \times (-1) \\
&=\cfrac{\partial E_k}{\partial \hat{y}_j^k} \cdot f\left(\beta_{j}-\theta_{j}\right)\times\left[1-f\left(\beta_{j}-\theta_{j}\right)\right]  \times (-1) \\
&=\cfrac{\partial E_k}{\partial \hat{y}_j^k} \cdot \hat{y}_j^k\left(1-\hat{y}_j^k\right)  \times (-1) \\
&=\cfrac{\partial\left[ \cfrac{1}{2} \sum\limits_{j=1}^{l}\left(\hat{y}_{j}^{k}-y_{j}^{k}\right)^{2}\right]}{\partial \hat{y}_{j}^{k}} \cdot \hat{y}_j^k\left(1-\hat{y}_j^k\right) \times (-1)  \\
&=\cfrac{1}{2}\times 2(\hat{y}_j^k-y_j^k)\times 1 \cdot\hat{y}_j^k\left(1-\hat{y}_j^k\right)  \times (-1) \\
&=(y_j^k-\hat{y}_j^k)\hat{y}_j^k\left(1-\hat{y}_j^k\right)  \\
&= g_j
\end{aligned}
$$
所以
$$
\Delta \theta_j = -\eta \cfrac{\partial E_k}{\partial \theta_j}=-\eta g_j
$$

## 5.13

$$
\Delta v_{ih} = \eta e_h x_i
$$

[推导]：因为
$$
\Delta v_{ih} = -\eta \cfrac{\partial E_k}{\partial v_{ih}}
$$
又
$$
\begin{aligned}	
\cfrac{\partial E_k}{\partial v_{ih}} &= \sum_{j=1}^{l} \cfrac{\partial E_k}{\partial \hat{y}_j^k} \cdot \cfrac{\partial \hat{y}_j^k}{\partial \beta_j} \cdot \cfrac{\partial \beta_j}{\partial b_h} \cdot \cfrac{\partial b_h}{\partial \alpha_h} \cdot \cfrac{\partial \alpha_h}{\partial v_{ih}} \\
&= \sum_{j=1}^{l} \cfrac{\partial E_k}{\partial \hat{y}_j^k} \cdot \cfrac{\partial \hat{y}_j^k}{\partial \beta_j} \cdot \cfrac{\partial \beta_j}{\partial b_h} \cdot \cfrac{\partial b_h}{\partial \alpha_h} \cdot x_i \\ 
&= \sum_{j=1}^{l} \cfrac{\partial E_k}{\partial \hat{y}_j^k} \cdot \cfrac{\partial \hat{y}_j^k}{\partial \beta_j} \cdot \cfrac{\partial \beta_j}{\partial b_h} \cdot f^{\prime}(\alpha_h-\gamma_h) \cdot x_i \\
&= \sum_{j=1}^{l} \cfrac{\partial E_k}{\partial \hat{y}_j^k} \cdot \cfrac{\partial \hat{y}_j^k}{\partial \beta_j} \cdot w_{hj} \cdot f^{\prime}(\alpha_h-\gamma_h) \cdot x_i \\
&= \sum_{j=1}^{l} (-g_j) \cdot w_{hj} \cdot f^{\prime}(\alpha_h-\gamma_h) \cdot x_i \\
&= -f^{\prime}(\alpha_h-\gamma_h) \cdot \sum_{j=1}^{l} g_j \cdot w_{hj}  \cdot x_i\\
&= -b_h(1-b_h) \cdot \sum_{j=1}^{l} g_j \cdot w_{hj}  \cdot x_i \\
&= -e_h \cdot x_i
\end{aligned}
$$
所以
$$
\Delta v_{ih} =-\eta \cfrac{\partial E_k}{\partial v_{ih}} =\eta e_h x_i
$$

## 5.14

$$
\Delta \gamma_h= -\eta e_h
$$

[推导]：因为
$$
\Delta \gamma_h = -\eta \cfrac{\partial E_k}{\partial \gamma_h}
$$
又
$$
\begin{aligned}	
\cfrac{\partial E_k}{\partial \gamma_h} &= \sum_{j=1}^{l} \cfrac{\partial E_k}{\partial \hat{y}_j^k} \cdot \cfrac{\partial \hat{y}_j^k}{\partial \beta_j} \cdot \cfrac{\partial \beta_j}{\partial b_h} \cdot \cfrac{\partial b_h}{\partial \gamma_h} \\
&= \sum_{j=1}^{l} \cfrac{\partial E_k}{\partial \hat{y}_j^k} \cdot \cfrac{\partial \hat{y}_j^k}{\partial \beta_j} \cdot \cfrac{\partial \beta_j}{\partial b_h} \cdot f^{\prime}(\alpha_h-\gamma_h) \cdot (-1) \\
&= -\sum_{j=1}^{l} \cfrac{\partial E_k}{\partial \hat{y}_j^k} \cdot \cfrac{\partial \hat{y}_j^k}{\partial \beta_j} \cdot w_{hj} \cdot f^{\prime}(\alpha_h-\gamma_h)\\
&= -\sum_{j=1}^{l} \cfrac{\partial E_k}{\partial \hat{y}_j^k} \cdot \cfrac{\partial \hat{y}_j^k}{\partial \beta_j} \cdot w_{hj} \cdot b_h(1-b_h)\\
&= \sum_{j=1}^{l}g_j\cdot w_{hj} \cdot b_h(1-b_h)\\
&=e_h
\end{aligned}
$$
所以
$$
\Delta \gamma_h=-\eta\cfrac{\partial E_k}{\partial \gamma_h} = -\eta e_h
$$

## 5.15

$$
\begin{aligned}
e_h&=-\frac{\partial {E_k}}{\partial{b_h}}\cdot \frac{\partial{b_h}}{\partial{\alpha_h}}
\\&=-\sum_{j=1}^l \frac{\partial {E_k}}{\partial{\beta_j}}\cdot \frac{\partial{\beta_j}}{\partial{b_h}}f^{\prime}(\alpha_h-\gamma_h)
\\&=\sum_{j=1}^l w_{hj}g_j f^{\prime}(\alpha_h-\gamma_h)
\\&=b_h(1-b_h)\sum_{j=1}^l w_{hj}g_j 
\end{aligned}
$$

[推导]：参见公式(5.13)

## 5.20 - Boltzmann Machine

![Boltzmann machine - Wikipedia](media\Chapter5_神经网络\Boltzmannexamplev1.png)

$$
E(\boldsymbol{s})=-\sum_{i=1}^{n-1}\sum_{j=i+1}^{n}w_{ij}s_is_j-\sum_{p=1}^n\theta_is_i
$$

[解析]：能量最初表示一个物理概念，用于描述系统某状态下的能量值。==能量值越大，当前状态越不稳定，当能量值达到最小时系统达到稳定状态。==Boltzmann机本质上是一个引入了隐变量的无向图模型，无向图的能量可理解为
$$
E_{graph}=E_{edges}+E_{nodes}
$$
其中，$E_{graph}$表示图的能量，$E_{edges}$表示图中边的能量，$E_{nodes}$表示图中结点的能量；边能量由两连接结点的值及其权重的乘积确定：$E_{{edge}_{ij}}=-w_{ij}s_is_j$，结点能量由结点的值及其阈值的乘积确定：$E_{{node}_i}=-\theta_is_i$ (bias term)；图中边的能量为图中所有边能量之和
$$
E_{edges}=\sum_{i=1}^{n-1}\sum_{j=i+1}^{n}E_{{edge}_{ij}}=-\sum_{i=1}^{n-1}\sum_{j=i+1}^{n}w_{ij}s_is_j
$$
图中结点的能量为图中所有结点能量之和
$$
E_{nodes}=\sum_{p=1}^nE_{{node}_i}=-\sum_{p=1}^n\theta_is_i
$$
故状态向量$\boldsymbol{s}$所对应的Boltzmann机能量为
$$
E_{graph}=E_{edges}+E_{nodes}=-\sum_{i=1}^{n-1}\sum_{j=i+1}^{n}w_{ij}s_is_j-\sum_{p=1}^n\theta_is_i
$$

因为这里面所有的值都是大于等于零的，所以他们的product和肯定也是。所以要让graph的Energy E(s)尽量低，就是要让-E(s)尽量大，也就是说$e^{-E(s)}$的取值一定是大于等于1的，且越大越好。而这个就是为什么P(s)可以写成5.21形式的原因。

## 5.21

$$
P(\boldsymbol{s})=\frac{e^{-E(\boldsymbol{s})}}{\sum_{\boldsymbol{t}}e^{-E(\boldsymbol{t})}}
$$

其实就是一个网络状态的出现概率。（softmax）

[推导]：一个无向图网络，其联合概率分布表示为：
$$
P(\boldsymbol{s})=\frac{1}{Z}\prod_{i=1}^{k}\Phi_i(\boldsymbol{s}_{c_i})
$$
其中，$k$为无向图网络中的极大团个数；$c_i$表示极大团的节点集合；$x_{c_i}$为该极大团所对用的节点变量；$\Phi_i$为势函数；$Z$表示规范化因子（极大团、势函数和规范化因子的具体定义参见西瓜书第14.2节）。假设一个Boltzmann机含有$n$个节点，$\boldsymbol{s}=\{0,1\}^n$为当前状态，状态集合$T$表示$2^n$种所有可能的状态构成的集合。由于Boltzmann机是一个全连接网络，故Boltzmann机中的极大团仅有一个，其节点集合为$c=\{s_1,s_2,\cdots,s_n\}$。其联合概率分布为
$$
P(\boldsymbol{s})=\frac{1}{Z}\Phi(\boldsymbol{s}_{c})
$$
势函数$\Phi(\boldsymbol{s}_{c})$一般定义为指数型函数，所以$\Phi(\boldsymbol{s}_{c})$的一般形式为
$$
\Phi(\boldsymbol{s}_{c})=e^{-E(\boldsymbol{s}_{c})}
$$
其中$\boldsymbol{s}_c=(s_1\,s_2\,\cdots,\, s_n)=\boldsymbol{s}$，则状态$\boldsymbol{s}$下的联合概率分布为
$$
P(\boldsymbol{s})=\frac{1}{Z}e^{-E(\boldsymbol{s})}
$$
状态集合$T$中的某个状态$\boldsymbol{s}$出现的概率定义为：状态$\boldsymbol{s}$的联合概率分布与所有可能的状态的联合概率分布的比值
$$
P(\boldsymbol{s})=\frac{e^{-E(\boldsymbol{s})}}{\sum_{\boldsymbol{t}\in T}e^{-E(\boldsymbol{t})}}
$$

## 5.22 - Restricted BM

![image-20200827003700812](media\Chapter5_神经网络\image-20200827003700812.png)


$$
P(\boldsymbol{v}|\boldsymbol{h})=\prod_{i=1}^dP(v_i\,  |  \, \boldsymbol{h})
$$

[解析]：==受限Boltzmann机仅保留显层与隐层之间的连接==，显层的状态向量为$\boldsymbol{v}$，隐层的状态向量为$\boldsymbol{h}$。
$$
\boldsymbol{v}=\left[\begin{array}{c}v_1\\ v_2\\ \vdots\\ v_d\\\end{array} \right]\qquad
\boldsymbol{h}=\left[\begin{array}{c}h_1\\h_2\\ \vdots\\ h_q\end{array} \right]
$$
对于显层状态向量$\boldsymbol{v}$中的变量$v_i$，既第i个node上的状态，出现的概率仅与隐层状态向量$\boldsymbol{h}$有关，所以给定隐层状态向量$\boldsymbol{h}$，$v_1,v_2,...,v_d$相互独立。

## 5.23

$$
P(\boldsymbol{h}|\boldsymbol{v})=\prod_{j=1}^qP(h_i\,  |  \, \boldsymbol{v})
$$

[解析]：由公式5.22的解析同理可得：给定显层状态向量$\boldsymbol{v}$，$h_1,h_2,\cdots,h_q$相互独立。

## 5.24

$$
\Delta w=\eta(\boldsymbol{v}\boldsymbol{h}^\mathrm{T}-\boldsymbol{v}’\boldsymbol{h}’^{\mathrm{T}})
$$

[推导]：由公式(5.20)可推导出受限Boltzmann机（以下简称RBM）的能量函数为：
$$
\begin{aligned}
E(\boldsymbol{v},\boldsymbol{h})&=-\sum_{i=1}^d\sum_{j=1}^qw_{ij}v_ih_j-\sum_{i=1}^d\alpha_iv_i-\sum_{j=1}^q\beta_jh_j \\
&=-\boldsymbol{h}^{\mathrm{T}}\mathbf{W}\boldsymbol{v}-\boldsymbol{\alpha}^{\mathrm{T}}\boldsymbol{v}-\boldsymbol{\beta}^{\mathrm{T}}\boldsymbol{h}
\end{aligned}
$$
其中
$$
\mathbf{W}=\begin{bmatrix}
\boldsymbol{w}_1\\
\boldsymbol{w}_2\\ 
\vdots\\
\boldsymbol{w}_q
\end{bmatrix}\in \mathbb{R}^{q*d}
$$
q是hidden的dimension，d是visible的dim.。再由公式(5.21)可知，RBM的联合概率分布为
$$
P(\boldsymbol{v},\boldsymbol{h})=\frac{1}{Z}e^{-E(\boldsymbol{v},\boldsymbol{h})}
$$
其中$Z$为规范化因子，这是所有v~i~和h~j~的可能的取值的组合发生的概率的sum。比如$\boldsymbol{h}\in [{0,1}]$然后如果q=10，那它就有2^10^种组合，我们要把每种组合出现的概率$e^{-E(\boldsymbol{v},\boldsymbol{h})}$都加起来。
$$
Z=\sum_{\boldsymbol{v}}\sum_{\boldsymbol{h}}e^{-E(\boldsymbol{v},\boldsymbol{h})}
$$
给定含$m$个独立同分布数据的数据集$V=\{\boldsymbol{v}_1,\boldsymbol{v}_2,\cdots,\boldsymbol{v}_m\}$，记$\boldsymbol{\theta}=\{\mathbf{W},\boldsymbol{\alpha},\boldsymbol{\beta}\}$，学习RBM的策略是求出参数$\boldsymbol{\theta}$的值，使得如下对数似然函数最大化（注意每个示例$\boldsymbol{v}_k$本身的dimension就是d）
$$
\begin{aligned}
LL(\boldsymbol{\theta})&=\ln\left(\prod_{k=1}^{m}P(\boldsymbol{v}_k)\right) \\
&=\sum_{k=1}^m\ln P(\boldsymbol{v}_k) \\
&= \sum_{k=1}^m L_k(\boldsymbol{\theta})
\end{aligned}
$$
具体采用的是梯度上升法来求解参数$\boldsymbol{\theta}$，因此，下面来考虑求对数似然函数$L(\boldsymbol{\theta})$的梯度。对于$V$中的任意一个样本$\boldsymbol{v}_k$来说，其$L_k(\boldsymbol{\theta})$的具体形式为
$$
\begin{aligned}
LL_k(\boldsymbol{\theta})&=\ln P(\boldsymbol{v}_k)
\\&=\ln\left(\sum_{\boldsymbol{h}}P(\boldsymbol{v}_k,\boldsymbol{h})\right)
\\&=\ln\left(\sum_{\boldsymbol{h}}\frac{1}{Z}e^{-E(\boldsymbol{v}_k,\boldsymbol{h})}\right)
\\&=\ln\left(\sum_{\boldsymbol{h}}e^{-E(\boldsymbol{v}_k,\boldsymbol{h})}\right)-\ln Z
\\&=\ln\left(\sum_{\boldsymbol{h}}e^{-E(\boldsymbol{v}_k,\boldsymbol{h})}\right)-\ln\left(\sum_{\boldsymbol{v},\boldsymbol{h}}e^{-E({\boldsymbol{v},\boldsymbol{h})}}\right)
\end{aligned}
$$
对$L_k(\boldsymbol{\theta})$进行求导
$$
\begin{aligned}
\frac{\partial{LL_k(\boldsymbol{\theta})}}{\partial{\boldsymbol{\theta}}}&=\frac{\partial}{\partial{\boldsymbol{\theta}}}\left[\ln\sum_{\boldsymbol{h}}e^{-E(\boldsymbol{v}_k,\boldsymbol{h})}\right]-\frac{\partial}{\partial{\boldsymbol{\theta}}}\left[\ln\sum_{\boldsymbol{v},\boldsymbol{h}}e^{-E({\boldsymbol{v},\boldsymbol{h})}}\right]
\\&=-\frac{\sum_{\boldsymbol{h}}e^{-E(\boldsymbol{v}_k,\boldsymbol{h})}\frac{\partial{E({\boldsymbol{v}_k,\boldsymbol{h})}}}{\partial{\boldsymbol{\theta}}}}{\sum_{\boldsymbol{h}}e^{-E(\boldsymbol{v}_k,\boldsymbol{h})}}+\frac{\sum_{\boldsymbol{v},\boldsymbol{h}}e^{-E(\boldsymbol{v},\boldsymbol{h})}\frac{\partial{E({\boldsymbol{v},\boldsymbol{h})}}}{\partial{\boldsymbol{\theta}}}}{\sum_{\boldsymbol{v},\boldsymbol{h}}e^{-E(\boldsymbol{v},\boldsymbol{h})}}
\\&=-\sum_{\boldsymbol{h}}\frac{e^{-E(\boldsymbol{v}_k,\boldsymbol{h})}\frac{\partial{E({\boldsymbol{v}_k,\boldsymbol{h})}}}{\partial{\boldsymbol{\theta}}}}{\sum_{\boldsymbol{h}}e^{-E(\boldsymbol{v}_k,\boldsymbol{h})}}+\sum_{\boldsymbol{v},\boldsymbol{h}}\frac{e^{-E(\boldsymbol{v},\boldsymbol{h})}\frac{\partial{E({\boldsymbol{v},\boldsymbol{h})}}}{\partial{\boldsymbol{\theta}}}}{\sum_{\boldsymbol{v},\boldsymbol{h}}e^{-E(\boldsymbol{v},\boldsymbol{h})}}
\end{aligned}
$$
上式第三行之所以能把sum_h从分子拉出来，也是由于分母本身有个sum_h，所以不影响结果。下式一的最右边的等号是利用了marginal probability的特性（$\sum_{h} P\left(\boldsymbol{v}_{k}, \boldsymbol{h}\right)=P(\boldsymbol{v}_{k})$）和Bayes定理。下式2的最右边的等号是因为$\sum_{v, h} P(\boldsymbol{v}, \boldsymbol{h})=1$。
$$
\frac{e^{-E({\boldsymbol{v}_k,\boldsymbol{h})}}}{\sum_{\boldsymbol{h}}e^{-E({\boldsymbol{v}_k,\boldsymbol{h})}}}=\frac{\frac{e^{-E({\boldsymbol{v}_k,\boldsymbol{h})}}}{Z}}{\frac{\sum_{\boldsymbol{h}}e^{-E({\boldsymbol{v}_k,\boldsymbol{h})}}}{Z}}=\frac{\frac{e^{-E({\boldsymbol{v}_k,\boldsymbol{h})}}}{Z}}{\sum_{\boldsymbol{h}}\frac{e^{-E({\boldsymbol{v}_k,\boldsymbol{h})}}}{Z}}=\frac{P(\boldsymbol{v}_k,\boldsymbol{h})}{\sum_{\boldsymbol{h}}P(\boldsymbol{v}_k,\boldsymbol{h})}=P(\boldsymbol{h}|\boldsymbol{v}_k)
$$

$$
\frac{e^{-E({\boldsymbol{v},\boldsymbol{h})}}}{\sum_{\boldsymbol{v},\boldsymbol{h}}e^{-E({\boldsymbol{v},\boldsymbol{h})}}}=\frac{\frac{e^{-E({\boldsymbol{v},\boldsymbol{h})}}}{Z}}{\frac{\sum_{\boldsymbol{v},\boldsymbol{h}}e^{-E({\boldsymbol{v},\boldsymbol{h})}}}{Z}}=\frac{\frac{e^{-E({\boldsymbol{v},\boldsymbol{h})}}}{Z}}{\sum_{\boldsymbol{v},\boldsymbol{h}}\frac{e^{-E({\boldsymbol{v},\boldsymbol{h})}}}{Z}}=\frac{P(\boldsymbol{v},\boldsymbol{h})}{\sum_{\boldsymbol{v},\boldsymbol{h}}P(\boldsymbol{v},\boldsymbol{h})}=P(\boldsymbol{v},\boldsymbol{h})
$$

故

$$
\begin{aligned}
\frac{\partial{LL_k(\boldsymbol{\theta})}}{\partial{\boldsymbol{\theta}}}&=-\sum_{\boldsymbol{h}}P(\boldsymbol{h}|\boldsymbol{v}_k)\frac{\partial{E({\boldsymbol{v}_k,\boldsymbol{h})}}}{\partial{\boldsymbol{\theta}}}+\sum_{\boldsymbol{v},\boldsymbol{h}}P(\boldsymbol{v},\boldsymbol{h})\frac{\partial{E({\boldsymbol{v},\boldsymbol{h})}}}{\partial{\boldsymbol{\theta}}}
\\&=-\sum_{\boldsymbol{h}}P(\boldsymbol{h}|\boldsymbol{v}_k)\frac{\partial{E({\boldsymbol{v}_k,\boldsymbol{h})}}}{\partial{\boldsymbol{\theta}}}+\sum_{\boldsymbol{v}}\sum_{\boldsymbol{h}}P(\boldsymbol{v})P(\boldsymbol{h}|\boldsymbol{v})\frac{\partial{E({\boldsymbol{v},\boldsymbol{h})}}}{\partial{\boldsymbol{\theta}}}
\\&=-\sum_{\boldsymbol{h}}P(\boldsymbol{h}|\boldsymbol{v}_k)\frac{\partial{E({\boldsymbol{v}_k,\boldsymbol{h})}}}{\partial{\boldsymbol{\theta}}}+\sum_{\boldsymbol{v}}P(\boldsymbol{v})\sum_{\boldsymbol{h}}P(\boldsymbol{h}|\boldsymbol{v})\frac{\partial{E({\boldsymbol{v},\boldsymbol{h})}}}{\partial{\boldsymbol{\theta}}}
\end{aligned}
$$

由于$\boldsymbol{\theta}=\{\mathbf{W},\boldsymbol{\alpha},\boldsymbol{\beta}\}$包含三个参数，在这里我们仅以$\mathbf{W}$中的任意一个分量$w_{ij}$为例进行详细推导。首先将上式中的$\boldsymbol{\theta}$替换为$w_{ij}$可得

$$
\frac{\partial{LL_k(\boldsymbol{\theta})}}{\partial{w_{ij}}}=-\sum_{\boldsymbol{h}}P(\boldsymbol{h}|\boldsymbol{v}_k)\frac{\partial{E({\boldsymbol{v}_k,\boldsymbol{h})}}}{\partial{w_{ij}}}+\sum_{\boldsymbol{v}}P(\boldsymbol{v})\sum_{\boldsymbol{h}}P(\boldsymbol{h}|\boldsymbol{v})\frac{\partial{E({\boldsymbol{v},\boldsymbol{h})}}}{\partial{w_{ij}}}
$$

根据公式(5.23)和(5.24)可知
$$
\begin{aligned}
&\sum_{\boldsymbol{h}}P(\boldsymbol{h}|\boldsymbol{v})\frac{\partial{E({\boldsymbol{v},\boldsymbol{h})}}}{\partial{w_{ij}}}\\
=&-\sum_{\boldsymbol{h}}P(\boldsymbol{h}|\boldsymbol{v})h_iv_j\\
=&-\sum_{\boldsymbol{h}}\prod_{l=1}^{q}P(h_l|\boldsymbol{v})h_iv_j\\
=&-\sum_{\boldsymbol{h}}P(h_i|\boldsymbol{v})\prod_{l=1,l\neq i}^{q}P(h_l|\boldsymbol{v})h_iv_j\\
=&-\sum_{\boldsymbol{h}}P(h_i|\boldsymbol{v})P(h_1,...,h_{i-1},h_{i+1},...,h_q|\boldsymbol{v})h_iv_j\\
=&-\sum_{h_i}P(h_i|\boldsymbol{v})h_iv_j\sum_{h_1,...,h_{i-1},h_{i+1},...,h_q}P(h_1,...,h_{i-1},h_{i+1},...,h_q|\boldsymbol{v})\\
=&-\sum_{h_i}P(h_i|\boldsymbol{v})h_iv_j\cdot 1\\
=&-\left[P(h_i=0|\boldsymbol{v})\cdot0\cdot v_j+P(h_i=1|\boldsymbol{v})\cdot 1\cdot v_j\right]\\
=&-P(h_i=1|\boldsymbol{v})v_j
\end{aligned}
$$

第3个等号是直接把P(h|v)展开了。第5个等号之所以h~i~v~j~可以被提到前面，是因为后面的h~i~只和第i个node的取值有关而v~j~则是固定的。我们这样做能把我们关心的第i个node的表达和剩下的hidden node的表达区分开。第6个等号，不考虑节点i的剩下所有h的可能取值的联合概率和就应该是1。然后由于hi的取值是{0,1}，可得倒数第2个等号。

同理可推得
$$
\sum_{\boldsymbol{h}}P(\boldsymbol{h}|\boldsymbol{v}_k)\frac{\partial{E({\boldsymbol{v}_k,\boldsymbol{h})}}}{\partial{w_{ij}}}=-P(h_i=1|\boldsymbol{v}_k)v_j^k
$$

将以上两式代回$\frac{\partial{L_k(\boldsymbol{\theta})}}{\partial{w_{ij}}}$中可得
$$
\frac{\partial{LL_k(\boldsymbol{\theta})}}{\partial{w_{ij}}}=P(h_i=1|\boldsymbol{v}_k){v_{j}^k}-\sum_{\boldsymbol{v}}P(\boldsymbol{v})P(h_i=1|\boldsymbol{v})v_j
$$
直觉上理解，这个式子就是说我们可以把第k个样本v的第j维发生后h~i~发生的概率调高，而把其他所有v发生后h发生的概率调低，这样来最大化似然函数。

观察此式可知，通过枚举所有可能的$\boldsymbol{v}$来计算$\sum_{\boldsymbol{v}}P(\boldsymbol{v})P(h_i=1|\boldsymbol{v})v_j$的复杂度太高，因此可以考虑求其近似值来简化计算。具体地，RBM通常采用的是西瓜书上所说的“对比散度”（Contrastive Divergence，简称CD）算法。CD算法的核心思想<sup>[2]</sup>是：用步长为$s$（通常设为1）的CD算法
$$
CD_s(\boldsymbol{\theta},\boldsymbol{v})=-\sum_{\boldsymbol{h}}P(\boldsymbol{h}|\boldsymbol{v}^{(0)})\frac{\partial{E({\boldsymbol{v}^{(0)},\boldsymbol{h})}}}{\partial{\boldsymbol{\theta}}}+\sum_{\boldsymbol{h}}P(\boldsymbol{h}|\boldsymbol{v}^{(s)})\frac{\partial{E({\boldsymbol{v}^{(s)},\boldsymbol{h})}}}{\partial{\boldsymbol{\theta}}}
$$
近似代替
$$
\frac{\partial{LL_k(\boldsymbol{\theta})}}{\partial{\boldsymbol{\theta}}}=-\sum_{\boldsymbol{h}}P(\boldsymbol{h}|\boldsymbol{v}_k)\frac{\partial{E({\boldsymbol{v}_k,\boldsymbol{h})}}}{\partial{\boldsymbol{\theta}}}+\sum_{\boldsymbol{v}}P(\boldsymbol{v})\sum_{\boldsymbol{h}}P(\boldsymbol{h}|\boldsymbol{v})\frac{\partial{E({\boldsymbol{v},\boldsymbol{h})}}}{\partial{\boldsymbol{\theta}}}
$$
由此可知对于$w_{ij}$来说，就是用
$$
CD_s(w_{ij},\boldsymbol{v})=P(h_i=1|\boldsymbol{v}^{(0)}){v_{j}^{(0)}}-P(h_i=1|\boldsymbol{v}^{(s)})v_j^{(s)}
$$
近似代替
$$
\frac{\partial{LL_k(\boldsymbol{\theta})}}{\partial{w_{ij}}}=P(h_i=1|\boldsymbol{v}_k){v_{j}^k}-\sum_{\boldsymbol{v}}P(\boldsymbol{v})P(h_i=1|\boldsymbol{v})v_j
$$
令$\Delta w_{ij}:=\frac{\partial{L_k(\boldsymbol{\theta})}}{\partial{w_{ij}}}$，$RBM(\boldsymbol\theta)$表示参数为$\boldsymbol{\theta}$的RBM网络，则$CD_s(w_{ij},\boldsymbol{v})$的具体算法可表示为：

- 输入：$s,V=\{\boldsymbol{v}_1,\boldsymbol{v}_2,\cdots,\boldsymbol{v}_m\},RBM(\boldsymbol\theta)$
- 过程：
  1. 初始化：$\Delta w_{ij}=0$
  2. $for\quad \boldsymbol{v}\in V \quad do$
  3. $\quad \boldsymbol{v}^{(0)}:=\boldsymbol{v}$
  4. $\quad for\quad t=0,1,2,...,s-1\quad do$
  5. $\qquad \boldsymbol{h}^{(t)}=h\_given\_v(\boldsymbol{v}^{(t)},RBM(\boldsymbol\theta))$
  6. $\qquad \boldsymbol{v}^{(t+1)}=v\_given\_h(\boldsymbol{h}^{(t)},RBM(\boldsymbol\theta))$
  7. $\quad end$
  8. $\quad for\quad i=1,2,...,q;j=1,2,...,d\quad do$
  9. $\qquad \Delta w_{ij}=\Delta w_{ij}+\left[P(h_i=1|\boldsymbol{v}^{(0)}){v_{j}^{(0)}}-P(h_i=1|\boldsymbol{v}^{(s)})v_j^{(s)}\right]$
  10. $\quad end$
  11. $end$
- 输出：$\Delta w_{ij}$

其中函数$\boldsymbol{h}=h\_given\_v(\boldsymbol{v},RBM(\boldsymbol\theta))$表示在给定$\boldsymbol{v}$的条件下，从$RBM(\boldsymbol\theta)$中采样生成$\boldsymbol{h}$，同理，函数$\boldsymbol{v}=v\_given\_h(\boldsymbol{h},RBM(\boldsymbol\theta))$表示在给定$\boldsymbol{h}$的条件下，从$RBM(\boldsymbol\theta)$中采样生成$\boldsymbol{v}$。就是根据W和输入v先算出P(hi=1|v)，再采样。由于两个函数的算法可以互相类比推得，因此，下面仅给出函数$h\_given\_v(\boldsymbol{v},RBM(\boldsymbol\theta))$的具体算法：

- 输入：$\boldsymbol{v},RBM(\boldsymbol\theta)$
- 过程：
  1. $for \quad i=1,2,...,q \quad do$
  2. $\quad \text{随机生成} 0\leq\alpha_i\leq 1$
  3. $\quad h_{j}=\left\{\begin{array}{ll}1, & \text { if } \alpha_{i}<P(h_i=1|\boldsymbol{v}) \\ 0, & \text { otherwise }\end{array}\right.$
  4. $end \quad for$
- 输出：$\boldsymbol{h}=(h_1,h_2,...,h_q)^{\mathrm{T}}$

中间的0/1采样是伯努利采样，这整个采样循环是gibbs sampling。

综上可知，==公式(5.24)其实就是带有学习率为$\eta$的$\Delta w_{ij}$的一种形式化的表示。==

$$
\Delta w=\eta\left(v h^{\top}-v^{\prime} h^{\prime \top}\right)
$$
理解RBM，还可以看以下ref：

https://www.youtube.com/watch?v=Fkw0_aAtwIw

https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine

https://towardsdatascience.com/deep-learning-meets-physics-restricted-boltzmann-machines-part-i-6df5c4918c15

https://towardsdatascience.com/deep-learning-meets-physics-restricted-boltzmann-machines-part-ii-4b159dce1ffb

## 附录

### ①数据集的线性可分<sup>[1]</sup>

给定一个数据集
$$
T=\{(\boldsymbol{x}_1,y_1),(\boldsymbol{x}_2,y_2),...,(\boldsymbol{x}_N,y_N)\}
$$
其中，$\boldsymbol{x}_i\in \mathbb{R}^n,y_i\in\{0,1\},i=1,2,...,N$，如果存在某个超平面
$$
\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x} +b=0
$$
能将数据集$T$中的正样本和负样本完全正确地划分到超平面两侧，即对所有$y_i=1$的样本$\boldsymbol{x}_i$，有$\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x}_i +b\geq0$，对所有$y_i=0$的样本$\boldsymbol{x}_i$，有$\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x} _i+b<0$，则称数据集$T$线性可分，否则称数据集$T$线性不可分。

### RBM代码实现

#### model structure

```python
class RBM:
    ''' Implementation of the Restricted Boltzmann Machine for collaborative filtering. The model is based on the paper of 
        Ruslan Salakhutdinov, Andriy Mnih and Geoffrey Hinton: https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf
    '''
    def __init__(self, FLAGS):
        '''Initialization of the model  '''
        self.FLAGS=FLAGS
        self.weight_initializer=model_helper._get_weight_init()
        self.bias_initializer=model_helper._get_bias_init()
        self.init_parameter()
        

    def init_parameter(self):
        ''' Initializes the weights and the bias parameters of the neural network.'''

        with tf.variable_scope('Network_parameter'):
            self.W=tf.get_variable('Weights', shape=(self.FLAGS.num_v, self.FLAGS.num_h),initializer=self.weight_initializer)
            self.bh=tf.get_variable('hidden_bias', shape=(self.FLAGS.num_h), initializer=self.bias_initializer)
            self.bv=tf.get_variable('visible_bias', shape=(self.FLAGS.num_v), initializer=self.bias_initializer)
```

#### sampling hidden state

![image-20200827003932089](media\Chapter5_神经网络\image-20200827003932089.png)

```python

 def _sample_h(self, v):
        ''' Uses the visible nodes for calculation of  the probabilities that a hidden neuron is activated. 
        After that Bernouille distribution is used to sample the hidden nodes.
        
        @param v: visible nodes
        @return probability that a hidden neuron is activated
        @return sampled hidden neurons (value 1 or 0 accroding to Bernouille distribution)
        '''
        
        with tf.name_scope('sampling_hidden_units'):

            a=tf.nn.bias_add(tf.matmul(v,self.W), self.bh)
            p_h_v=tf.nn.sigmoid(a)
            h_=self._bernouille_sampling(p_h_v, shape=[self.FLAGS.batch_size, int(p_h_v.shape[-1])])
            
            return p_h_v, h_
          
          
def _bernouille_sampling(self,p, shape):
        '''Samples from the Bernoulli distribution
        
        @param p: probability 
        @return samples from Bernoulli distribution
        
        '''
        return tf.where(tf.less(p, tf.random_uniform(shape,minval=0.0,maxval=1.0)),
                        x=tf.zeros_like(p),
                        y=tf.ones_like(p))
```

### Sampling of the Visible States

![image-20200827003951769](media\Chapter5_神经网络\image-20200827003951769.png)

```python
def _sample_v(self, h):
        ''' Uses the hidden nodes for calculation of  the probabilities that a visible neuron is activated. 
        After that Bernouille distribution is used to sample the visible nodes.
        
        @param h: hidden nodes
        @return probability that a visible neuron is activated
        @return sampled visible neurons (value 1 or 0 accroding to Bernouille distribution)
        '''
        
        with tf.name_scope('sampling_visible_units'):
            a=tf.nn.bias_add(tf.matmul(h,tf.transpose(self.W, [1,0])), self.bv)
            p_v_h=tf.nn.sigmoid(a)
            v_=self._bernouille_sampling(p_v_h, shape=[self.FLAGS.batch_size, int(p_v_h.shape[-1])])
            
            return p_v_h, v_
```

#### Gibbs Sampling

![image-20200827004005662](media\Chapter5_神经网络\image-20200827004005662.png)

```python
def _gibbs_sampling(self, v):
        ''' Performing the Gibbs Sampling.
        
        @param v: visible neurons
        @return visible neurons before gibbs sampling
        @return visible neurons before gibbs sampling
        @return probability that hidden neurons are activated before gibbs sampling.
        @return probability that hidden neurons are activated after gibbs sampling.
        '''
 
        #end condition for the while loop
        def condition(i, vk, hk,v):
            r= tf.less(i,k)
            return r[0]
        
        #loop body
        def body(i, vk, hk,v):
            
            _,hk=self._sample_h(vk)
            _,vk=self._sample_v(hk)

            vk=tf.where(tf.less(v,0),v,vk)
            
            return [i+1, vk, hk,v]
            
        ph0,_=self._sample_h(v)
        
        vk=v
        hk=tf.zeros_like(ph0)
            
        i = 0 # start counter for the while loop
        k=tf.constant([self.FLAGS.k]) # number for the end condition of the while loop
        
        [i, vk,hk,v]=tf.while_loop(condition, body,[i, vk,hk,v])
        
        phk,_=self._sample_h(vk)
        
        return v, vk,ph0, phk, i
```

#### Computing the Gradients

![image-20200827004023127](media\Chapter5_神经网络\image-20200827004023127.png)

```python
def _compute_gradients(self,v0, vk, ph0, phk):
        ''' Computing the gradients of the weights and bias terms with Contrastive Divergence.
        
        @param v0: visible neurons before gibbs sampling
        @param vk: visible neurons after gibbs sampling
        @param ph0: probability that hidden neurons are activated before gibbs sampling.
        @param phk: probability that hidden neurons are activated after gibbs sampling.
        
        @return gradients of the network parameters
        
        '''
        
        #end condition for the while loop
        def condition(i, v0, vk, ph0, phk, dW,db_h,db_v):
            r=tf.less(i,k)
            return r[0]
        
        #loop body
        def body(i, v0, vk, ph0, phk, dW,dbh,dbv):
            
            v0_=v0[i]
            ph0_=ph0[i]
            
            vk_=vk[i]
            phk_=phk[i]       
            
            #reshaping for making the outer product possible
            ph0_=tf.reshape(ph0_, [1,self.FLAGS.num_h])
            v0_=tf.reshape(v0_, [self.FLAGS.num_v,1])
            phk_=tf.reshape(phk_, [1,self.FLAGS.num_h])
            vk_=tf.reshape(vk_, [self.FLAGS.num_v,1])
            
            #calculating the gradients for weights 
            dw_=tf.subtract(tf.multiply(ph0_, v0_),tf.multiply(phk_, vk_))
            #calculating the gradients for hidden bias
            dbh_=tf.subtract(ph0_,phk_)
            #calculating the gradients for visible bias
            dbv_=tf.subtract(v0_,vk_)
            
            dbh_=tf.reshape(dbh_,[self.FLAGS.num_h])
            dbv_=tf.reshape(dbv_,[self.FLAGS.num_v])
            
            # add the computed gradients to previosly computed gradients
            return [i+1, v0, vk, ph0, phk,tf.add(dW,dw_),tf.add(dbh,dbh_),tf.add(dbv,dbv_)]
        
        i = 0 # start counter for the while loop
        k=tf.constant([self.FLAGS.batch_size]) # number for the end condition of the while loop
        
        #init empty placeholders wherer the gradients will be stored              
        dW=tf.zeros((self.FLAGS.num_v, self.FLAGS.num_h))
        dbh=tf.zeros((self.FLAGS.num_h))
        dbv=tf.zeros((self.FLAGS.num_v))
        
        #iterate over the batch and compute for each sample a gradient
        [i, v0, vk, ph0, phk, dW,db_h,db_v]=tf.while_loop(condition, body,[i, v0, vk, ph0, phk, dW,dbh,dbv])
          
        #devide the summed gradients by the batch size
        dW=tf.div(dW, self.FLAGS.batch_size)
        dbh=tf.div(dbh, self.FLAGS.batch_size)
        dbv=tf.div(dbv, self.FLAGS.batch_size)
        
        return dW,dbh,dbv
```

#### Update Step

![image-20200827004037814](media\Chapter5_神经网络\image-20200827004037814.png)

```python
def optimize(self, v):
        ''' Optimization step. Gibbs sampling, calculating of gradients and doing an update operation.
        
        @param v: visible nodes
        @return update operation
        @return accuracy
        '''

        with tf.name_scope('optimization'):
            v0, vk,ph0, phk, _=self._gibbs_sampling(v)
            dW,db_h,db_v=self._compute_gradients(v0, vk, ph0, phk)
            update_op =self._update_parameter(dW,db_h,db_v)
        
        with tf.name_scope('accuracy'):
            mask=tf.where(tf.less(v0,0.0),x=tf.zeros_like(v0),y=tf.ones_like(v0))
            bool_mask=tf.cast(tf.where(tf.less(v0,0.0),x=tf.zeros_like(v0),y=tf.ones_like(v0)), dtype=tf.bool)
            acc=tf.where(bool_mask,x=tf.abs(tf.subtract(v0,vk)),y=tf.zeros_like(v0))
            n_values=tf.reduce_sum(mask)
            acc=tf.subtract(1.0,tf.div(tf.reduce_sum(acc), n_values))
            
        return update_op, acc
   
 def _update_parameter(self,dW,db_h,db_v):
        ''' Creating TF assign operations. Updated weight and bias values are replacing old parameter values.
        
        @return assign operations
        '''
        
        alpha=self.FLAGS.learning_rate
        
        update_op=[tf.assign(self.W, alpha*tf.add(self.W,dW)),
                   tf.assign(self.bh, alpha*tf.add(self.bh,db_h)),
                   tf.assign(self.bv, alpha*tf.add(self.bv,db_v))]

        return update_op
```

#### Inference

```python
def inference(self, v):
        '''Inference step. Training samples are used to activate the hidden neurons which are used for calculation of input neuron values.
        This new input values are the prediction, for already rated movies as well as not yet rated movies
        @param v: visible nodes
        @return sampled visible neurons (value 1 or 0 accroding to Bernouille distribution)
        '''
        p_h_v=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(v,self.W), self.bh))
        h_=self._bernouille_sampling(p_h_v, shape=[1,int(p_h_v.shape[-1])])
        
        p_v_h=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(h_,tf.transpose(self.W, [1,0])), self.bv))
        v_=self._bernouille_sampling(p_v_h, shape=[1,int(p_v_h.shape[-1])])
        
        return v_
```

#### Outline

```python

def main(_):
    '''Building the graph, opening of a session and starting the training od the neural network.'''
    
    num_batches=int(FLAGS.num_samples/FLAGS.batch_size)

    with tf.Graph().as_default():
        
        # get the training data for training and inference time
        train_data, train_data_infer=_get_training_data(FLAGS)
        # get the test data for inference
        test_data=_get_test_data(FLAGS)
        
        # make iterators
        iter_train = train_data.make_initializable_iterator()
        iter_train_infer = train_data_infer.make_initializable_iterator()
        iter_test = test_data.make_initializable_iterator()
        
        x_train= iter_train.get_next()
        x_train_infer=iter_train_infer.get_next()
        x_test=iter_test.get_next()
        
        # buid the model operations
        model=RBM(FLAGS)
        update_op, accuracy=model.optimize(x_train)
        v_infer=model.inference(x_train_infer)
 
        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer())

            for epoch in range(FLAGS.num_epoch):
                
                acc_train=0
                acc_infer=0

                sess.run(iter_train.initializer)
                
                # training 
                for batch_nr in range(num_batches):
                    
                    # run the update and accuracy operation
                    _, acc=sess.run((update_op, accuracy))
                    acc_train+=acc
                
                    # validation
                    if batch_nr>0 and batch_nr%FLAGS.eval_after==0:
                        
                        sess.run(iter_train_infer.initializer)
                        sess.run(iter_test.initializer)

                        num_valid_batches=0
                    
                        for i in range(FLAGS.num_samples):
    
                            v_target=sess.run(x_test)[0]
                    
                            if len(v_target[v_target>=0])>0:
                                
                                # make an prediction
                                v_=sess.run(v_infer)[0]
                                # predict this prediction with the target fro the test ste.
                                acc=1.0-np.mean(np.abs(v_[v_target>=0]-v_target[v_target>=0]))
                                acc_infer+=acc
                                num_valid_batches+=1
        
                        print('epoch_nr: %i, batch: %i/%i, acc_train: %.3f, acc_test: %.3f'%
                              (epoch, batch_nr, num_batches, (acc_train/FLAGS.eval_after), (acc_infer/num_valid_batches)))
                    
                        acc_train=0
                        acc_infer=0
```

## 参考文献

[1]李航编著.统计学习方法[M].清华大学出版社,2012.<br>
[2]https://blog.csdn.net/itplus/article/details/19408143