## MDM
### 摘要
级联回归最近已经成为解决可变形的人脸对齐等非线性最小二乘问题的适用方法.对于给定尺寸大小的训练集,级联回归学习一组通用规则从而应用于最小二乘问题。尽管级联回归已经在人脸对齐、头部姿势估计等问题取得成功，却仍然出现了一些缺陷。特别是在以下几个方面：

        1. 级联回归学习方式独立
        2. 下降方向可能影响另一个输出MDM
        3. 手工特征（如，HOG,SIFT）主要用于驱动级联，可能导致次优化问题
 
在本文，我们提出了一个混合的联合训练的卷积递归神经网络架构以实现端到端的训练系统并减轻上述缺陷。循环模块通过假设级联形成非线性动态系统来促进回归器的联合优化，实际上通过引入记忆单元充分利用在所有级联层级间的信息在所有层级间的共享。卷积模块允许专门用于现在任务的特征提取并且通过实验证明输出优于手工制作的特征。我们表明，所提出的架构在人脸对齐问题上的应用导致了一个有效的提升超过了当前的技术。
### 介绍
非线性最小二乘优化问题通常应用于计算机视觉，包括但是不限于Structure-from-Motion（从运动中重建3D场景），精确可变形的图形对齐，光流估计，估计镜头参数用于校准。标准牛顿步方法的应用对于取回标准参数是具有挑战性的，因为损失函数的高度非凸性以及常用图像特征的不可微性。最近，为了处理牛顿步方法的缺陷，一组通用的梯度下降方法应用到了级联回归器中。通常，这些方向通过模拟独立的逐级学习。因此，为了处理可变形的人脸对齐，训练图像的真实人脸形状随机扰动（根据固定的变化）。之后下降方向独立地估计寻求从扰乱形状到真实形状前进。最简单的模式，这些规则通过应用连续的线性回归阶段来学习，每个最小化所有样本的最小误差。
    
为了解决非线性最值问题的回归/学习基础方式的在计算机视觉的应用有非常丰富的历史，首先是AAM，从训练集中学习评价Jacobian矩阵。级联回归方法论在最近的很多工作中都有所提及，然而，据我们所知，对于非线性最小化二乘的解决最有促进的是SDM。在现存的级联回归方法在可变形的人脸对齐有几个缺陷：

    *    级联步骤学习是独立的。每个线性回归器简单地学习怎么从一个形状扰动的特定图像到真实图像。  
         因此，不考虑诸如面部姿势之类的语义相关图像特征之间的相关性。
    *    最优化的结果紧紧地与用于驱动回归的特征相关。手工制作的特征不是数据驱动的，因此对于人脸对齐的任务不是最佳的选择。  
         相反的，基于二进制或者树的特征是数据驱动的并且被证明在人脸对齐中十分有效。
         然而，这些简单的像素强度差异无法以端到端的方式学习。对于各种计算机视觉任务的卷积特征所看到的成功尚未实现面部对齐。
         特别是，目前没有提出的系统训练端到端的卷积特征。
       
在本文，我们提出了MDM来解决上述问题。特别是，MDM将可变形的人脸对齐模型化为非线性动态系统。MDM维护一个内部存储器单元累积从输入空间的所有过去观察的历史中提取的信息。这具有以下优点：根据先前计算的下降方向自然地划分下降方向。当应用到人脸对齐时，这种范例映射到非常直观的合理。例如，任何近侧剖面与正面初始化的对齐似乎是合理的，它将具有非常相似的下降方向序列。MDM利用这些丰富的信息并训练端到端的人脸对齐方法，该方法可以学习一组数据驱动的功能，使用卷积神经网络（CNN），直接从图像中以级联方式，最重要的是使用递归神经网络（RNN）对下降方向施加记忆约束
