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
       
在本文，我们提出了MDM来解决上述问题。特别是，MDM将可变形的人脸对齐模型化为非线性动态系统。MDM维护一个内部存储器单元累积从输入空间的所有过去观察的历史中提取的信息。这具有以下优点：根据先前计算的下降方向自然地划分下降方向。当应用到人脸对齐时，这种范例映射到非常直观的合理。例如，任何近侧剖面与正面初始化的对齐似乎是合理的，它将具有非常相似的下降方向序列。MDM利用这些丰富的信息并训练端到端的人脸对齐方法，该方法可以学习一组数据驱动的功能，使用卷积神经网络（CNN），直接从图像中以级联方式，最重要的是使用递归神经网络（RNN）对下降方向施加记忆约束。我们的工作也受到卷积递归网络端到端训练的成功推动，以完成图像标题生成任务，场景解析，图像检索、注释生成。据我们所知，这是第一次端到端的递归卷积神经网络系统应用到可变形的目标对齐。总体上，这项工作有一下贡献：

    *    我们提出了一个非线性级联框架，处理端到端的非线性功能的下降方向的学习。
         这种类型的函数被广泛应用于计算机视觉和现存的工作，如SDM。已经显示出下降方向的学习能够以更高效的方式。
    *    这是在人脸对齐方面的首次工作通过单个模块从原生图像像素到最终的预测结果。
         我们通过CNN学习新的图像特征，在训练过程中纳入特定于问题的信息。  
    *    我们在下降方向的学习中引入了记忆的理念。我们相信他具有高度差异性并且是我们的主要优势之一
    *    我们改进了面部对齐的最新技术，在300w具有挑战的数据集上有一个大幅度的提升。

在本文的后面部分，将用以下方式组织内容。在第二部分，我们概览相关工作，特别需要强调的是SDM。接下来，在第4部分，我们介绍改进的MDM，不失一般性，介绍它在人脸对齐上的应用，最后，在第五部分，我们展示了我们模型的精确评估，为了演示改进MDM超过最新技术的优势。
### 相关工作
对于这项工作的应用可变形的人脸对齐构成了一个非常直观的领域，因此它也被选为评估的主要领域。

人脸对齐有一个非常长久丰富的历史包括在计算机视觉中非常重要的工作，如AAM、CLM、和3D MM。在最近，因为非约束的大数据集的引入，人脸对齐的问题不断增加。问题的增长在数据的多样性和质量方面扩大了辨别模式的能力，如基于回归的方式。特别是，最近很多成功的技术都将回归器依次连接到一起，并称之为级联。级联回归策略在很多流行人脸对齐算法中占据了大比例，因为它们通常表现良好且高效。最有效的级联回归方法是那些通过增强弱学习单元的方法，如随机蕨或者随机森林。然而，这一领域的一项开创性工作是SDM，它可以推广到大量的问题，并能有效地处理大量的非线性最小二乘问题。SDM是第一个将级联回归问题描述为一个更通用的学习框架的工作，它使用从训练数据中学习的下降方向优化非线性目标函数。特别地，我们假设每个级联的回归量是线性的，并在目标函数的空间中模拟平均下降方向。然而，学习到的下降方向，尽管被链锁在一个级联，只有通过从上一个级联剩余的方差彼此相关。因此，初始级联层可能有大的下降步数有可能步数最优收敛。在Global Supervised Descent Method 方法中通过在训练期间根据下降方向划分内聚组的方式解决。在测试时，选择一个分区代表正确的下降方向。例如，对于面部对齐，这需要对形状进行初始估计，并根据面部姿态划分下降方向。然而，这意味着GSM只适用于跟踪前一帧提供了选择正确分区的先验信息的场景。

Asthana，Incremental face alignment in the wild提出了一种SDM类型方法的增量学习框架，该框架支持级联层次的完全独立性。它们假设每个级联是独立的，因此级联级别可以是通过模拟前后剩余的方差来并行学习。尽管，每个层次的独立性可能对增量学习很有吸引力，我们认为下降方向应该受到先前下降步骤的先验知识的影响。我们建议将过程建模为一个非线性动态系统，其中一个连续的潜在状态变量适当地驱动过程。在这篇文章中，我们展示了如果不是使用手工制作的特性，而是以端到端方式学习给定问题的最优特性，就有可能获得很大的改进。

我们提出的方法也让人想起以前提出的人脸对齐深度学习方法。Deep Convolutional Network Cascade for Facial Point Detection提出使用独立的卷积神经网络来表现由粗到细的形状搜寻。CFAN也使用粗到细的形状搜索，首先使用全局搜索，然后使用一组本地堆积的自动编码器。然而，每个自动编码器的训练时孤立的。Learning Deep Representation for Face Alignment with Auxiliary Attributes提出一种将辅助信息融入拟合过程的新方法。与其他相关方法不同的是，他们不包含级联网络，而是将问题框定为多任务学习问题。Facial Feature Tracking Under Varying Facial Expressions and Face Poses Based on Restricted Boltzmann Machines 使用深度置信网络来训练更加灵活的过程，但不要学习任务卷积特征。Deep Regression for Face Alignment 提出联合学习一个级联的线性回归器。虽然这个回归器通过反馈联合更新，但是其其使用线性回归器并且使用手工制作的特征而不是直接从图像中提取特征。同时在Deep Regression for Face Alignment 结果的中我们可以发现在对齐精度上并没有超过独立训练的级联回归器。在接下来的第三部分，我们将系统得介绍人脸对齐的问题并简要的描述SDM算法。
### 级联回归
人脸对齐的定义是在图像上找到一组稀疏基准点，![集合](https://github.com/purity77/mdm-translation/blob/master/useimage/1.jpg).给出一个图形以及形状的初始估计，![初始估计](https://github.com/purity77/mdm-translation/blob/master/useimage/2.png).人脸对齐要求恢复到真实形状x* 例如SDM回归模型，从大量的训练集中通过接连地一系列的线性回归器学习从x<sub>（0）</sub>到x* 。最常见的，回归参数的优化是基于一组复杂的特征提取从每个图像周围的L个基准点的局部区域。我们表示这些从一个图像提取的固定大小的特征为：![特征](https://github.com/purity77/mdm-translation/blob/master/useimage/3.png)
因为SDM提出学习一个级联的回归函数，回归的目标变量表示为形状增量，定义为：![目标](https://github.com/purity77/mdm-translation/blob/master/useimage/4.png) 在这里k是当前级联的索引，因此x<sub>i</sub><sup>(k)</sup>是当前第i个图形的估计形状。SDM提出利用k线性回归器得到公式为：![公式](https://github.com/purity77/mdm-translation/blob/master/useimage/5.png)
### 记忆下降方法
人脸对齐的级联回归技术从特征提取阶段开始，比较典型的是手工提取的代表图形块如（HOG,SIFT）.特征提取阶段是必须的因为图像的捕获是在无约束条件下因此可能包含外观的变化等，进而在环境中产生局部最小值。上述的表现平滑了环境，以尽量减少这些变化的影响。我们注意到MDM与特性无关，可以直接与任何此类非线性表示一起使用。然而，尽管传统的手工制作功能已被证明对计算机视觉中的许多任务有效，这个过程仍然可以被认为是次优的，因为这些表示也是独立于当前任务提取的。MDM通过提供端到端训练方法来解决这个问题，实际上是联合发现合适的非线性图像表示以及最优的地标位置。将特征提取阶段替换为卷积网络模块，方便学习方向滤波器，实现函数优化。由于训练是通过反向传播以端到端方式进行的，所以我们本质上学习了用于将patches与拟合过程联合卷积的滤波器。正如上一节所讨论的，与其他最先进的算法相比，该算法有几个优点。其中一个核心贡献是发现了最优的特征表示，因为这消除了使用可能不是最优的手工特征的需求。在下一节中，我们将在(4.2节)中介绍MDM算法及其端到端训练（4.3节）。
#### 模型
提出MDM的主要目的是通过将以前独立的级联步骤视为非线性动态系统下的时间步长，从而促进平滑收敛。在这种模式下，我们实际上在每次迭代中学习的是单个模型，而不是独立的回归量，通过保存助记模块，MDM使每次迭代所采取的步骤依赖于前面的步骤。这有效地防止了一些陷阱，比如通过“逐步过渡”错过了最优的函数。为此，我们使用循环神经网络(RNN)实现MDM，这是众所周知的非线性动态系统的通用近似器。本质上，RNNs促进了反馈连接，从而在网络中产生循环和递归。这使得递归网络能够解释数据中出现的时间依赖性。就MDM而言，这支持在迭代和下降方向之间建模依赖关系。同时，RNNs保持了前馈网络的拓扑结构，反馈连接支持对系统当前状态的表示，该状态封装了来自以前输入的信息。RNNs在机器翻译[48]、语音识别[25]、图像字幕[53]等领域的应用取得了很大的成功。

在最简单的形式,一个RNN观察zt型对应于当前时间步t,并基于前面的状态ht−1生成下一个隐藏的状态,ht。下式被认为是递归网络的基本方程(也称为阶跃函数或递归方程)：![递归方程](https://github.com/purity77/mdm-translation/blob/master/useimage/6.png) 现在让我们在级联回归的背景下考虑上面的问题，特别是在脸部对齐的情况下.MDM的目标是，根据对能量图谱的最小值的初步粗略估计，生成一系列下降方向，从而迭代地达到最佳状态。我们用x<sup>(0)</sup>表示这个初始估计值，对于人脸对齐这个值通常是对齐到人脸检测器的输出的平均值。在每个时间步长t，助记模块观测能量图景Z<sup>（t）</sup>.根据这一观察，MDM的内部状态将通过调整递归方程的递归表达式相应地更新。给定一个新的训练样本，递归方程为，![新方程](https://github.com/purity77/mdm-translation/blob/master/useimage/7.png) 此处W<sub>hi</sub>是隐藏层到输入层矩阵用于对能量图谱的局部观测，W<sub>hh</sub>是隐藏层到隐藏层矩阵前一个时间步长和U对应于循环模式内部状态的维数。在训练过程中，网络通过将算法的助记元素对应的隐藏状态投影到隐藏到输出矩阵中来更新当前的形状位移W<sub>ho</sub>公式为：![更新](https://github.com/purity77/mdm-translation/blob/master/useimage/8.png) 这一估计基本上构成了对新时间步长的观测，并转化为对x周围局部能量图谱的观测网络围绕![](https://github.com/purity77/mdm-translation/blob/master/useimage/9.png)。注意，在这种情况下，传统的SDM只会训练一个独立的回归量，因此不能利用算法的前一种状态。实际上，通过实验分析我们发现，通过可视化，在拟合过程的早期，隐藏的网络状态封装了头部姿态信息。然后可以在随后的阶段中利用这一点来划分能源图谱，从而使模型能够选择合适的下降路径。然后对预定义的时间步长重复该过程，通过使用时间反向传播（BPTT）进行训练。综上所述，MDM目标函数可以这样定义： ![](https://github.com/purity77/mdm-translation/blob/master/useimage/10.png)，此处![](https://github.com/purity77/mdm-translation/blob/master/useimage/11.png) 表示时间t时n个图像对应的所有矩阵状态，和所有的模型参数θ。

在图二，我们图解了MDM在人脸对齐任务中的应用。给定一个对应于人脸平均形状的初始估计X<sup>(0)</sup>，通过卷积模块来提取和传播patches，获得合适的非线性特征表示。然后，递归模块，f<sub>r</sub>产生下一个状态h<sup>(1)</sup>这个过程根据固定数量的时步而重复。通过对验证集的实验，我们发现T=4时间步骤的网络是足够的。这与之前级联回归工作是一致的，4级级联被认为是足够的。整个网络都是端到端可微的，我们可以使用时间反向传播去学习模型的参数。
### 实验分析
为了评估所建议的MDM的有效性，我们对最先进的面部对齐方法进行严格的评价（5.1段）。在5.2段，我们进一步测试了我们的模组，通过

    *       研究更多的时间步骤的影响T
    *       通过比较使用不同特征提取器的学习结果：用卷积模块提取和手工提取
    *       可视化拟合过程的内部状态，它揭示了它封装头部的构成信息,网络可以在随后的时间步骤中划分下降方向的空间。

#### 数据集
为了与最近的其他人脸对齐方法进行比较，我们集中于由Sagonas所注释的68个特征点，这些注解提供给3个广泛应用的数据集（LFPW HELEN AFW），这些数据集最初标注使用不同和矛盾的审定方式。Sagonas 等也提出了一个新的具有挑战性的数据集IBUG，同样在68个点上使用CMU MultiPIE标记。通常，这些注释被分割成以下的子集合：

    *     训练集 LFPW(811张） HELEN(2000张) AFW(337张) 
    *     挑战子集 IBUG(135)张
    *     共用子集 LFPW(224) HELEN(330)
    *     完全子集 挑战＋共用

上述注释实际上是作为300W人脸对齐竞赛的训练/验证集提供的，使用另一组图像进行严格的评估，被称作300W测试集。这个300W包含600个图像分成两个子集，包括indoor和outdoor，那些数据据说是从与IBUG数据集类似的分布中提取的。
#### 评估
不幸的是，没有一致的方法报告错误的人脸对齐，即使是对于常见的300W测试集。这主要是由于误差归一化的变化。为了与300W比赛[ The first facial landmark localization challenge]的结果保持一致，我们使用了他们对眼间距的定义，即外眼角之间的距离。我们认为，平均误差，特别是没有伴随标准差的平均误差，并不是一个非常有用的误差度量标准，因为它们可能会被少量非常差的拟合严重偏差。因此，我们以CED曲线的形式给出我们的评价，因为这与[ The first facial landmark localization challenge]得到的结果保持一致。我们已经从CED曲线中计算出了一些进一步的统计数据作为曲线下面积(area-under-the-curve, AUC)和每种方法的故障率(我们认为任何点对点误差大于0.08的拟合都是失败的)














