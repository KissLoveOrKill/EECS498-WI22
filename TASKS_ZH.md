# EECS 498 WI22 中文任务提示

Notebook 中的英文任务说明均已保留；每个 Notebook 开头也包含对应的中文任务清单。

## `A1/pytorch101.ipynb`

本 Notebook 的英文说明保留在下方；以下是对应的中文任务清单。

1. 完成 `pytorch101.py` 中的 `create_sample_tensor`、`mutate_tensor` 和 `count_tensor_elements`，练习张量创建、原地修改和形状计算。
2. 实现 `create_tensor_of_pi`，使用张量构造器创建指定形状、元素值为 pi 的张量。
3. 实现 `multiples_of_ten`，返回给定范围内所有 10 的倍数，并确保 dtype 为 `torch.float64`。
4. 实现 `slice_indexing_practice` 和 `slice_assignment_practice`，练习切片访问与切片赋值；后者必须原地修改输入张量，并只修改指定的前 4 行、前 6 列。
5. 实现 `shuffle_cols`、`reverse_rows` 和 `take_one_elem_per_col`，使用整数索引；每个函数都只能对输入执行一次索引操作。
6. 实现 `make_one_hot`，把整数列表转换为 one-hot 矩阵；实现 `sum_positive_entries`，用一次布尔索引求所有正数之和。
7. 实现 `reshape_practice`，只用 reshape/view/transpose 等张量操作，把 0 到 23 变成题目要求的 `(3, 8)` 结果。
8. 实现 `zero_row_min`，把每行最小值置零，不使用显式循环。
9. 实现批量矩阵乘法的循环版和无循环版，并比较两者速度。
10. 实现 `normalize_columns`，按列减均值并除以标准差。
11. 在 GPU 可用时完成 GPU 矩阵乘法练习；运行并保存所有单元格后再提交。

## `A1/knn.ipynb`

本 Notebook 的英文说明保留在下方；以下是对应的中文任务清单。

1. 加载并可视化 CIFAR-10，先使用较小的训练集和测试集进行快速调试。
2. 在 `knn.py` 中实现 `compute_distances_two_loops`；距离计算不得使用 `torch.norm`、`torch.dist`、`torch.nn` 或 `torch.nn.functional`。
3. 实现只遍历一次训练集的 `compute_distances_one_loop`，并实现完全向量化、无 Python 循环的 `compute_distances_no_loops`。
4. 实现 `predict_labels`，根据最近的 k 个训练样本进行多数投票；完成 `KnnClassifier` 类。
5. 在 CIFAR-10 子集上测试 kNN，检查 k=1 和 k=5 的准确率与英文说明中的预期是否接近。
6. 实现 `knn_cross_validate`，用交叉验证选择 k，并在更大的训练集和完整数据集上重新测试。
7. 保存所有 Notebook 和 Python 文件，并运行提交单元格生成作业压缩包。

## `A2/linear_classifier.ipynb`

本 Notebook 的英文说明保留在下方；以下是对应的中文任务清单。

1. 实现 SVM 损失的朴素版本梯度，并包含 L2 正则化梯度；用数值梯度检查验证结果。
2. 实现完全向量化的 `svm_loss_vectorized`，同时实现其梯度，比较数值、误差和速度。
3. 完成 `train_linear_classifier`、`sample_batch` 和分类器预测流程，使用 SGD 训练线性分类器。
4. 使用验证集调节学习率和正则化强度，记录最佳超参数并可视化最终权重。
5. 实现 Softmax 损失与梯度的循环版和向量化版，进行数值梯度检查和速度比较。
6. 完成 Softmax 分类器的训练、验证和测试流程；运行所有单元格并保存结果。

## `A2/two_layer_net.ipynb`

本 Notebook 的英文说明保留在下方；以下是对应的中文任务清单。

1. 在 `two_layer_net.py` 中完成 `nn_forward_pass` 的类别分数计算。
2. 完成 `nn_forward_backward` 的损失计算和反向传播，包含正则化项，并用数值梯度检查。
3. 完成参数更新、预测、`nn_train` 和 `nn_get_search_param` 等训练辅助函数。
4. 使用验证集调节超参数，训练两层神经网络，并在测试集上报告准确率。
5. 保留英文说明要求的可视化和运行结果，提交前运行全部单元格。

## `A2/challenge_problem.ipynb`

本 Notebook 的英文说明保留在下方；以下是对应的中文任务清单。

1. 使用网页交互工具手工设计两层神经网络：第一层为手写数字模板，第二层为全连接权重。
2. 下载并保存手工设计的模型文件，按英文说明的文件名放入 A2 目录。
3. 加载 MNIST 测试集，运行手工设计的网络并观察准确率；目标是超过约 20%。
4. 思考两层网络如何用隐藏层模板组合出任意数字，并保存最终 Notebook 与模型文件。

## `A3/fully_connected_networks.ipynb`

本 Notebook 的英文说明保留在下方；以下是对应的中文任务清单。

1. 实现线性层和 ReLU 的前向、反向传播，并用数值梯度检查。
2. 完成 `TwoLayerNet` 的参数初始化、前向传播和反向传播。
3. 完成 `FullyConnectedNet` 的初始化、前向传播和反向传播，先实现基础网络，再加入 dropout 和 batch normalization。
4. 完成 solver 创建函数；让小型两层、三层和五层网络在小数据集上过拟合，检查损失和准确率曲线。
5. 实现 SGD、Momentum、RMSProp 和 Adam 更新规则，比较数值结果。
6. 实现 dropout 的训练/测试前向传播与反向传播，并通过梯度检查。
7. 调整学习率、权重初始化等超参数，训练效果较好的全连接网络并保存结果。

## `A3/convolutional_networks.ipynb`

本 Notebook 的英文说明保留在下方；以下是对应的中文任务清单。

1. 实现卷积层和最大池化层的前向、反向传播；先用朴素实现通过梯度检查。
2. 完成 `ThreeLayerConvNet` 的初始化、前向传播和反向传播。
3. 完成 `DeepConvNet` 的参数初始化、前向传播和反向传播，并在小数据集上过拟合。
4. 实现线性层和卷积层的 Kaiming 初始化。
5. 实现 batch normalization 和 spatial batch normalization 的训练前向、测试前向及反向传播。
6. 创建 solver，调整权重尺度和学习率，训练并评估卷积网络；保存英文要求的实验结果。

## `A4/one_stage_detector.ipynb`

本 Notebook 的英文说明保留在下方；以下是对应的中文任务清单。

1. 完成 FPN backbone 和预测头，输出每个 FPN 层的位置、类别、框回归和 centerness 预测。
2. 在 `common.py` 中实现 `get_fpn_location_coords` 和 NMS 等基础函数。
3. 在 `one_stage_detector.py` 中实现 FCOS 的位置到框回归量、框解码和 centerness 计算。
4. 完成图像前向传播、真实框与特征位置的匹配，以及分类、框回归和 centerness 损失。
5. 完成预测阶段的分数筛选、框解码和 NMS；运行单元测试和可视化检查。
6. 训练 FCOS 单阶段检测器，保存损失、预测结果和英文要求的提交文件。

## `A4/two_stage_detector.ipynb`

本 Notebook 的英文说明保留在下方；以下是对应的中文任务清单。

1. 完成 FPN backbone、RPN 预测头和各层特征的输出。
2. 生成 anchor boxes，实现 IoU、anchor 与真实框之间的 delta 编码和解码。
3. 完成 RPN 的训练前向传播、anchor 匹配、分类/回归损失和候选框生成。
4. 实现 proposal 筛选、NMS 与 ROI Align；完成第二阶段分类器头的初始化和前向传播。
5. 训练两阶段检测器并检查预测框、分类结果和损失；保存英文说明要求的结果。

## `A5/rnn_lstm_captioning.ipynb`

本 Notebook 的英文说明保留在下方；以下是对应的中文任务清单。

1. 实现 vanilla RNN 单步前向/反向传播，以及完整序列的前向/反向传播。
2. 实现 WordEmbedding 和 temporal softmax loss，并通过梯度检查。
3. 完成 `CaptioningRNN` 的初始化、训练前向传播和测试阶段采样。
4. 实现 LSTM 单步前向传播和完整时间序列前向传播。
5. 实现点积注意力、Attention LSTM 单步及完整序列前向传播。
6. 运行训练和采样流程，在训练集与验证集上观察图像描述结果并保存 Notebook。

## `A5/Transformers.ipynb`

本 Notebook 的英文说明保留在下方；以下是对应的中文任务清单。

1. 实现序列元素编码、单头注意力的双循环版和无循环版，并验证输出形状和数值。
2. 实现带 mask 的自注意力，并构造解码器的后续位置 mask。
3. 完成 MultiHeadAttention、LayerNormalization 和 FeedForwardBlock。
4. 完成 EncoderBlock、DecoderBlock 和完整 Transformer 的初始化与前向传播。
5. 实现简单位置编码和正弦位置编码，并将它们接入数据加载流程。
6. 在小数据集上过拟合验证实现，再用完整数据训练；调整损失、位置编码、学习率等设置。
7. 可视化 encoder 和 decoder 的 attention 权重，保存训练曲线和结果。

## `A6/variational_autoencoders.ipynb`

本 Notebook 的英文说明保留在下方；以下是对应的中文任务清单。

1. 在 `vae.py` 中定义全连接 VAE 的 encoder、decoder、hidden dimension、mu 层和 logvar 层。
2. 实现重参数化技巧、VAE 前向传播和负变分下界损失；检查均值、标准差和损失误差。
3. 训练 VAE，生成数字图像，并进行潜空间插值。
4. 实现条件 VAE：把 one-hot 类别向量拼接到输入图像和潜变量，完成 encoder、decoder 和 forward。
5. 训练 CVAE，按指定数字类别生成图像；保存训练损失和生成结果。

## `A6/generative_adversarial_networks.ipynb`

本 Notebook 的英文说明保留在下方；以下是对应的中文任务清单。

1. 实现 `sample_noise`，生成范围为 [-1, 1] 的均匀噪声。
2. 使用 `nn.Sequential` 实现全连接 discriminator 和 generator。
3. 实现原始 GAN 的 discriminator loss、generator loss 和 Adam 优化器。
4. 训练全连接 GAN，观察生成图像，并保存英文要求的结果文件。
5. 实现 Least Squares GAN 的两个损失，检查相对误差并重新训练。
6. 实现 DCGAN 的卷积 discriminator 和反卷积 generator，检查参数量并训练。
7. 对训练好的 GAN 做潜空间插值；保存损失、图像和最终 Notebook。

## `A6/network_visualization.ipynb`

本 Notebook 的英文说明保留在下方；以下是对应的中文任务清单。

1. 加载 ImageNet 预训练模型和示例图像，确认模型及设备设置正确。
2. 实现 `compute_saliency_maps`，计算正确类别分数对输入像素的梯度并可视化显著图。
3. 实现 `make_adversarial_attack`，用图像梯度生成 L2 对抗样本，使模型产生错误预测。
4. 实现 `class_visualization_step`，通过梯度上升生成模型认为属于目标类别的图像。
5. 尝试其他类别，保存原图、攻击图、差异图、类别可视化和训练结果。

## `A6/style_transfer.ipynb`

本 Notebook 的英文说明保留在下方；以下是对应的中文任务清单。

1. 加载预训练 backbone、内容图和风格图，理解特征空间中的内容与风格表示。
2. 实现 content loss、Gram matrix、style loss 和 total variation loss，并通过误差检查。
3. 完成普通风格迁移，运行英文指定的参数组并保存生成图像。
4. 完成 feature inversion：从随机噪声出发重建内容特征。
5. 挑战部分实现 guided Gram matrix 和 guided style loss，完成空间控制的风格迁移。
6. 保存损失、普通风格迁移、特征反演和空间风格迁移结果。
