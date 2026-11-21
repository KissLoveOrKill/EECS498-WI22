# 风格迁移：内容、纹理与区域控制

对应 [style_transfer.py](/D:/Study/Phd-Study/EECS498-WI22-main/A6/style_transfer.py) 和 [Style Transfer Notebook](/D:/Study/Phd-Study/EECS498-WI22-main/A6/style_transfer.ipynb)。核心是：**固定预训练 SqueezeNet，直接优化生成图片的像素，使其保留内容结构并匹配目标风格。**

## 1. 整体流程

```text
内容图 → CNN 特征 → 固定内容目标
风格图 → CNN 特征 → Gram 矩阵 → 固定风格目标
生成图 → CNN 特征 → 内容损失 + 风格损失 + TV 损失 → 更新生成图
```

生成图从内容图或随机噪声初始化，通过 `Adam([img])` 迭代更新。CNN 参数固定，但必须保留从损失到输入图片的梯度路径。

$$
L=L_{\mathrm{content}}+L_{\mathrm{style}}+L_{\mathrm{TV}}
$$

三个损失函数内部已经乘上各自权重，汇总时直接相加。

## 2. 普通风格迁移

### 内容损失：保留结构

比较生成图与内容图在某一层的特征，位置一一对应：

$$
L_{\mathrm{content}}=w_c\|F^\ell-P^\ell\|_F^2
$$

代码为 `content_weight * ((content_current - content_original) ** 2).sum()`。这里使用平方和，不取均值；无需展平特征。

### Gram 矩阵与风格损失：匹配纹理

将特征从 `(N,C,H,W)` 展平为 `(N,C,HW)`，按每张图片计算通道之间的内积：

$$
G^\ell=\frac{F^\ell(F^\ell)^T}{C_\ell H_\ell W_\ell}
$$

输出为 `(N,C,C)`。Gram 矩阵汇总各空间位置的通道相关性，用来描述纹理、笔触等风格统计，不保留每个位置的排列。

$$
L_{\mathrm{style}}=\sum_i w_i\|G^{\ell_i}-A_i\|_F^2
$$

其中 `A_i = style_targets[i]` 已经是目标 Gram 矩阵。多层分别计算平方误差，再按权重相加。

### TV 损失：让图片平滑

$$
L_{\mathrm{TV}}=w_t\left[
\sum_{c,h,w}(I_{c,h+1,w}-I_{c,h,w})^2
+\sum_{c,h,w}(I_{c,h,w+1}-I_{c,h,w})^2
\right]
$$

两项分别遍历有效的上下、左右相邻像素：

```python
vertical = img[:, :, 1:, :] - img[:, :, :-1, :]
horizontal = img[:, :, :, 1:] - img[:, :, :, :-1]
loss = tv_weight * (vertical.square().sum() + horizontal.square().sum())
```

增大内容权重更保留原图结构；增大风格权重更强调纹理；增大 TV 权重使画面更平滑。

## 3. 空间风格迁移：指定区域使用哪种风格

内容 mask 指定风格应用到哪里，风格 mask 指定从哪里提取风格，两者按区域编号 `r` 配对。本实验的 mask 来自 `A6/images` 中预先生成的黑白图片：

| 区域 | 风格来源 | 应用位置 |
| --- | --- | --- |
| `r=0` | `fig2_style1_nosky.jpg` 选出的非天空区域 | `fig2_content_nosky.jpg` 选出的区域 |
| `r=1` | `fig2_style2_sky.jpg` 选出的天空区域 | `fig2_content_sky.jpg` 选出的区域 |

`get_zero_one_masks` 将掩码图片二值化为 0/1，`extract_layer_masks` 再将其缩放到各层特征图大小。两个对应区域的形状、坐标可以不同；换图片时需要准备对应的新 mask。

### Guided Gram 矩阵

| 张量 | 形状 |
| --- | --- |
| 区域特征 | `(N,R,C,H,W)` |
| 区域掩码 | `(N,R,H,W)` |
| `masks.unsqueeze(2)` | `(N,R,1,H,W)` |
| Guided Gram 矩阵 | `(N,R,C,C)` |

先用掩码保留指定区域，再展平空间维度，为每张图片的每个区域独立计算 Gram 矩阵：

$$
\widetilde F^{\ell,r}=T^{\ell,r}\odot F^{\ell,r},\qquad
G^{\ell,r}=\frac{\widetilde F^{\ell,r}(\widetilde F^{\ell,r})^T}{C_\ell H_\ell W_\ell}
$$

```python
F = (features * masks.unsqueeze(2)).reshape(N, R, C, H * W)
guided_gram = F @ F.transpose(-2, -1)
```

`matmul` / `@` 支持多个批次维度；开启归一化时仍除以 `C*H*W`，不除以区域数或 mask 面积。

### Guided Style Loss

$$
L_{\mathrm{guided}}=\sum_i w_i\sum_r\|G^{\ell_i,r}-A_i^r\|_F^2
$$

与普通 `style_loss` 相同，只需改用 `guided_gram_matrix(feats[layer], content_masks[layer])`。目标矩阵已用风格 mask 提前算好，平方差 `.sum()` 会同时累加所有区域。

## 4. 实现时记住

- **索引**：`feats[layer]`、`content_masks[layer]` 按 CNN 层号取值；`style_targets[i]`、`style_weights[i]` 按所选层的序号取值。
- **求和**：各项使用平方误差之和，避免误用 `mean()`；风格损失的 `return loss` 放在循环外。
- **梯度**：`img.requires_grad_()`；每轮 `zero_grad → forward → backward → step`，只更新生成图。
- **两种控制**：普通 Gram 匹配全图风格统计；Guided Gram 在指定区域内匹配风格统计。
