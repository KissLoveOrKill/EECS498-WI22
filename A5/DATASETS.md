# A5 数据集

## COCO 图像字幕

本地工作区中的数据文件位于 `A5/datasets/coco.pt`（未随 Git 仓库发布）。在 **A5 目录**运行
`rnn_lstm_captioning.ipynb`，原有的 `load_coco_captions("./datasets/coco.pt")`
即可读取。

课程原链接 `http://web.eecs.umich.edu/~justincj/teaching/eecs498/coco.pt`
在 2026-11-07 检查时返回 HTTP 404。因此本地文件是从公开 COCO 镜像重建的
**作业格式兼容版本**，不保证与课程原文件的图片划分、词表或预处理完全相同，
训练结果也不应直接与原文件的结果比较。

| 字段 | 形状 / 内容 |
| --- | --- |
| `train_images` | `(10000, 3, 112, 112)`，`torch.uint8`，像素范围 0–255 |
| `val_images` | `(500, 3, 112, 112)`，`torch.uint8`，像素范围 0–255 |
| `train_captions` | `(10000, 17)`，`torch.int64` |
| `val_captions` | `(500, 17)`，`torch.int64` |
| `vocab` | 2,516 个 token，包含 `idx_to_token` 列表和 `token_to_idx` 字典 |

来源为 [clip-benchmark/wds_mscoco_captions](https://huggingface.co/datasets/clip-benchmark/wds_mscoco_captions)，
固定版本为 `3bf9ad5d280b03126293bcc691585015e29ce691`，使用 `train/0.tar` 和
`train/1.tar` 中顺序排列的前 10,500 张图片，每张图片使用第一条非空字幕。
前 10,000 张用于训练，后 500 张作为验证集；这里的验证集是从镜像训练分片中留出的，
并非 COCO 官方验证划分。训练、验证图片在本地生成时做过重复检查。数据文件、provenance 元数据和预览图仅存在于本地工作区，不随 Git 仓库发布。

图像经过 EXIF 方向校正、RGB 转换和双线性缩放。字幕转为小写，保留英文单词、
数字和单词内部的英文撇号，截取最多 15 个词，然后添加 `<START>` 和 `<END>`，
使用 `<NULL>` 补齐到长度 17。词表只从训练字幕建立；训练中出现少于 2 次的词及
验证中的未登录词替换为 `<UNK>`。四个特殊 token 的编号依次为 0、1、2、3。

来源、分片 SHA256、每个划分的图片标识、预处理参数和生成文件 SHA256 保存在
`A5/datasets/coco.provenance.json` 中。

本地验证已通过：Notebook 原始数据加载单元格、序列与特殊 token、全部图片的重复
检查，以及预训练 RegNet-X 400MF 的特征提取（4 张图片得到 `(4, 400, 4, 4)`）。
验证记录保存在 `A5/datasets/validation.json`；图像与字幕示例见
`A5/datasets/coco_preview.jpg`。

本仓库未包含用于重建该文件的 `tools/build_coco.py`。完整复现需要下载约 2.04 GB 的原始分片，并按照上述固定版本、图片划分和预处理步骤自行生成数据；因此仅凭本仓库无法一键重建本地 `coco.pt`。生成文件约为 378 MiB。

## Transformer 算术数据

`A5/two_digit_op.json` 已随项目提供，包含 5,000 对输入和输出表达式，
`Transformers.ipynb` 可直接使用，无需额外下载。
