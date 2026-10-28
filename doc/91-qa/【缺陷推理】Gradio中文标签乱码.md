# 问题描述

Gradio 调试页结果图上的中文标签显示为方框（豆腐块），例如「短带 0.37」中「短带」不可读，英文数字正常。

复现环境：Ubuntu（Gradio）。macOS 同样可能因缺 PingFang 路径触发。

Ubuntu 建议安装：`sudo apt install fonts-wqy-microhei fonts-noto-cjk` 后重启服务。

# 关联文档

| 类型 | 文档路径 |
|:---|:---|
| 原始需求 | `scanly/doc/01-or/【缺陷推理】边端HTTP推理服务.md` |
| 开发设计 | `scanly/doc/02-dr/缺陷推理/02.推理服务与判定逻辑.md` |
| 测试设计 | `scanly/doc/03-tr/缺陷推理/HTTP接口与推理服务测试设计.md` |

# 根因分析

`predict_all.draw_results` 用 Pillow 画「胶缝 0.72 [NG+标+排]」。4090 镜像没有文泉驿或 Noto，系统字体候选全部落空，回退到 `ImageFont.load_default()`。该字体没有中文轮廓，「胶缝」「标」「排」显示为方框；`NG` 与数字正常。

# 涉及文件

| 文件 | 修改摘要 |
|:---|:---|
| `scanly/defects/predict/predict_all.py` | 优先加载随包 `fonts/NotoSansSC-label.ttf`；仍找不到可画中文的字体时，类名改英文、贴标/排改 `L`/`D` |
| `scanly/defects/predict/fonts/NotoSansSC-label.ttf` | Noto Sans SC 子集，只含标签用到的汉字与数字符号 |

# 设计约束更新

结果图画字优先使用随包字体。系统中文字体仍作后备。找不到可画中文的字体时不得再画方框，改用英文类名，贴标写成 `L`、排出写成 `D`。
