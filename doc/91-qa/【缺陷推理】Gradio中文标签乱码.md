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

`predict_all.draw_results` 通过 Pillow 绘制中文，字体解析 `_resolve_cjk_font` 优先使用 `/System/Library/Fonts/PingFang.ttc`。新 macOS 上该路径不存在；部分 `.ttc` 未指定子字体索引时加载失败，最终回退到 `ImageFont.load_default()`（仅拉丁字形），中文显示为方框。

# 涉及文件

| 文件 | 修改摘要 |
|:---|:---|
| `scanly/defects/predict/predict_all.py` | 扩展系统中文字体候选；TTC 尝试多 index；用样字校验能否绘制中文；失败打警告 |

# 设计约束更新

Gradio 结果图画字须使用可渲染中文的系统字体（macOS：Hiragino Sans GB / STHeiti / Songti / Arial Unicode；Windows：微软雅黑/黑体；Linux：文泉驿/Noto），不得静默回退到默认拉丁字体而不告警。
