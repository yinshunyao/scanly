# 关联文档

| 类型 | 文档路径 |
|:---|:---|
| 原始需求 | `scanly/doc/01-or/【缺陷推理】边端HTTP推理服务.md` |
| 类别角色 | `scanly/doc/01-or/【缺陷类别】核心缺陷与屏蔽非缺陷.md` |
| 开发设计 | `scanly/doc/02-dr/缺陷推理/01.HTTP接口文档.md` |
| 关联设计 | `scanly/doc/02-dr/缺陷推理/02.推理服务与判定逻辑.md` |

# 测试范围

| 范围 | 覆盖点 |
|:---|:---|
| 健康检查 | 有/无模型时 `GET /health/ready` 状态码与 `ready` |
| 预测入参 | 相机数组、对象包一层、`infer_type` detect/obb、缺 camera/images、空路径、文件不存在；`/v1/predict` 与 `/scanly_predict` 相同 |
| 上传预测 | `POST /v1/predict/upload`：单路 `camera`+`files`；`meta.cameras[].count` 多相机；文件数与 count 不一致；空文件；响应与路径接口结构一致 |
| 判定 | 或/与/仅长度；面积 0 跳过；双 0 关闭通道；NG/贴标/排出面积勾选互相独立；obb 用旋转框宽高 |
| 过滤 | `enable=false`、覆膜开关、置信度门限、未知类不置 NG |
| 量纲 | detect：外接框短边/长边；obb：旋转框短边/长边 |
| 方案 | `GET /v1/labels`；`GET/PUT/POST /v1/scheme` 查询、合并、整表替换、落盘 |
| Swagger | `/docs` 200 示例含 `results[].camera` 与 `defects` 二维数组，非 `"string"` |

不在第一版强制：满速产线、三路汇总、真实 GPU 精度验收、Go 服务实现 upload。

# 测试用例

| ID | 用例名 | 前置 | 步骤 | 预期 | 自动化 |
|:---|:---|:---|:---|:---|:---|
| TC-H01 | 无模型就绪 | 不配置权重 | GET `/health/ready` | HTTP 503，`ready=false` | 是 |
| TC-H02 | 有模型就绪 | 权重可加载 | GET `/health/ready` | HTTP 200，`ready=true` | 是 |
| TC-P01 | 最小预测 | 服务就绪、本地图存在 | POST `/v1/predict` 数组 `[{camera,images:[path]}]` | `code=0`，`results[0].defects[0]` 为该图标列表 | 是 |
| TC-P02 | 多相机多图对齐 | 两路各 2 张，其中一张无框 | 同上结构 | 相机顺序一致；`defects` 长度=images 长度；无框为 `[]` | 是 |
| TC-P03 | 缺相机列表 | 服务启动 | body `{}` 或 `[]` | `code=500`，msg 含相机与图片路径 | 是 |
| TC-P04 | 两路径等价 | 服务就绪 | `/scanly_predict` 与 `/v1/predict` 同 body | 结构相同 | 是 |
| TC-P05 | 文件不存在 | 服务就绪 | 路径虚构 | `读取图片异常` | 是 |
| TC-P07 | infer_type=obb 无权重 | 未配 obb 权重 | POST 对象 `infer_type=obb` | `code=500`，msg 含 OBB模型未加载 | 是 |
| TC-P08 | infer_type 非法 | 服务启动 | `infer_type=seg` | `不支持的推理类型` | 是 |
| TC-P09 | OBB 结果结构 | OBB 权重已加载 | POST `infer_type=obb` | 顶层 `infer_type=obb`；缺陷含 `obb.points` 长度 4，且仍有 `location` | 否（有权重时） |
| TC-P10 | detect 也输出 obb | detect 权重已加载 | POST `infer_type=detect` | 缺陷含 `obb.points` 长度 4，`angle=0` | 否 |
| TC-P11 | 相对模型路径 | `models/cam2-0819.pt` 存在，配置 `obb_model_path=cam2-0819.pt` | 启动服务 | `/health/ready` ready=true，路径解析到 models 下 | 是 |
| TC-P12 | RT-DETRv2 ONNX | `model_path` 为训练导出 `.onnx`，`infer_type=detect`，`class_names`/`imgsz` 与训练一致 | 启动后 POST `/v1/predict` | `code=0`；缺陷 `obb.points` 长度 4、`angle=0`；类名来自 `class_names` | 否（有权重时） |
| TC-P13 | YOLO ONNX | `model_path` 为 Ultralytics/convert 导出的 detect `.onnx` | 启动后 POST `infer_type=detect` | 可加载；结果结构与 `.pt` 一致 | 否（有权重时） |
| TC-P14 | RT-DETR 误配 obb | 仅配 RT-DETR `.onnx` 到 `model_path`，`infer_type=obb` | 启动服务 | 按 detect 加载；ready 对 detect 为 true；日志有警告 | 否 |
| TC-P15 | predict.json 备注 | 打开 `config/predict.json` | 查看 `_readme` | 含 `infer_type` 与路径字段说明 | 否-文档 |
| TC-U01 | 单路上传 | 服务就绪、样张可读 | POST `/v1/predict/upload`，`camera=ch1` + 一个 `files` | `code=0`，`results[0].camera=ch1`，`defects` 长度 1 | 是 |
| TC-U02 | meta 多相机 | 服务就绪 | `meta.cameras` count 2+1，三个 files | `results` 两路；defects 长度分别为 2、1 | 是 |
| TC-U03 | 文件数不一致 | 服务启动 | count 合计 2，只传 1 个 file | `code=500`，msg 含不一致 | 是 |
| TC-U04 | 缺 files | 服务启动 | 不传 files | `code=500`，msg 含必须上传 | 是 |
| TC-U05 | 与路径同图一致 | 同图、同 infer_type | 先路径再 upload | `code=0`；缺陷 name/框大致一致（允许浮点误差） | 否-手工 |
| TC-J01 | OR 命中 | 合成框 高度超、长度不超 | 方案 logic=or | `ng=true` | 是-单测 |
| TC-J02 | AND 未齐 | 仅高度超 | logic=and | `ng=false` | 是-单测 |
| TC-J03 | 仅长度 | 长度超、高度很小 | length_only | `ng=true` | 是-单测 |
| TC-J04 | 通道双 0 | ng 高长度均为 0 | 判定 | `ng=false` | 是-单测 |
| TC-J05 | 仅 NG 看面积 | `ng_area=true`，`label_area=false`，面积不足、尺寸均命中 | 判定 | `ng=false`，`need_label=true` | 是-单测 |
| TC-J06 | 贴标看面积 | `label_area=true`，面积不足 | 判定 | `need_label=false`；未勾 `discharge_area` 时排出仍可 true | 是-单测 |
| TC-F01 | 停用类 | enable=false | 即使模型出框 | 不在 results | 是-单测 |
| TC-F02 | 覆膜过滤 | laminating=true，类未勾覆膜检测 | 过滤 | 丢弃 | 是-单测 |
| TC-F03 | 低置信 | score&lt;confidence | 过滤 | 丢弃 | 是-单测 |
| TC-F04 | 未知类 | name 不在表 | 输出 | 入 results，ng=false，board_ok 不变 | 是-单测 |
| TC-C01 | 标签表 | 启动后 | GET `/v1/labels` | 含 GlueSeam/Tackless 等；屏蔽四类 `is_defect=false`；`Below`/`Shorter`/`BandBroken` 的 `in_train_set=true` | 是 |
| TC-C02 | 查询方案 | 启动后 | GET `/v1/scheme` | `code=0`，含 defects | 是 |
| TC-C03 | 合并配置 | 启动后，可无模型 | PUT 只改某类 confidence | GET 可见新值；其它类不变 | 是 |
| TC-C04 | 配置落盘 | persist=true | PUT 后重启服务 | 新值仍在 | 是 |
| TC-C05 | 仅内存 | persist=false | PUT 后读文件 | 磁盘未改；进程内已改 | 是 |
| TC-C06 | 整表替换缺 defects | replace=true 无 defects | PUT | `code=500` | 是 |
| TC-C07 | 空 body | POST `{}` | 无有效字段 | `code=500` | 是 |
| TC-C08 | 无模型可配置 | 未配权重 | PUT `/v1/scheme` | `code=0` | 是 |
| TC-G01 | Gradio 出图 | 本机打开 `/` | 上传样张运行 | 框图+表格 | 否-手工 |
| TC-G02 | Gradio 中文标签 | 本机含中文字体 | 检出带 cn_name 的框 | 结果图标签中文可读，非方框 | 否-手工 |
| TC-D01 | Swagger 响应结构 | 服务启动 | 打开 `/docs` 展开 POST `/v1/predict` Responses | 200 Example 含 camera、defects 二维数组和缺陷字段 | 否-手工 |
| TC-DIR01 | 目录批量 VOC | `cam2-test` 有图，`predict.json` 可加载 | 跑 `predict_dir.py`（可 `LIMIT=3`） | `Annotations` 有对应 xml；`vis` 有画框图；`summary.json` 计数正确 | 否-手工 |
| TC-DIR02 | 空检出仍写 xml | 某图无框 | 同上 | 该图仍有 xml，`object` 为空 | 否-手工 |
| TC-DIR03 | 跳过 AppleDouble | 目录含 `._*.jpg` | 同上 | 不处理 `._` 文件 | 否-手工 |

# 自动化测试标识

| 层级 | 框架 | 目录（用户要求测试时） | 覆盖 |
|:---|:---|:---|:---|
| 单测 | pytest | `scanly/test/test_defects/test_predict/` | TC-J、TC-F |
| 接口 | pytest + HTTP | 同上 | TC-H、TC-P、TC-C |
| 上传联调 | 脚本（IDE 变量入口） | `scanly/defects/test/test_predict_upload_api.py` | TC-U01（需已启动 serve） |
| 手工 | 浏览器 | — | TC-G01 |

上传联调脚本：先启动 `scanly/defects/predict/serve.py`，再运行 `test_predict_upload_api.py`（默认上传 `defects/samples` 样张）。`client_example.py` 也可作手工接口脚本。
