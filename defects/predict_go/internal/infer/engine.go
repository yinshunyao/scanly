package infer

import (
	"encoding/json"
	"fmt"
	"image"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

type Meta struct {
	ModelID   string            `json:"model_id"`
	Task      string            `json:"task"`
	Imgsz     int               `json:"imgsz"`
	Names     map[string]string `json:"names"`
	Precision string            `json:"precision"`
	Half      bool              `json:"half"`
}

type OBB struct {
	Points [][]int `json:"points"`
	CX     float64 `json:"cx"`
	CY     float64 `json:"cy"`
	Width  float64 `json:"width"`
	Height float64 `json:"height"`
	Angle  float64 `json:"angle"`
}

type Det struct {
	Name  string
	Score float64
	XYXY  [4]float64
	OBB   OBB
}

type Engine struct {
	mu         sync.Mutex
	session    *ort.DynamicAdvancedSession
	inputName  string
	outputName string
	inputFP16  bool
	outputFP16 bool
	imgsz      int
	task       string
	names      map[int]string
	conf       float64
	iou        float64
}

func LoadMeta(path string) (Meta, error) {
	var meta Meta
	raw, err := os.ReadFile(path)
	if err != nil {
		return meta, err
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		return meta, err
	}
	if meta.Imgsz <= 0 {
		meta.Imgsz = 640
	}
	if strings.TrimSpace(meta.Task) == "" {
		meta.Task = "obb"
	}
	return meta, nil
}

func EnsureORT(libPath string) error {
	if ort.IsInitialized() {
		return nil
	}
	path := strings.TrimSpace(libPath)
	if path == "" {
		path = defaultORTName()
	}
	ort.SetSharedLibraryPath(path)
	return ort.InitializeEnvironment()
}

func defaultORTName() string {
	switch runtime.GOOS {
	case "windows":
		return "onnxruntime.dll"
	case "darwin":
		return "libonnxruntime.dylib"
	default:
		return "libonnxruntime.so"
	}
}

func NewEngine(onnx []byte, meta Meta, device string, conf, iou float64, trtCachePath string, trtFP16 bool) (*Engine, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer opts.Destroy()
	if wantTensorRT(device) && trtFP16 && metaIsFP16(meta) {
		fmt.Println("ONNX 已是 FP16，关闭额外 TensorRT FP16（叠半精度易 NaN）")
		trtFP16 = false
	}
	if wantTensorRT(device) {
		if err := appendTensorRT(opts, trtCachePath, trtFP16); err != nil {
			return nil, err
		}
		// TRT 未覆盖算子回退到 CUDA（与 ORT 推荐顺序一致）
		if err := appendCUDA(opts, device); err != nil {
			return nil, fmt.Errorf("TensorRT 已启用，但附加 CUDA EP 失败: %w", err)
		}
		fmt.Printf("ONNX Runtime TensorRT EP 已启用 fp16=%v cache=%s device=%s\n", trtFP16, trtCachePath, device)
	} else if wantCUDA(device) {
		if err := appendCUDA(opts, device); err != nil {
			return nil, err
		}
		fmt.Printf("ONNX Runtime CUDA EP 已启用 device=%s\n", device)
	}
	inputs, outputs, err := ort.GetInputOutputInfoWithONNXData(onnx)
	if err != nil {
		return nil, err
	}
	if len(inputs) == 0 || len(outputs) == 0 {
		return nil, fmt.Errorf("ONNX 无输入或输出")
	}
	inName := inputs[0].Name
	outName := outputs[0].Name
	inputFP16 := inputs[0].DataType == ort.TensorElementDataTypeFloat16
	outputFP16 := outputs[0].DataType == ort.TensorElementDataTypeFloat16
	if inputs[0].DataType != ort.TensorElementDataTypeFloat && !inputFP16 {
		return nil, fmt.Errorf("不支持的 ONNX 输入类型: %s", inputs[0].DataType)
	}
	if outputs[0].DataType != ort.TensorElementDataTypeFloat && !outputFP16 {
		return nil, fmt.Errorf("不支持的 ONNX 输出类型: %s", outputs[0].DataType)
	}
	session, err := ort.NewDynamicAdvancedSessionWithONNXData(onnx, []string{inName}, []string{outName}, opts)
	if err != nil {
		return nil, err
	}
	names := map[int]string{}
	for k, v := range meta.Names {
		var id int
		if _, err := fmt.Sscanf(k, "%d", &id); err == nil {
			names[id] = v
		}
	}
	imgsz := meta.Imgsz
	if imgsz <= 0 {
		imgsz = 640
	}
	task := strings.ToLower(strings.TrimSpace(meta.Task))
	if task == "" {
		task = "obb"
	}
	if conf <= 0 {
		conf = 0.25
	}
	if iou <= 0 {
		iou = 0.45
	}
	if inputFP16 || outputFP16 {
		fmt.Printf("ONNX 精度: input=%s output=%s (meta precision=%s)\n",
			inputs[0].DataType, outputs[0].DataType, strings.TrimSpace(meta.Precision))
	}
	return &Engine{
		session:    session,
		inputName:  inName,
		outputName: outName,
		inputFP16:  inputFP16,
		outputFP16: outputFP16,
		imgsz:      imgsz,
		task:       task,
		names:      names,
		conf:       conf,
		iou:        iou,
	}, nil
}

func appendCUDA(opts *ort.SessionOptions, device string) error {
	cudaOpts, err := ort.NewCUDAProviderOptions()
	if err != nil {
		return fmt.Errorf("创建 CUDA Provider 失败（device=%s）: %w", device, err)
	}
	// V100 / 新驱动上关闭 TF32、放宽 workspace，降低 Conv 选错 algo 概率
	_ = cudaOpts.Update(map[string]string{
		"device_id":                    "0",
		"use_tf32":                     "0",
		"cudnn_conv_algo_search":       "DEFAULT",
		"cudnn_conv_use_max_workspace": "1",
	})
	if err := opts.AppendExecutionProviderCUDA(cudaOpts); err != nil {
		_ = cudaOpts.Destroy()
		return fmt.Errorf("启用 CUDA EP 失败（device=%s，不会回退 CPU）: %w", device, err)
	}
	_ = cudaOpts.Destroy()
	return nil
}

func appendTensorRT(opts *ort.SessionOptions, cachePath string, fp16 bool) error {
	trtOpts, err := ort.NewTensorRTProviderOptions()
	if err != nil {
		return fmt.Errorf("创建 TensorRT Provider 失败（ORT 可能未编入 TRT）: %w", err)
	}
	cache := strings.TrimSpace(cachePath)
	if cache == "" {
		cache = "models/trt_cache"
	}
	if err := os.MkdirAll(cache, 0o755); err != nil {
		_ = trtOpts.Destroy()
		return fmt.Errorf("创建 TensorRT 缓存目录失败 %s: %w", cache, err)
	}
	fp16Flag := "False"
	if fp16 {
		fp16Flag = "True"
	}
	if err := trtOpts.Update(map[string]string{
		"device_id":                "0",
		"trt_fp16_enable":          fp16Flag,
		"trt_engine_cache_enable":  "True",
		"trt_engine_cache_path":    cache,
	}); err != nil {
		_ = trtOpts.Destroy()
		return fmt.Errorf("配置 TensorRT Provider 失败: %w", err)
	}
	if err := opts.AppendExecutionProviderTensorRT(trtOpts); err != nil {
		_ = trtOpts.Destroy()
		return fmt.Errorf("启用 TensorRT EP 失败: %w", err)
	}
	_ = trtOpts.Destroy()
	return nil
}

func wantCUDA(device string) bool {
	d := strings.ToLower(strings.TrimSpace(device))
	return d == "cuda" || d == "gpu" || strings.HasPrefix(d, "cuda:")
}

func wantTensorRT(device string) bool {
	d := strings.ToLower(strings.TrimSpace(device))
	return d == "tensorrt" || d == "trt" || d == "trt_fp16"
}

func metaIsFP16(meta Meta) bool {
	p := strings.ToLower(strings.TrimSpace(meta.Precision))
	if p == "fp16" || p == "float16" || p == "half" || p == "16" {
		return true
	}
	return meta.Half
}

func (e *Engine) Close() {
	if e == nil || e.session == nil {
		return
	}
	_ = e.session.Destroy()
	e.session = nil
}

func (e *Engine) Predict(img image.Image, ox, oy int) ([]Det, error) {
	if e == nil || e.session == nil {
		return nil, fmt.Errorf("模型未加载")
	}
	data, meta := letterboxCHW(img, e.imgsz)
	shape := ort.NewShape(1, 3, int64(e.imgsz), int64(e.imgsz))
	var in ort.Value
	var err error
	if e.inputFP16 {
		in, err = ort.NewCustomDataTensor(shape, float32SliceToFloat16Bytes(data), ort.TensorElementDataTypeFloat16)
	} else {
		in, err = ort.NewTensor[float32](shape, data)
	}
	if err != nil {
		return nil, err
	}
	defer in.Destroy()
	outputs := []ort.Value{nil}
	e.mu.Lock()
	err = e.session.Run([]ort.Value{in}, outputs)
	e.mu.Unlock()
	if err != nil {
		return nil, err
	}
	if len(outputs) == 0 || outputs[0] == nil {
		return nil, fmt.Errorf("ONNX 无输出")
	}
	defer outputs[0].Destroy()

	outData, outShape, err := readOutputFloat32(outputs[0], e.outputFP16)
	if err != nil {
		return nil, err
	}
	dets := decodeYOLO(outData, outShape, e.task, e.names, e.conf, meta, ox, oy)
	return NMS(dets, e.iou), nil
}

func readOutputFloat32(v ort.Value, expectFP16 bool) ([]float32, ort.Shape, error) {
	if t, ok := v.(*ort.Tensor[float32]); ok {
		return t.GetData(), t.GetShape(), nil
	}
	if t, ok := v.(*ort.CustomDataTensor); ok {
		dt := ort.TensorElementDataType(t.DataType())
		if dt == ort.TensorElementDataTypeFloat16 || expectFP16 {
			return float16BytesToFloat32Slice(t.GetData()), t.GetShape(), nil
		}
		return nil, nil, fmt.Errorf("ONNX 输出 CustomData 类型不支持: %s", dt)
	}
	return nil, nil, fmt.Errorf("ONNX 输出类型不支持")
}

func decodeYOLO(data []float32, shape ort.Shape, task string, names map[int]string, conf float64, lb letterboxMeta, ox, oy int) []Det {
	if len(shape) < 2 {
		return nil
	}
	var channels, anchors int
	channelFirst := true
	switch len(shape) {
	case 3:
		a, b, c := int(shape[0]), int(shape[1]), int(shape[2])
		if a == 1 {
			if b <= c && b < 128 {
				channels, anchors = b, c
				channelFirst = true
			} else {
				anchors, channels = b, c
				channelFirst = false
			}
		} else {
			anchors, channels = a, b
			channelFirst = false
		}
	case 2:
		anchors, channels = int(shape[0]), int(shape[1])
		channelFirst = false
	default:
		return nil
	}
	isOBB := task == "obb"
	minC := 5
	if isOBB {
		minC = 6
	}
	if channels < minC || anchors <= 0 {
		return nil
	}
	nc := channels - 4
	if isOBB {
		nc = channels - 5
	}
	if nc < 1 {
		nc = 1
	}
	at := func(ch, anc int) float32 {
		if channelFirst {
			return data[ch*anchors+anc]
		}
		return data[anc*channels+ch]
	}
	out := make([]Det, 0, 64)
	for i := 0; i < anchors; i++ {
		best := 0
		bestScore := float64(at(4, i))
		if nc > 1 {
			bestScore = 0
			for c := 0; c < nc; c++ {
				s := float64(at(4+c, i))
				if s > bestScore {
					bestScore = s
					best = c
				}
			}
		}
		if bestScore < conf {
			continue
		}
		cx := (float64(at(0, i)) - lb.padX) / lb.ratio
		cy := (float64(at(1, i)) - lb.padY) / lb.ratio
		bw := float64(at(2, i)) / lb.ratio
		bh := float64(at(3, i)) / lb.ratio
		angle := 0.0
		if isOBB {
			angle = float64(at(channels-1, i))
		}
		if !finite(cx) || !finite(cy) || !finite(bw) || !finite(bh) || !finite(angle) || !finite(bestScore) {
			continue
		}
		cx += float64(ox)
		cy += float64(oy)
		name := names[best]
		if name == "" {
			name = fmt.Sprintf("%d", best)
		}
		var det Det
		det.Name = name
		det.Score = bestScore
		if isOBB {
			pts := xywhrToPoints(cx, cy, bw, bh, angle)
			if pts == nil {
				continue
			}
			det.OBB = OBB{
				Points: pts,
				CX:     round2(cx),
				CY:     round2(cy),
				Width:  round2(bw),
				Height: round2(bh),
				Angle:  round6(angle),
			}
			x1, y1, x2, y2 := aabbOf(pts)
			det.XYXY = [4]float64{x1, y1, x2, y2}
		} else {
			x1 := cx - bw/2
			y1 := cy - bh/2
			x2 := cx + bw/2
			y2 := cy + bh/2
			det.XYXY = [4]float64{x1, y1, x2, y2}
			det.OBB = xyxyToOBB(x1, y1, x2, y2)
		}
		out = append(out, det)
		if len(out) >= 3000 {
			break
		}
	}
	return out
}

func xyxyToOBB(x1, y1, x2, y2 float64) OBB {
	left := math.Min(x1, x2)
	right := math.Max(x1, x2)
	top := math.Min(y1, y2)
	bottom := math.Max(y1, y2)
	w := math.Max(right-left, 0)
	h := math.Max(bottom-top, 0)
	return OBB{
		Points: [][]int{
			{int(math.Round(left)), int(math.Round(top))},
			{int(math.Round(right)), int(math.Round(top))},
			{int(math.Round(right)), int(math.Round(bottom))},
			{int(math.Round(left)), int(math.Round(bottom))},
		},
		CX:     round2((left + right) / 2),
		CY:     round2((top + bottom) / 2),
		Width:  round2(w),
		Height: round2(h),
		Angle:  0,
	}
}

func xywhrToPoints(cx, cy, w, h, angle float64) [][]int {
	cosV := math.Cos(angle)
	sinV := math.Sin(angle)
	vx1 := (w / 2) * cosV
	vy1 := (w / 2) * sinV
	vx2 := -(h / 2) * sinV
	vy2 := (h / 2) * cosV
	pts := [][2]float64{
		{cx + vx1 + vx2, cy + vy1 + vy2},
		{cx + vx1 - vx2, cy + vy1 - vy2},
		{cx - vx1 - vx2, cy - vy1 - vy2},
		{cx - vx1 + vx2, cy - vy1 + vy2},
	}
	out := make([][]int, 4)
	for i, p := range pts {
		if !finite(p[0]) || !finite(p[1]) {
			return nil
		}
		out[i] = []int{int(math.Round(p[0])), int(math.Round(p[1]))}
	}
	return out
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

func aabbOf(pts [][]int) (float64, float64, float64, float64) {
	x1, y1 := float64(pts[0][0]), float64(pts[0][1])
	x2, y2 := x1, y1
	for _, p := range pts[1:] {
		x := float64(p[0])
		y := float64(p[1])
		if x < x1 {
			x1 = x
		}
		if y < y1 {
			y1 = y
		}
		if x > x2 {
			x2 = x
		}
		if y > y2 {
			y2 = y
		}
	}
	return x1, y1, x2, y2
}

func round2(v float64) float64  { return math.Round(v*100) / 100 }
func round6(v float64) float64  { return math.Round(v*1e6) / 1e6 }

func AxisStarts(size, clip, overlap int) []int {
	if clip <= 0 || size <= clip {
		return []int{0}
	}
	step := clip - overlap
	if step < 1 {
		step = 1
	}
	last := size - clip
	var starts []int
	for x := 0; x <= last; x += step {
		starts = append(starts, x)
	}
	if starts[len(starts)-1] != last {
		starts = append(starts, last)
	}
	return starts
}

func Tiles(h, w, clipSize, overlap int) [][4]int {
	if clipSize <= 0 {
		return [][4]int{{0, 0, w, h}}
	}
	clipW := clipSize
	if clipW > w {
		clipW = w
	}
	clipH := clipSize
	if clipH > h {
		clipH = h
	}
	var tiles [][4]int
	for _, y0 := range AxisStarts(h, clipH, overlap) {
		for _, x0 := range AxisStarts(w, clipW, overlap) {
			tiles = append(tiles, [4]int{x0, y0, x0 + clipW, y0 + clipH})
		}
	}
	return tiles
}

func XYXYToOBB(x1, y1, x2, y2 float64) OBB {
	return xyxyToOBB(x1, y1, x2, y2)
}
