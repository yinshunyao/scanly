package pipeline

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"scanly/defects/predict_go/internal/config"
	"scanly/defects/predict_go/internal/gpu"
	"scanly/defects/predict_go/internal/infer"
	"scanly/defects/predict_go/internal/scheme"
)

type Service struct {
	mu          sync.Mutex
	cfg         config.Predict
	defaultType string
	detect      *infer.Engine
	obb         *infer.Engine
	Scheme      *scheme.Store
	LicenseOK   bool
	LicenseMsg  string
	LocalGPUs   []gpu.Info
	LicensedGPU []gpu.Info
}

func New(cfg config.Predict, sch *scheme.Store) *Service {
	return &Service{
		cfg:         cfg,
		defaultType: config.NormalizeInferType(cfg.InferType),
		Scheme:      sch,
	}
}

func (s *Service) SetEngines(detect, obb *infer.Engine) {
	s.detect = detect
	s.obb = obb
}

func (s *Service) DetectEngine() *infer.Engine { return s.detect }
func (s *Service) OBBEngine() *infer.Engine    { return s.obb }

func (s *Service) Ready(kind string) bool {
	if kind == "" {
		kind = s.defaultType
	}
	if kind == "obb" {
		return s.obb != nil
	}
	return s.detect != nil
}

func (s *Service) Readiness() map[string]any {
	ready := s.LicenseOK && s.Ready(s.defaultType)
	msg := "ok"
	if !s.LicenseOK {
		msg = s.LicenseMsg
		if msg == "" {
			msg = "许可证无效"
		}
	} else if !ready {
		if s.defaultType == "obb" {
			msg = "OBB模型未加载"
		} else {
			msg = "检测模型未加载"
		}
	}
	return map[string]any{
		"ready":           ready,
		"model_loaded":    ready,
		"infer_type":      s.defaultType,
		"detect_loaded":   s.detect != nil,
		"obb_loaded":      s.obb != nil,
		"model_path":      s.cfg.ModelEncPath,
		"obb_model_path":  s.cfg.OBBModelEncPath,
		"license_ok":      s.LicenseOK,
		"msg":             msg,
	}
}

func (s *Service) PredictCameras(cameras []map[string]any, extras map[string]any) (map[string]any, error) {
	kind := config.NormalizeInferType(fmt.Sprint(extras["infer_type"]))
	if extras["infer_type"] == nil || fmt.Sprint(extras["infer_type"]) == "" {
		kind = s.defaultType
	}
	if kind == "" {
		return nil, fmt.Errorf("不支持的推理类型")
	}
	if !s.LicenseOK {
		msg := s.LicenseMsg
		if msg == "" {
			msg = "许可证无效"
		}
		return nil, fmt.Errorf("%s", msg)
	}
	if !s.Ready(kind) {
		if kind == "obb" {
			return nil, fmt.Errorf("OBB模型未加载")
		}
		return nil, fmt.Errorf("检测模型未加载")
	}
	laminating := asBool(extras["laminating"])
	mm := extras["mm_per_px"]
	if calib, ok := extras["calib"].(map[string]any); ok && mm == nil {
		mm = calib["mm_per_px"]
	}
	var overlay map[string]any
	if sch, ok := extras["scheme"].(map[string]any); ok {
		overlay = sch
	}
	if name, ok := extras["scheme_name"].(string); ok && name != "" {
		if overlay == nil {
			overlay = map[string]any{"scheme_name": name}
		} else if _, exists := overlay["scheme_name"]; !exists {
			overlay["scheme_name"] = name
		}
	}
	boardID := fmt.Sprint(extras["board_id"])
	if boardID == "<nil>" {
		boardID = ""
	}
	grouped := make([]map[string]any, 0, len(cameras))
	ngAny := false
	t0 := time.Now()
	for _, item := range cameras {
		cam := strings.TrimSpace(fmt.Sprint(item["camera"]))
		if cam == "" || cam == "<nil>" {
			return nil, fmt.Errorf("camera不能为空")
		}
		images, ok := item["images"].([]any)
		if !ok {
			return nil, fmt.Errorf("images必须是数组")
		}
		slots := make([][]map[string]any, 0, len(images))
			for _, rawPath := range images {
				path := strings.TrimSpace(fmt.Sprint(rawPath))
				if path == "" || path == "<nil>" {
					return nil, fmt.Errorf("图片路径不能为空")
				}
			one, err := s.predictPath(path, boardID, cam, laminating, asFloatPtr(mm), overlay, kind)
			if err != nil {
				return nil, err
			}
			res, _ := one["results"].([]map[string]any)
			if res == nil {
				res = []map[string]any{}
			}
			slots = append(slots, res)
			if ok, _ := one["image_ok"].(bool); !ok {
				ngAny = true
			}
		}
		grouped = append(grouped, map[string]any{"camera": cam, "defects": slots})
	}
	return map[string]any{
		"code":        0,
		"msg":         "ok",
		"engine":      "scanly-defects-go-v1",
		"infer_type":  kind,
		"board_id":    boardID,
		"board_ok":    !ngAny,
		"laminating":  laminating,
		"elapsed_ms":  float64(time.Since(t0).Milliseconds()),
		"results":     grouped,
	}, nil
}

func (s *Service) predictPath(path, boardID, cameraID string, laminating bool, mm *float64, overlay map[string]any, kind string) (map[string]any, error) {
	img, w, h, err := infer.LoadImage(path)
	if err != nil {
		return nil, err
	}
	nrgba := infer.ToNRGBA(img)
	engine := s.obb
	if kind == "detect" {
		engine = s.detect
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	sch := scheme.Merge(s.Scheme.Snapshot(), overlay)
	scale := s.cfg.MMPerPx
	if v, ok := sch["mm_per_px"].(float64); ok && v > 0 {
		scale = v
	}
	if mm != nil {
		scale = *mm
	}
	tiles := infer.Tiles(h, w, s.cfg.ClipSize, s.cfg.OverlapSize)
	var raw []infer.Det
	for _, t := range tiles {
		tile := infer.CropNRGBA(nrgba, t[0], t[1], t[2], t[3])
		dets, err := engine.Predict(tile, t[0], t[1])
		if err != nil {
			return nil, err
		}
		raw = append(raw, dets...)
	}
	raw = infer.NMS(raw, s.cfg.Iou)
	var results []map[string]any
	ngCount := 0
	for _, det := range raw {
		obb := det.OBB
		judged := scheme.JudgeBox(det.Name, det.Score, det.XYXY[0], det.XYXY[1], det.XYXY[2], det.XYXY[3], sch, scale, laminating, obb.Width, obb.Height, map[string]any{
			"infer_type": kind,
			"obb":        obb,
		})
		if judged == nil {
			continue
		}
		results = append(results, judged)
		if ng, _ := judged["ng"].(bool); ng {
			ngCount++
		}
	}
	if results == nil {
		results = []map[string]any{}
	}
	return map[string]any{
		"image_ok": ngCount == 0,
		"results":  results,
		"board_id": boardID,
		"camera":   cameraID,
	}, nil
}

func asBool(v any) bool {
	b, _ := v.(bool)
	return b
}

func asFloatPtr(v any) *float64 {
	if v == nil {
		return nil
	}
	switch t := v.(type) {
	case float64:
		return &t
	case int:
		f := float64(t)
		return &f
	}
	return nil
}
