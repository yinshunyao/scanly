package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

type Predict struct {
	Host               string  `json:"host"`
	Port               int     `json:"port"`
	InferType          string  `json:"infer_type"`
	ModelEncPath       string  `json:"model_enc_path"`
	OBBModelEncPath    string  `json:"obb_model_enc_path"`
	DetectMetaPath     string  `json:"detect_meta_path"`
	OBBMetaPath        string  `json:"obb_meta_path"`
	LicensePath        string  `json:"license_path"`
	SchemePath         string  `json:"scheme_path"`
	ORTLibPath         string  `json:"ort_lib_path"`
	Device             string  `json:"device"`
	TrtEngineCachePath string  `json:"trt_engine_cache_path"`
	TrtFp16Enable      *bool   `json:"trt_fp16_enable"`
	ClipSize           int     `json:"clip_size"`
	OverlapSize        int     `json:"overlap_size"`
	Conf               float64 `json:"conf"`
	Iou                float64 `json:"iou"`
	MMPerPx            float64 `json:"mm_per_px"`
}

func FindRoot() (string, error) {
	if v := strings.TrimSpace(os.Getenv("SCANLY_PREDICT_GO_ROOT")); v != "" {
		return filepath.Clean(v), nil
	}
	var candidates []string
	if exe, err := os.Executable(); err == nil {
		candidates = append(candidates, filepath.Dir(exe))
	}
	if wd, err := os.Getwd(); err == nil {
		candidates = append(candidates, wd)
	}
	if _, file, _, ok := runtime.Caller(0); ok {
		candidates = append(candidates, filepath.Clean(filepath.Join(filepath.Dir(file), "..", "..")))
	}
	for _, c := range candidates {
		if st, err := os.Stat(filepath.Join(c, "config", "predict.json")); err == nil && !st.IsDir() {
			return c, nil
		}
	}
	return "", fmt.Errorf("找不到 predict_go 根目录（可设环境变量 SCANLY_PREDICT_GO_ROOT）")
}

func Load(root string) (Predict, error) {
	path := filepath.Join(root, "config", "predict.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		return Predict{}, err
	}
	var cfg Predict
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return Predict{}, err
	}
	if strings.TrimSpace(cfg.Host) == "" {
		cfg.Host = "0.0.0.0"
	}
	if cfg.Port <= 0 {
		cfg.Port = 37871
	}
	if strings.TrimSpace(cfg.InferType) == "" {
		cfg.InferType = "obb"
	}
	if cfg.Conf <= 0 {
		cfg.Conf = 0.25
	}
	if cfg.Iou <= 0 {
		cfg.Iou = 0.45
	}
	if cfg.MMPerPx <= 0 {
		cfg.MMPerPx = 0.03
	}
	if strings.TrimSpace(cfg.Device) == "" {
		cfg.Device = "cpu"
	}
	if strings.TrimSpace(cfg.TrtEngineCachePath) == "" {
		cfg.TrtEngineCachePath = "models/trt_cache"
	}
	cfg.ModelEncPath = resolve(root, cfg.ModelEncPath)
	cfg.OBBModelEncPath = resolve(root, cfg.OBBModelEncPath)
	cfg.DetectMetaPath = resolve(root, cfg.DetectMetaPath)
	cfg.OBBMetaPath = resolve(root, cfg.OBBMetaPath)
	cfg.LicensePath = resolve(root, cfg.LicensePath)
	cfg.SchemePath = resolve(root, cfg.SchemePath)
	cfg.ORTLibPath = resolve(root, cfg.ORTLibPath)
	cfg.TrtEngineCachePath = resolve(root, cfg.TrtEngineCachePath)
	return cfg, nil
}

func (c Predict) TrtFP16() bool {
	if c.TrtFp16Enable == nil {
		return true
	}
	return *c.TrtFp16Enable
}

func resolve(root, raw string) string {
	text := strings.TrimSpace(raw)
	if text == "" {
		return ""
	}
	if filepath.IsAbs(text) {
		return filepath.Clean(text)
	}
	return filepath.Clean(filepath.Join(root, text))
}

func NormalizeInferType(raw string) string {
	text := strings.ToLower(strings.TrimSpace(raw))
	if text == "" {
		return "obb"
	}
	switch text {
	case "obb", "oriented", "rotate", "rotated":
		return "obb"
	case "detect", "det", "bbox":
		return "detect"
	default:
		return ""
	}
}
