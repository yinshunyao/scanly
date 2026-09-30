package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"scanly/defects/predict_go/internal/config"
	"scanly/defects/predict_go/internal/cryptoenc"
	"scanly/defects/predict_go/internal/httpapi"
	"scanly/defects/predict_go/internal/infer"
	"scanly/defects/predict_go/internal/license"
	"scanly/defects/predict_go/internal/pipeline"
	"scanly/defects/predict_go/internal/scheme"
)

// 发布双二进制时用 -ldflags "-X main.ortFlavor=ort118" 等注入；默认 ort-new（API24 / ORT≥1.24）
var ortFlavor = "ort-new"

func main() {
	log.SetFlags(log.LstdFlags)
	log.Printf("scanly-defects-go flavor=%s goos=%s goarch=%s", ortFlavor, runtime.GOOS, runtime.GOARCH)
	root, err := config.FindRoot()
	if err != nil {
		log.Fatal(err)
	}
	cfg, err := config.Load(root)
	if err != nil {
		log.Fatal(err)
	}
	sch, err := scheme.Load(cfg.SchemePath)
	if err != nil {
		log.Fatalf("加载方案失败: %v", err)
	}
	svc := pipeline.New(cfg, sch)

	lic, st := license.LoadAndVerify(cfg.LicensePath)
	svc.LicenseOK = st.Valid
	svc.LicenseMsg = st.Msg
	svc.LocalGPUs = st.GPUsLocal
	svc.LicensedGPU = st.GPUsLicensed
	log.Printf("license valid=%v msg=%s", st.Valid, st.Msg)

	if st.Valid && lic != nil {
		if err := infer.EnsureORT(ortLib(cfg, root)); err != nil {
			log.Printf("ONNX Runtime 初始化失败: %v", err)
		} else {
			detectPath := cfg.ModelEncPath
			obbPath := cfg.OBBModelEncPath
			if config.NormalizeInferType(cfg.InferType) == "obb" && obbPath == "" && detectPath != "" {
				obbPath = detectPath
				detectPath = ""
			}
			var detectEng, obbEng *infer.Engine
			if detectPath != "" {
				eng, err := loadEncrypted(detectPath, cfg.DetectMetaPath, "detect", lic.AESKey, cfg)
				if err != nil {
					log.Printf("检测模型未加载: %v", err)
				} else {
					detectEng = eng
					log.Printf("已加载 detect: %s", detectPath)
				}
			}
			if obbPath != "" {
				eng, err := loadEncrypted(obbPath, cfg.OBBMetaPath, "obb", lic.AESKey, cfg)
				if err != nil {
					log.Printf("OBB模型未加载: %v", err)
				} else {
					obbEng = eng
					log.Printf("已加载 obb: %s", obbPath)
				}
			}
			svc.SetEngines(detectEng, obbEng)
		}
	}

	addr := fmt.Sprintf("%s:%d", cfg.Host, cfg.Port)
	log.Printf("scanly-defects-go 监听 %s （无 Gradio）", addr)
	if err := http.ListenAndServe(addr, httpapi.New(svc, st)); err != nil {
		log.Fatal(err)
	}
}

func ortLib(cfg config.Predict, root string) string {
	if strings.TrimSpace(cfg.ORTLibPath) != "" {
		return cfg.ORTLibPath
	}
	dir := defaultORTDir(root)
	for _, name := range defaultORTNames() {
		p := filepath.Join(dir, name)
		if fileExists(p) {
			return p
		}
	}
	return ""
}

// defaultORTDir：Linux/macOS 用 third_party/onnxruntime；Windows 用 third_party_win/onnxruntime。
func defaultORTDir(root string) string {
	if runtime.GOOS == "windows" {
		return filepath.Join(root, "third_party_win", "onnxruntime")
	}
	return filepath.Join(root, "third_party", "onnxruntime")
}

func defaultORTNames() []string {
	switch runtime.GOOS {
	case "windows":
		return []string{"onnxruntime.dll"}
	case "darwin":
		return []string{"libonnxruntime.dylib"}
	default:
		return []string{"libonnxruntime.so"}
	}
}

func loadEncrypted(encPath, metaPath, task string, key []byte, cfg config.Predict) (*infer.Engine, error) {
	blob, err := os.ReadFile(encPath)
	if err != nil {
		return nil, err
	}
	plain, err := cryptoenc.Decrypt(blob, key)
	if err != nil {
		return nil, err
	}
	meta, err := infer.LoadMeta(metaPath)
	if err != nil {
		meta = infer.Meta{Task: task, Imgsz: 640}
	}
	if strings.TrimSpace(meta.Task) == "" {
		meta.Task = task
	}
	return infer.NewEngine(plain, meta, cfg.Device, cfg.Conf, cfg.Iou, cfg.TrtEngineCachePath, cfg.TrtFP16())
}

func fileExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && !st.IsDir()
}
