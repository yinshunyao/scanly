package httpapi

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"

	"scanly/defects/predict_go/internal/config"
	"scanly/defects/predict_go/internal/labels"
	"scanly/defects/predict_go/internal/license"
	"scanly/defects/predict_go/internal/pipeline"
	"scanly/defects/predict_go/internal/scheme"
)

type Server struct {
	svc     *pipeline.Service
	licStat license.Status
}

func New(svc *pipeline.Service, licStat license.Status) http.Handler {
	s := &Server{svc: svc, licStat: licStat}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", s.root)
	mux.HandleFunc("GET /health/ready", s.health)
	mux.HandleFunc("GET /v1/license", s.license)
	mux.HandleFunc("GET /v1/labels", s.labels)
	mux.HandleFunc("GET /v1/scheme", s.getScheme)
	mux.HandleFunc("PUT /v1/scheme", s.putScheme)
	mux.HandleFunc("POST /v1/scheme", s.putScheme)
	mux.HandleFunc("POST /v1/predict", s.predict)
	mux.HandleFunc("POST /scanly_predict", s.predict)
	return mux
}

func (s *Server) root(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, 200, map[string]any{
		"name":   "scanly-defects-go",
		"engine": "scanly-defects-go-v1",
		"docs":   "scanly/defects/predict_go/README.md ；接口契约 scanly/doc/02-dr/缺陷推理/01.HTTP接口文档.md",
		"endpoints": []string{
			"GET /health/ready",
			"GET /v1/license",
			"GET /v1/labels",
			"GET /v1/scheme",
			"PUT|POST /v1/scheme",
			"POST /v1/predict",
			"POST /scanly_predict",
		},
	})
}

func (s *Server) health(w http.ResponseWriter, _ *http.Request) {
	payload := s.svc.Readiness()
	status := 200
	if ready, _ := payload["ready"].(bool); !ready {
		status = 503
	}
	writeJSON(w, status, payload)
}

func (s *Server) license(w http.ResponseWriter, _ *http.Request) {
	st := s.licStat
	writeJSON(w, 200, map[string]any{
		"code":           0,
		"valid":          st.Valid,
		"msg":            st.Msg,
		"model_id":       st.ModelID,
		"expires_at":     st.ExpiresAt,
		"gpus_licensed":  st.GPUsLicensed,
		"gpus_local":     st.GPUsLocal,
	})
}

func (s *Server) labels(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, 200, map[string]any{
		"code":   0,
		"engine": "scanly-defects-go-v1",
		"items":  labels.Payload(),
	})
}

func (s *Server) getScheme(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, 200, s.svc.Scheme.PublicView())
}

func (s *Server) putScheme(w http.ResponseWriter, r *http.Request) {
	body, err := readObject(r)
	if err != nil {
		writeErr(w, "请求body必须是JSON")
		return
	}
	persist := true
	if v, ok := body["persist"]; ok {
		persist = asBool(v)
	}
	replace := asBool(body["replace"])
	overlay := scheme.IncomingOverlay(body)
	if replace {
		if _, ok := body["defects"].([]any); !ok {
			writeErr(w, "replace时必须提供defects数组")
			return
		}
	}
	if !scheme.HasSchemeFields(overlay) {
		writeErr(w, "请求body无有效方案字段")
		return
	}
	view, err := s.svc.Scheme.Apply(overlay, persist, replace)
	if err != nil {
		writeErr(w, "写入方案文件失败")
		return
	}
	writeJSON(w, 200, view)
}

func (s *Server) predict(w http.ResponseWriter, r *http.Request) {
	cameras, extras, err := parsePredict(r)
	if err != nil {
		writeErr(w, err.Error())
		return
	}
	kind := config.NormalizeInferType(fmt.Sprint(extras["infer_type"]))
	if extras["infer_type"] == nil || strings.TrimSpace(fmt.Sprint(extras["infer_type"])) == "" {
		kind = ""
	} else if kind == "" {
		writeErr(w, "不支持的推理类型")
		return
	}
	if kind == "" {
		ready := s.svc.Readiness()
		if ok, _ := ready["ready"].(bool); !ok {
			if msg, _ := ready["msg"].(string); strings.Contains(msg, "许可") || strings.Contains(msg, "GPU") {
				writeErr(w, msg)
				return
			}
		}
	} else if !s.svc.Ready(kind) {
		if !s.svc.LicenseOK {
			writeErr(w, s.svc.LicenseMsg)
			return
		}
		if kind == "obb" {
			writeErr(w, "OBB模型未加载")
			return
		}
		writeErr(w, "检测模型未加载")
		return
	}
	result, err := s.svc.PredictCameras(cameras, extras)
	if err != nil {
		msg := err.Error()
		if strings.Contains(msg, "不存在") || strings.Contains(msg, "解码") || strings.Contains(msg, "图片") {
			writeErr(w, "读取图片异常")
			return
		}
		writeErr(w, msg)
		return
	}
	log.Printf("predict cameras=%d", len(cameras))
	writeJSON(w, 200, result)
}

func parsePredict(r *http.Request) ([]map[string]any, map[string]any, error) {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, nil, fmt.Errorf("请求body必须是JSON")
	}
	var anyJSON any
	if err := json.Unmarshal(raw, &anyJSON); err != nil {
		return nil, nil, fmt.Errorf("请求body必须是JSON")
	}
	extras := map[string]any{}
	var cameras any
	switch body := anyJSON.(type) {
	case []any:
		cameras = body
	case map[string]any:
		extras = body
		cameras = body["cameras"]
		if cameras == nil {
			cameras = body["items"]
		}
		if cameras == nil && body["camera"] != nil {
			cameras = []any{map[string]any{"camera": body["camera"], "images": body["images"]}}
		}
	default:
		return nil, nil, fmt.Errorf("请求body必须是JSON")
	}
	list, ok := cameras.([]any)
	if !ok || len(list) == 0 {
		return nil, nil, fmt.Errorf("请求body必须包含相机与图片路径")
	}
	out := make([]map[string]any, 0, len(list))
	for _, item := range list {
		m, ok := item.(map[string]any)
		if !ok {
			return nil, nil, fmt.Errorf("请求body必须包含相机与图片路径")
		}
		out = append(out, m)
	}
	return out, extras, nil
}

func readObject(r *http.Request) (map[string]any, error) {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil || body == nil {
		return nil, fmt.Errorf("not object")
	}
	return body, nil
}

func writeErr(w http.ResponseWriter, msg string) {
	writeJSON(w, 200, map[string]any{"code": 500, "msg": msg})
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(payload)
}

func asBool(v any) bool {
	b, _ := v.(bool)
	return b
}
