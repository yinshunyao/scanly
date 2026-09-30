package scheme

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"scanly/defects/predict_go/internal/labels"
)

type Store struct {
	mu   sync.Mutex
	path string
	raw  map[string]any
}

func Load(path string) (*Store, error) {
	raw, err := readJSON(path)
	if err != nil {
		return nil, err
	}
	return &Store{path: path, raw: raw}, nil
}

func (s *Store) Snapshot() map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	return cloneMap(s.raw)
}

func (s *Store) PublicView() map[string]any {
	view := publicView(s.Snapshot())
	view["code"] = 0
	view["msg"] = "ok"
	return view
}

func (s *Store) Apply(overlay map[string]any, persist, replace bool) (map[string]any, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var stored map[string]any
	if replace {
		stored = storageDict(map[string]any{
			"scheme_name": firstNonEmpty(asString(overlay["scheme_name"]), asString(s.raw["scheme_name"])),
			"mm_per_px":   overlay["mm_per_px"],
			"channels":    overlay["channels"],
			"defects":     overlay["defects"],
		})
		if overlay["mm_per_px"] == nil {
			stored["mm_per_px"] = s.raw["mm_per_px"]
		}
		if overlay["channels"] == nil {
			stored["channels"] = s.raw["channels"]
		}
	} else {
		stored = storageDict(Merge(s.raw, overlay))
	}
	if persist {
		if err := writeJSON(s.path, stored); err != nil {
			return nil, err
		}
	}
	s.raw = stored
	view := publicView(stored)
	view["code"] = 0
	view["msg"] = "ok"
	view["persisted"] = persist
	return view, nil
}

func IncomingOverlay(body map[string]any) map[string]any {
	out := map[string]any{}
	for k, v := range body {
		switch k {
		case "code", "msg", "engine", "persist", "replace", "persisted":
			continue
		default:
			out[k] = v
		}
	}
	return out
}

func HasSchemeFields(overlay map[string]any) bool {
	if asString(overlay["scheme_name"]) != "" {
		return true
	}
	if overlay["mm_per_px"] != nil {
		return true
	}
	if _, ok := overlay["channels"].(map[string]any); ok {
		return true
	}
	if _, ok := overlay["defects"].([]any); ok {
		return true
	}
	return false
}

func Merge(base, overlay map[string]any) map[string]any {
	out := cloneMap(base)
	if overlay == nil {
		return out
	}
	if name := asString(overlay["scheme_name"]); name != "" {
		out["scheme_name"] = name
	}
	if ch, ok := overlay["channels"].(map[string]any); ok {
		cur := normalizeChannels(out["channels"])
		for k, v := range ch {
			if isChannelKey(k) {
				cur[k] = asBool(v)
			}
		}
		if _, ok := ch["ng_area"]; !ok {
			if v, exists := ch["area"]; exists {
				cur["ng_area"] = asBool(v)
			}
		}
		out["channels"] = normalizeChannels(cur)
	}
	if overlay["mm_per_px"] != nil {
		out["mm_per_px"] = asFloat(overlay["mm_per_px"])
	}
	byName := indexDefects(out)
	defects, _ := out["defects"].([]any)
	for _, item := range asSlice(overlay["defects"]) {
		m, ok := item.(map[string]any)
		if !ok || asString(m["name"]) == "" {
			continue
		}
		name := asString(m["name"])
		if dst, ok := byName[name]; ok {
			deepUpdate(dst, m)
		} else {
			defects = append(defects, cloneMap(m))
			out["defects"] = defects
		}
	}
	return out
}

func JudgeBox(name string, score, left, top, right, bottom float64, sch map[string]any, mmPerPx float64, laminating bool, widthPx, heightPx float64, extra map[string]any) map[string]any {
	channels := normalizeChannels(sch["channels"])
	cfg := indexDefects(sch)[name]
	cnName := labels.CNNameOf(name)
	if cfg != nil && asString(cfg["cn_name"]) != "" {
		cnName = asString(cfg["cn_name"])
	}
	if cfg != nil {
		if enable, exists := cfg["enable"]; exists && !asBool(enable) {
			return nil
		}
		if laminating && !asBool(cfg["detect_when_laminating"]) {
			return nil
		}
		confTh := asFloat(cfg["confidence"])
		if score < confTh {
			return nil
		}
	}
	metrics := boxMetrics(widthPx, heightPx, mmPerPx)
	item := map[string]any{
		"name":       name,
		"cn_name":    cnName,
		"score":      round(score, 4),
		"location": map[string]any{
			"left":   int(math.Round(left)),
			"top":    int(math.Round(top)),
			"width":  int(math.Round(math.Max(right-left, 0))),
			"height": int(math.Round(math.Max(bottom-top, 0))),
		},
		"height_mm":       round(metrics["height_mm"], 4),
		"length_mm":       round(metrics["length_mm"], 4),
		"area_mm2":        round(metrics["area_mm2"], 4),
		"ng":              false,
		"need_label":      false,
		"need_discharge":  false,
	}
	for k, v := range extra {
		item[k] = v
	}
	if cfg == nil {
		return item
	}
	areaTh := asFloat(cfg["area_mm2"])
	areaOK := areaTh <= 0 || metrics["area_mm2"] > areaTh
	if asBoolDefault(channels["ng"], true) {
		item["ng"] = sizeRuleHit(metrics["height_mm"], metrics["length_mm"], asMap(cfg["ng"])) && (!asBool(channels["ng_area"]) || areaOK)
	}
	if asBoolDefault(channels["label"], true) {
		item["need_label"] = sizeRuleHit(metrics["height_mm"], metrics["length_mm"], asMap(cfg["label"])) && (!asBool(channels["label_area"]) || areaOK)
	}
	if asBoolDefault(channels["discharge"], true) {
		item["need_discharge"] = sizeRuleHit(metrics["height_mm"], metrics["length_mm"], asMap(cfg["discharge"])) && (!asBool(channels["discharge_area"]) || areaOK)
	}
	return item
}

func sizeRuleHit(heightMM, lengthMM float64, rule map[string]any) bool {
	if rule == nil {
		return false
	}
	logic := normalizeLogic(rule["logic"])
	hTh := asFloat(rule["height_mm"])
	lTh := asFloat(rule["length_mm"])
	if logic == "length_only" {
		return lTh > 0 && lengthMM > lTh
	}
	if hTh <= 0 && lTh <= 0 {
		return false
	}
	if logic == "and" {
		hOK := true
		if hTh > 0 {
			hOK = heightMM > hTh
		}
		lOK := true
		if lTh > 0 {
			lOK = lengthMM > lTh
		}
		return hOK && lOK
	}
	parts := make([]bool, 0, 2)
	if hTh > 0 {
		parts = append(parts, heightMM > hTh)
	}
	if lTh > 0 {
		parts = append(parts, lengthMM > lTh)
	}
	for _, p := range parts {
		if p {
			return true
		}
	}
	return false
}

func boxMetrics(widthPx, heightPx, mmPerPx float64) map[string]float64 {
	w := math.Max(widthPx, 0)
	h := math.Max(heightPx, 0)
	short := math.Min(w, h)
	long := math.Max(w, h)
	scale := mmPerPx
	return map[string]float64{
		"width_px":  w,
		"height_px": h,
		"height_mm": short * scale,
		"length_mm": long * scale,
		"area_mm2":  w * h * scale * scale,
	}
}

func publicView(sch map[string]any) map[string]any {
	var defects []any
	for _, item := range asSlice(sch["defects"]) {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		name := asString(m["name"])
		defects = append(defects, map[string]any{
			"name":                    name,
			"cn_name":                 firstNonEmpty(asString(m["cn_name"]), labels.CNNameOf(name)),
			"enable":                  asBoolDefault(m["enable"], true),
			"confidence":              asFloat(m["confidence"]),
			"detect_when_laminating":  asBool(m["detect_when_laminating"]),
			"area_mm2":                asFloat(m["area_mm2"]),
			"ng":                      asRule(m["ng"]),
			"label":                   asRule(m["label"]),
			"discharge":               asRule(m["discharge"]),
		})
	}
	if defects == nil {
		defects = []any{}
	}
	mm := asFloat(sch["mm_per_px"])
	if mm == 0 {
		mm = 0.03
	}
	return map[string]any{
		"scheme_name": asString(sch["scheme_name"]),
		"mm_per_px":   mm,
		"channels":    normalizeChannels(sch["channels"]),
		"defects":     defects,
	}
}

func storageDict(sch map[string]any) map[string]any {
	view := publicView(sch)
	return map[string]any{
		"scheme_name": view["scheme_name"],
		"mm_per_px":   view["mm_per_px"],
		"channels":    view["channels"],
		"defects":     view["defects"],
	}
}

func normalizeChannels(raw any) map[string]any {
	out := map[string]any{
		"ng": true, "ng_area": true, "label": true, "label_area": false,
		"discharge": true, "discharge_area": false,
	}
	m, ok := raw.(map[string]any)
	if !ok {
		return out
	}
	for _, k := range []string{"ng", "ng_area", "label", "label_area", "discharge", "discharge_area"} {
		if _, exists := m[k]; exists {
			out[k] = asBool(m[k])
		}
	}
	if _, exists := m["ng_area"]; !exists {
		if _, exists := m["area"]; exists {
			out["ng_area"] = asBool(m["area"])
		}
	}
	return out
}

func asRule(raw any) map[string]any {
	m, ok := raw.(map[string]any)
	if !ok {
		return map[string]any{"height_mm": 0.0, "logic": "or", "length_mm": 0.0}
	}
	return map[string]any{
		"height_mm": asFloat(m["height_mm"]),
		"logic":     normalizeLogic(m["logic"]),
		"length_mm": asFloat(m["length_mm"]),
	}
}

func normalizeLogic(raw any) string {
	text := strings.TrimSpace(asString(raw))
	if text == "" {
		text = "or"
	}
	switch strings.ToLower(text) {
	case "or", "或":
		return "or"
	case "and", "与":
		return "and"
	case "length_only", "length", "仅长度":
		return "length_only"
	default:
		return "or"
	}
}

func indexDefects(sch map[string]any) map[string]map[string]any {
	out := map[string]map[string]any{}
	for _, item := range asSlice(sch["defects"]) {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		name := asString(m["name"])
		if name != "" {
			out[name] = m
		}
	}
	return out
}

func deepUpdate(dst, src map[string]any) {
	for k, v := range src {
		if (k == "ng" || k == "label" || k == "discharge") && asMap(v) != nil {
			cur := asMap(dst[k])
			if cur == nil {
				cur = map[string]any{}
			}
			merged := cloneMap(cur)
			for kk, vv := range asMap(v) {
				merged[kk] = vv
			}
			if _, ok := merged["logic"]; ok {
				merged["logic"] = normalizeLogic(merged["logic"])
			}
			dst[k] = merged
			continue
		}
		dst[k] = v
	}
}

func isChannelKey(k string) bool {
	switch k {
	case "ng", "ng_area", "label", "label_area", "discharge", "discharge_area":
		return true
	}
	return false
}

func readJSON(path string) (map[string]any, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var data map[string]any
	if err := json.Unmarshal(raw, &data); err != nil {
		return nil, err
	}
	return data, nil
}

func writeJSON(path string, data map[string]any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, append(raw, '\n'), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func cloneMap(in map[string]any) map[string]any {
	raw, _ := json.Marshal(in)
	var out map[string]any
	_ = json.Unmarshal(raw, &out)
	if out == nil {
		out = map[string]any{}
	}
	return out
}

func asSlice(v any) []any {
	s, _ := v.([]any)
	return s
}

func asMap(v any) map[string]any {
	m, _ := v.(map[string]any)
	return m
}

func asString(v any) string {
	s, _ := v.(string)
	return s
}

func asBool(v any) bool {
	switch t := v.(type) {
	case bool:
		return t
	case float64:
		return t != 0
	case string:
		return t == "true" || t == "1"
	}
	return false
}

func asBoolDefault(v any, def bool) bool {
	if v == nil {
		return def
	}
	if b, ok := v.(bool); ok {
		return b
	}
	return asBool(v)
}

func asFloat(v any) float64 {
	switch t := v.(type) {
	case float64:
		return t
	case int:
		return float64(t)
	case json.Number:
		f, _ := t.Float64()
		return f
	}
	return 0
}

func firstNonEmpty(a, b string) string {
	if strings.TrimSpace(a) != "" {
		return a
	}
	return b
}

func round(v float64, n int) float64 {
	p := math.Pow(10, float64(n))
	return math.Round(v*p) / p
}
