package license

import (
	"bytes"
	"crypto/ed25519"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	_ "embed"
	"fmt"
	"os"
	"strings"
	"time"

	"scanly/defects/predict_go/internal/gpu"
)

//go:embed license_ed25519.pub
var embeddedPublicPEM []byte

type File struct {
	Version   int        `json:"version"`
	IssuedAt  string     `json:"issued_at"`
	ExpiresAt string     `json:"expires_at"`
	Customer  string     `json:"customer"`
	ModelID   string     `json:"model_id"`
	Note      string     `json:"note"`
	GPUs      []gpu.Info `json:"gpus"`
	AESKeyB64 string     `json:"aes_key_b64"`
	Signature string     `json:"signature_b64"`
	AESKey    []byte     `json:"-"`
}

type Status struct {
	Valid         bool       `json:"valid"`
	Msg           string     `json:"msg"`
	ModelID       string     `json:"model_id,omitempty"`
	ExpiresAt     string     `json:"expires_at,omitempty"`
	GPUsLicensed  []gpu.Info `json:"gpus_licensed,omitempty"`
	GPUsLocal     []gpu.Info `json:"gpus_local,omitempty"`
}

func LoadAndVerify(path string) (*File, Status) {
	st := Status{Valid: false, Msg: "许可证无效"}
	raw, err := os.ReadFile(path)
	if err != nil {
		st.Msg = "许可证文件不存在"
		return nil, st
	}
	lic, err := parseAndVerify(raw)
	if err != nil {
		st.Msg = "许可证无效"
		return nil, st
	}
	st.ModelID = lic.ModelID
	st.ExpiresAt = lic.ExpiresAt
	st.GPUsLicensed = append([]gpu.Info{}, lic.GPUs...)
	if err := checkExpiry(lic); err != nil {
		st.Msg = err.Error()
		return lic, st
	}
	local, err := gpu.List()
	if err != nil || len(local) == 0 {
		st.Msg = "未检测到NVIDIA GPU"
		return lic, st
	}
	st.GPUsLocal = append([]gpu.Info{}, local...)
	if !gpuMatched(lic.GPUs, local) {
		st.Msg = "本机GPU信息与授权不一致"
		return lic, st
	}
	st.Valid = true
	st.Msg = "ok"
	return lic, st
}

func parseAndVerify(raw []byte) (*File, error) {
	var generic map[string]any
	if err := json.Unmarshal(raw, &generic); err != nil {
		return nil, err
	}
	sigB64, _ := generic["signature_b64"].(string)
	if strings.TrimSpace(sigB64) == "" {
		return nil, fmt.Errorf("missing signature")
	}
	delete(generic, "signature_b64")
	canonical, err := marshalCanonical(generic)
	if err != nil {
		return nil, err
	}
	sig, err := base64.StdEncoding.DecodeString(sigB64)
	if err != nil {
		return nil, err
	}
	pub, err := parsePublicKey(embeddedPublicPEM)
	if err != nil {
		return nil, err
	}
	if !ed25519.Verify(pub, canonical, sig) {
		return nil, fmt.Errorf("bad signature")
	}
	var lic File
	if err := json.Unmarshal(raw, &lic); err != nil {
		return nil, err
	}
	key, err := base64.StdEncoding.DecodeString(lic.AESKeyB64)
	if err != nil || len(key) != 32 {
		return nil, fmt.Errorf("bad aes key")
	}
	if len(lic.GPUs) == 0 {
		return nil, fmt.Errorf("empty gpu list")
	}
	for _, g := range lic.GPUs {
		if !g.Valid() {
			return nil, fmt.Errorf("gpu fields incomplete")
		}
	}
	lic.AESKey = key
	return &lic, nil
}

func parsePublicKey(pemBytes []byte) (ed25519.PublicKey, error) {
	block, _ := pem.Decode(pemBytes)
	if block == nil {
		return nil, fmt.Errorf("invalid public pem")
	}
	parsed, err := x509.ParsePKIXPublicKey(block.Bytes)
	if err != nil {
		return nil, err
	}
	pub, ok := parsed.(ed25519.PublicKey)
	if !ok {
		return nil, fmt.Errorf("not ed25519 public key")
	}
	return pub, nil
}

func marshalCanonical(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	b := bytes.TrimRight(buf.Bytes(), "\n")
	return b, nil
}

func checkExpiry(lic *File) error {
	text := strings.TrimSpace(lic.ExpiresAt)
	if text == "" {
		return nil
	}
	exp, err := time.Parse(time.RFC3339, text)
	if err != nil {
		exp, err = time.Parse("2006-01-02T15:04:05Z07:00", text)
	}
	if err != nil {
		return fmt.Errorf("许可证无效")
	}
	if time.Now().After(exp) {
		return fmt.Errorf("许可证已过期")
	}
	return nil
}

func gpuMatched(licensed, local []gpu.Info) bool {
	for _, want := range licensed {
		for _, have := range local {
			if gpu.Equal(want, have) {
				return true
			}
		}
	}
	return false
}
