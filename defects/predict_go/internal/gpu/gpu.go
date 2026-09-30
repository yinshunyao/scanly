package gpu

import (
	"encoding/csv"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

type Info struct {
	UUID   string `json:"uuid"`
	Name   string `json:"name"`
	Serial string `json:"serial"`
}

func NormalizeUUID(uuid string) string {
	return strings.ToUpper(strings.TrimSpace(uuid))
}

func NormalizeText(s string) string {
	return strings.TrimSpace(s)
}

func (g Info) Valid() bool {
	return NormalizeUUID(g.UUID) != "" && NormalizeText(g.Name) != "" && NormalizeText(g.Serial) != ""
}

func Equal(a, b Info) bool {
	return NormalizeUUID(a.UUID) == NormalizeUUID(b.UUID) &&
		NormalizeText(a.Name) == NormalizeText(b.Name) &&
		NormalizeText(a.Serial) == NormalizeText(b.Serial)
}

func List() ([]Info, error) {
	exe, err := lookNvidiaSMI()
	if err != nil {
		return nil, err
	}
	out, err := exec.Command(exe, "--query-gpu=uuid,name,serial", "--format=csv,noheader").CombinedOutput()
	if err != nil {
		return nil, errNoGPU
	}
	gpus, err := ParseCSV(string(out))
	if err != nil || len(gpus) == 0 {
		return nil, errNoGPU
	}
	return gpus, nil
}

func ParseCSV(raw string) ([]Info, error) {
	r := csv.NewReader(strings.NewReader(raw))
	r.TrimLeadingSpace = true
	r.FieldsPerRecord = -1
	rows, err := r.ReadAll()
	if err != nil {
		return nil, err
	}
	var gpus []Info
	for _, row := range rows {
		if len(row) < 3 {
			continue
		}
		item := Info{
			UUID:   strings.TrimSpace(row[0]),
			Name:   strings.TrimSpace(row[1]),
			Serial: strings.TrimSpace(row[2]),
		}
		if item.Valid() {
			gpus = append(gpus, item)
		}
	}
	return gpus, nil
}

var errNoGPU = errString("未检测到NVIDIA GPU")

type errString string

func (e errString) Error() string { return string(e) }

func lookNvidiaSMI() (string, error) {
	if p, err := exec.LookPath("nvidia-smi"); err == nil {
		return p, nil
	}
	if runtime.GOOS == "windows" {
		for _, p := range []string{
			filepath.Join(`C:\Program Files\NVIDIA Corporation\NVSMI`, "nvidia-smi.exe"),
			filepath.Join(`C:\Windows\System32`, "nvidia-smi.exe"),
		} {
			if st, err := os.Stat(p); err == nil && !st.IsDir() {
				return p, nil
			}
		}
	}
	return "", errNoGPU
}
