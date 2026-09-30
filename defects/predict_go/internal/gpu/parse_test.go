package gpu

import "testing"

func TestParseCSVAndEqual(t *testing.T) {
	raw := "GPU-aec2d77c-763b-4295-4875-55b9887cd265, Tesla V100S-PCIE-32GB, 1562721015128\n"
	gpus, err := ParseCSV(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 1 {
		t.Fatalf("len=%d", len(gpus))
	}
	want := Info{
		UUID:   "GPU-aec2d77c-763b-4295-4875-55b9887cd265",
		Name:   "Tesla V100S-PCIE-32GB",
		Serial: "1562721015128",
	}
	if !Equal(gpus[0], want) {
		t.Fatalf("got %+v", gpus[0])
	}
	mismatch := want
	mismatch.Serial = "0"
	if Equal(gpus[0], mismatch) {
		t.Fatal("serial mismatch should fail")
	}
}
