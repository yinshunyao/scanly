package infer

import (
	"fmt"
	"image"
	"os"

	_ "image/jpeg"
	_ "image/png"

	_ "golang.org/x/image/bmp"
)

func LoadImage(path string) (image.Image, int, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("图片不存在: %s", path)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("无法解码图片: %s", path)
	}
	b := img.Bounds()
	return img, b.Dx(), b.Dy(), nil
}
