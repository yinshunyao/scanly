package infer

import (
	"image"
	"image/color"
	"math"

	"github.com/disintegration/imaging"
)

type letterboxMeta struct {
	ratio float64
	padX  float64
	padY  float64
	origW int
	origH int
	size  int
}

func ToNRGBA(src image.Image) *image.NRGBA {
	if v, ok := src.(*image.NRGBA); ok {
		return v
	}
	return imaging.Clone(src)
}

func letterboxCHW(src image.Image, size int) ([]float32, letterboxMeta) {
	nrgba := ToNRGBA(src)
	b := nrgba.Bounds()
	w, h := b.Dx(), b.Dy()
	if w < 1 {
		w = 1
	}
	if h < 1 {
		h = 1
	}
	r := math.Min(float64(size)/float64(w), float64(size)/float64(h))
	newW := int(math.Round(float64(w) * r))
	newH := int(math.Round(float64(h) * r))
	if newW < 1 {
		newW = 1
	}
	if newH < 1 {
		newH = 1
	}
	resized := imaging.Resize(nrgba, newW, newH, imaging.Linear)
	padX := (float64(size) - float64(newW)) / 2
	padY := (float64(size) - float64(newH)) / 2
	canvas := imaging.New(size, size, color.NRGBA{R: 114, G: 114, B: 114, A: 255})
	canvas = imaging.Paste(canvas, resized, image.Pt(int(math.Round(padX)), int(math.Round(padY))))
	data := make([]float32, 3*size*size)
	plane := size * size
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			c := canvas.NRGBAAt(x, y)
			idx := y*size + x
			data[idx] = float32(c.R) / 255
			data[plane+idx] = float32(c.G) / 255
			data[2*plane+idx] = float32(c.B) / 255
		}
	}
	return data, letterboxMeta{ratio: r, padX: padX, padY: padY, origW: w, origH: h, size: size}
}

func CropNRGBA(src *image.NRGBA, x1, y1, x2, y2 int) *image.NRGBA {
	b := src.Bounds()
	if x1 < b.Min.X {
		x1 = b.Min.X
	}
	if y1 < b.Min.Y {
		y1 = b.Min.Y
	}
	if x2 > b.Max.X {
		x2 = b.Max.X
	}
	if y2 > b.Max.Y {
		y2 = b.Max.Y
	}
	if x2 <= x1 || y2 <= y1 {
		return imaging.New(1, 1, color.NRGBA{A: 255})
	}
	return imaging.Crop(src, image.Rect(x1, y1, x2, y2))
}
