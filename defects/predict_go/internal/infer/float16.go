package infer

import (
	"encoding/binary"
	"math"
)

// float32SliceToFloat16Bytes converts float32 values to little-endian IEEE float16 bytes.
func float32SliceToFloat16Bytes(src []float32) []byte {
	out := make([]byte, len(src)*2)
	for i, v := range src {
		binary.LittleEndian.PutUint16(out[i*2:], float32ToFloat16Bits(v))
	}
	return out
}

// float16BytesToFloat32Slice converts little-endian IEEE float16 bytes to float32.
func float16BytesToFloat32Slice(src []byte) []float32 {
	n := len(src) / 2
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = float16BitsToFloat32(binary.LittleEndian.Uint16(src[i*2:]))
	}
	return out
}

func float32ToFloat16Bits(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	exp := int((b>>23)&0xff) - 127 + 15
	mant := b & 0x7fffff

	switch {
	case exp <= 0:
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint32(14 - exp)
		mant = (mant + (1 << (shift - 1))) >> shift
		return sign | uint16(mant)
	case exp >= 31:
		if mant != 0 {
			return sign | 0x7e00 // qNaN
		}
		return sign | 0x7c00 // Inf
	default:
		return sign | uint16(exp<<10) | uint16((mant+0x1000)>>13)
	}
}

func float16BitsToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := int((h >> 10) & 0x1f)
	mant := uint32(h & 0x3ff)

	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		exp = 1
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		mant &= 0x3ff
		exp32 := uint32(exp - 1 + 127 - 15)
		return math.Float32frombits(sign | (exp32 << 23) | (mant << 13))
	case 31:
		if mant == 0 {
			return math.Float32frombits(sign | 0x7f800000)
		}
		return math.Float32frombits(sign | 0x7fc00000 | (mant << 13))
	default:
		exp32 := uint32(exp + 127 - 15)
		return math.Float32frombits(sign | (exp32 << 23) | (mant << 13))
	}
}
