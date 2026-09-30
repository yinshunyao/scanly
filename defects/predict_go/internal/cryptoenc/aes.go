package cryptoenc

import (
	"crypto/aes"
	"crypto/cipher"
	"fmt"
)

const (
	magic     = "SLYENC01"
	nonceLen  = 12
	aad       = "scanly-defects-onnx-v1"
	keyBytes  = 32
)

func Decrypt(blob, key []byte) ([]byte, error) {
	if len(key) != keyBytes {
		return nil, fmt.Errorf("AES 密钥长度无效")
	}
	if len(blob) < len(magic)+nonceLen+16 {
		return nil, fmt.Errorf("密文过短")
	}
	if string(blob[:len(magic)]) != magic {
		return nil, fmt.Errorf("不是 scanly ONNX 密文（魔数不匹配）")
	}
	nonce := blob[len(magic) : len(magic)+nonceLen]
	ct := blob[len(magic)+nonceLen:]
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	plain, err := gcm.Open(nil, nonce, ct, []byte(aad))
	if err != nil {
		return nil, fmt.Errorf("模型解密失败")
	}
	return plain, nil
}
