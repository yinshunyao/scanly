# 厂商密钥目录

- `license_ed25519.pem`：签发 License 的私钥，**禁止**打进现场安装包、禁止提交 Git。
- `license_ed25519.pub`：公钥副本；Go 服务实际使用 `internal/license/license_ed25519.pub`（编译进二进制）。

首次或轮换密钥：在 `tools/gen_license.py` 设 `INIT_KEYS = True` 后运行，然后重新编译 Go 服务。
