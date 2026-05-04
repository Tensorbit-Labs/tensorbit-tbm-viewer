# Tensorbit .tbm Viewer

A lightweight Python GUI application for inspecting .tbm model container files produced by `tensorbit-core`. Requires no external dependencies — uses only the Python standard library (tkinter).

## .tbm Container Format

```
┌─────────────────────┐
│  Tensor 0 (.tb)     │  4096-byte TBHeader + FP32 weights + mask bytes
├─────────────────────┤
│  Tensor 1 (.tb)     │
├─────────────────────┤
│       ...           │
├─────────────────────┤
│  Tensor K (.tb)     │
├─────────────────────┤
│  JSON Index         │  UTF-8 JSON: architecture, tensor names, shapes,
│                     │  byte offsets, N:M params, dtypes, weight/mask counts
├─────────────────────┤
│  4-byte LE uint32   │  Byte length of the JSON index (little-endian)
└─────────────────────┘
```

The JSON index enables random-access reading of individual tensors without loading the entire file into memory.

## Usage

```bash
python tbm_viewer.py
```

Click **Open .tbm file...** or press **Ctrl+O** to choose a `.tbm` file.

The left panel lists all tensors with a search/filter bar. Select a tensor to view:

- **Tensor Details** — metadata (name, shape, dtype, N:M sparsity, byte offsets, data sizes)
- **Weight Preview** — first N floating-point values (configurable, max 1024)
- **Mask Preview** — first 256 mask bytes in hex dump format

## License

Apache 2.0. See [LICENSE](LICENSE).
