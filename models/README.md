# CodeGraph model files

Large model binaries are intentionally excluded from Git. Put each ONNX model
beside its matching `tokenizer.json` using this layout:

```text
models/
├── minilm/
│   ├── model-int8.onnx
│   └── tokenizer.json
├── bge-small/
│   ├── model-int8.onnx
│   └── tokenizer.json
└── jina-161m/
    ├── model-fp16.onnx
    └── tokenizer.json
```

| Mode | Source | Runtime |
| --- | --- | --- |
| Fast | [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2/tree/main) | INT8 / CPU |
| Balanced | [Xenova/bge-small-en-v1.5](https://huggingface.co/Xenova/bge-small-en-v1.5/tree/main) | INT8 / CPU |
| Code Quality | [jinaai/jina-embeddings-v2-base-code](https://huggingface.co/jinaai/jina-embeddings-v2-base-code/tree/main) | FP16 / CUDA GPU |

Fast is the default. A mode is unavailable when either required file is
missing. Code Quality is also rejected unless ONNX Runtime actually executes
the model on CUDA; it does not silently fall back to CPU.

To keep model files elsewhere, start Java with
`-Dcodegraph.models.dir=FULL_DIRECTORY_PATH`.
