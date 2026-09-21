<p align="center">
  <img src="docs/codegraph-logo.svg" alt="CodeGraph logo" width="620">
</p>

# CodeGraph

CodeGraph is a local Java desktop application for exploring large source-code
repositories using hybrid lexical and semantic search. It scans source files,
ranks exact identifiers with BM25, finds related code with ONNX embeddings,
and opens the strongest matching line range in a source preview.

## Features

- Java Swing desktop interface
- Local repository scanning
- BM25 lexical search
- ONNX semantic embeddings
- Hybrid BM25 + embedding ranking
- Line-based code chunk retrieval with overlap
- Real matching line ranges
- Full source-code preview with matching-range highlighting
- Fast, Balanced, and Code Quality modes
- Persistent embedding cache for unchanged files
- Local-first processing with no server, API key, or LLM

## Architecture

```text
Repository
   |
   v
RepositoryScanner
   |
   +------------------+
   |                  |
   v                  v
BM25Search      Code chunking
                     |
                     v
              EmbeddingSearch
                     |
              ONNX Runtime
                     |
          +----------+
          |
          v
       HybridSearch
          |
          v
      SearchResult
          |
          v
       Swing UI
```

## Search modes

| Mode | Model | Precision | Device | Dimensions | Index batch |
| --- | --- | --- | --- | ---: | ---: |
| Fast | MiniLM | INT8 | CPU | 384 | 32 |
| Balanced | BGE-small | INT8 | CPU | 384 | 32 |
| Code Quality | Jina 161M | FP16 | CUDA GPU | 768 | 8 |

Fast is the default and works without a GPU. Code Quality is accepted only
after ONNX Runtime completes a CUDA-backed test inference; it never silently
uses CPU for the main model computation.

## Search result example

```text
AuthService.java
src/auth/AuthService.java
Lines 41-51        Score 0.87
```

The displayed range is the strongest semantic code chunk for that file, not a
generated estimate. Each file appears once. Selecting it opens the complete
file, scrolls to the first matching line, and highlights the matching range.

## Requirements

- JDK 21
- Apache Maven 3.9 or newer
- ONNX model and tokenizer files for the modes you want to use
- For Code Quality only: a compatible NVIDIA CUDA/cuDNN installation

Maven installs ONNX Runtime GPU for Java 1.23.2 (which also supports CPU
execution) and DJL Hugging Face tokenizers 0.36.0.

## Build

```powershell
cd CodeGraph-Java
mvn clean compile
```

## Run

The `exec-maven-plugin` is configured with `Main` as the entry point:

```powershell
mvn exec:java
```

In the application:

1. Click **Browse** and choose a repository.
2. Choose a mode; Fast is selected initially.
3. Click **Index**.
4. Enter a code question or identifier and click **Explore Code**.
5. Select a result to open and highlight its matching lines.

Changing modes after indexing invalidates semantic results and requires a new
index. Model-specific embedding caches are stored in the ignored `cache/`
directory and reused for unchanged files.

## Models

ONNX weights and tokenizers are not committed because they are large runtime
files. See [models/README.md](models/README.md) for sources and this layout:

```text
models/
├── README.md
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

Only `models/README.md` belongs in Git.

## Project structure

```text
CodeGraph-Java/
├── pom.xml
├── README.md
├── models/
│   └── README.md
└── src/
    ├── main/java/
    │   ├── Main.java
    │   ├── embedding/
    │   │   ├── EmbeddingModelManager.java
    │   │   ├── EmbeddingPreset.java
    │   │   ├── OnnxEmbeddingModel.java
    │   │   └── Tokenizer.java
    │   ├── model/
    │   │   ├── IndexedFile.java
    │   │   └── SearchResult.java
    │   ├── scanner/
    │   │   └── RepositoryScanner.java
    │   ├── search/
    │   │   ├── BM25Search.java
    │   │   ├── EmbeddingSearch.java
    │   │   └── HybridSearch.java
    │   └── ui/
    │       └── CodeExplorerGUI.java
    └── test/java/
        └── SearchSmokeTest.java
```

## How hybrid search works

`RepositoryScanner` reads supported UTF-8 source files while skipping build,
dependency, IDE, and version-control directories. `BM25Search` ranks exact
query terms, including camelCase and snake_case identifiers.

`EmbeddingSearch` divides files into 80-line chunks with 12 lines of overlap,
embeds missing chunks in batches, and retains the best chunk per file. The
query needs one embedding inference. `HybridSearch` normalizes both score
ranges and combines them as 55% BM25 and 45% semantic similarity.

## Limitations

- Chunking is line-based rather than AST-aware.
- Dense 80-line chunks may be truncated by a model's token limit.
- BM25 indexes whole files, while the displayed range comes from the strongest
  semantic chunk.
- Files over 1 MB, unreadable files, unsupported extensions, and non-UTF-8
  files are skipped.
- Model files must be downloaded separately.
- GPU mode requires compatible ONNX Runtime, CUDA, cuDNN, and NVIDIA hardware.
