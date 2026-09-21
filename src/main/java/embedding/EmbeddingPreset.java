package embedding;

// One choice controls the model, precision, and execution device together.
public enum EmbeddingPreset {
    FAST("Fast", "MiniLM", "INT8", "CPU", "minilm", "model-int8.onnx", 384, 32, 256,
            "Optimized for fast indexing and searching on CPU."),
    BALANCED("Balanced", "BGE-small", "INT8", "CPU", "bge-small", "model-int8.onnx", 384, 32, 256,
            "Better semantic retrieval while remaining CPU-friendly."),
    CODE_QUALITY("Code Quality", "Jina 161M", "FP16", "GPU", "jina-161m", "model-fp16.onnx", 768, 8, 512,
            "Higher-quality code embeddings using GPU acceleration.");

    private final String displayName, model, precision, device, folder, fileName, description;
    private final int dimensions, batchSize, maxTokens;

    EmbeddingPreset(String displayName, String model, String precision, String device,
                    String folder, String fileName, int dimensions, int batchSize, int maxTokens,
                    String description) {
        this.displayName = displayName;
        this.model = model;
        this.precision = precision;
        this.device = device;
        this.folder = folder;
        this.fileName = fileName;
        this.dimensions = dimensions;
        this.batchSize = batchSize;
        this.maxTokens = maxTokens;
        this.description = description;
    }

    public String displayName() { return displayName; }
    public String model() { return model; }
    public String precision() { return precision; }
    public String device() { return device; }
    public String folder() { return folder; }
    public String fileName() { return fileName; }
    public int dimensions() { return dimensions; }
    public int batchSize() { return batchSize; }
    public int maxTokens() { return maxTokens; }
    public String description() { return description; }

    @Override public String toString() { return displayName; }
}
