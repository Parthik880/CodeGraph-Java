package embedding;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class OnnxEmbeddingModel implements AutoCloseable {
    private final OrtEnvironment environment;
    private final OrtSession session;
    private final Tokenizer tokenizer;
    private final EmbeddingPreset preset;
    private final String cacheKey;

    public OnnxEmbeddingModel(Path modelDirectory, EmbeddingPreset preset) throws IOException, OrtException {
        this.preset = preset;
        Path modelFile = modelDirectory.resolve(preset.fileName());
        Path tokenizerFile = modelDirectory.resolve("tokenizer.json");
        if (!Files.isRegularFile(modelFile) || !Files.isRegularFile(tokenizerFile)) {
            throw new IOException(preset.model() + " model files are missing.");
        }
        cacheKey = preset.name() + ":" + Files.size(modelFile) + ":" + Files.getLastModifiedTime(modelFile) + ":"
                + Files.size(tokenizerFile) + ":" + Files.getLastModifiedTime(tokenizerFile);
        environment = OrtEnvironment.getEnvironment();
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            if (preset.device().equals("GPU")) {
                // Jina FP16 needs CUDA; disabling optimizer passes also avoids a bad FP16 fusion.
                Files.createDirectories(Path.of("cache"));
                options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.NO_OPT);
                options.addCUDA(0);
                options.enableProfiling(Path.of("cache", "cuda-check").toString());
            }
            session = environment.createSession(modelFile.toString(), options);
        }
        try {
            tokenizer = new Tokenizer(tokenizerFile, preset.maxTokens());
        } catch (IOException | RuntimeException error) {
            session.close();
            throw error;
        }
        if (preset.device().equals("GPU")) {
            try {
                embed("CUDA provider check");
                Path profile = Path.of(session.endProfiling());
                Matcher providers = Pattern.compile("\\\"provider\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"")
                        .matcher(Files.readString(profile));
                int cudaNodes = 0, cpuNodes = 0;
                while (providers.find()) {
                    if (providers.group(1).equals("CUDAExecutionProvider")) cudaNodes++;
                    if (providers.group(1).equals("CPUExecutionProvider")) cpuNodes++;
                }
                Files.deleteIfExists(profile);
                // Shape helpers may use CPU, but the model's work must be on CUDA.
                if (cudaNodes < 10 || cudaNodes <= cpuNodes) {
                    throw new IOException("The ONNX session did not run primarily on CUDA.");
                }
                System.out.println("CUDA inference: " + cudaNodes + " CUDA nodes, " + cpuNodes + " CPU nodes");
            } catch (IOException | OrtException | RuntimeException error) {
                close();
                throw error;
            }
        }
    }

    /*
     * MiniLM and Jina use mean pooling; BGE uses its first [CLS] token.
     * Normalization makes later dot products equal cosine similarity.
     */
    public float[] embed(String text) throws OrtException {
        if (preset == EmbeddingPreset.BALANCED) {
            text = "Represent this sentence for searching relevant passages: " + text;
        }
        return embedBatch(List.of(text)).get(0);
    }

    public int dimensions() { return preset.dimensions(); }
    public int batchSize() { return preset.batchSize(); }
    public EmbeddingPreset preset() { return preset; }
    public String cacheKey() {
        return cacheKey;
    }

    public List<float[]> embedBatch(List<String> texts) throws OrtException {
        List<Tokenizer.Tokens> tokens = new ArrayList<>();
        int width = 0;
        for (String text : texts) {
            Tokenizer.Tokens encoded = tokenizer.encode(text);
            tokens.add(encoded);
            width = Math.max(width, encoded.ids().length);
        }
        long[][] idsData = new long[texts.size()][width];
        long[][] maskData = new long[texts.size()][width];
        for (int row = 0; row < tokens.size(); row++) {
            System.arraycopy(tokens.get(row).ids(), 0, idsData[row], 0, tokens.get(row).ids().length);
            System.arraycopy(tokens.get(row).attentionMask(), 0, maskData[row], 0,
                    tokens.get(row).attentionMask().length);
        }
        try (OnnxTensor ids = OnnxTensor.createTensor(environment, idsData);
             OnnxTensor mask = OnnxTensor.createTensor(environment, maskData);
             OnnxTensor types = OnnxTensor.createTensor(environment, new long[texts.size()][width]);
             OrtSession.Result output = session.run(inputs(ids, mask, types))) {
            float[][][] hidden = (float[][][]) output.get("last_hidden_state").orElseThrow().getValue();
            List<float[]> vectors = new ArrayList<>();
            for (int row = 0; row < hidden.length; row++) {
                float[] vector = new float[hidden[row][0].length];
                if (preset == EmbeddingPreset.BALANCED) {
                    System.arraycopy(hidden[row][0], 0, vector, 0, vector.length);
                } else {
                    int count = 0;
                    for (int token = 0; token < hidden[row].length; token++) {
                        if (maskData[row][token] == 0) continue;
                        for (int dimension = 0; dimension < vector.length; dimension++) {
                            vector[dimension] += hidden[row][token][dimension];
                        }
                        count++;
                    }
                    for (int i = 0; i < vector.length; i++) vector[i] /= count;
                }
                if (vector.length != preset.dimensions()) {
                    throw new IllegalStateException("Unexpected " + preset.model() + " embedding size: " + vector.length);
                }
                double norm = 0;
                for (int i = 0; i < vector.length; i++) {
                    norm += vector[i] * vector[i];
                }
                norm = Math.sqrt(norm);
                if (norm > 0) for (int i = 0; i < vector.length; i++) vector[i] /= norm;
                vectors.add(vector);
            }
            return vectors;
        }
    }

    private Map<String, OnnxTensor> inputs(OnnxTensor ids, OnnxTensor mask, OnnxTensor types) {
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input_ids", ids);
        inputs.put("attention_mask", mask);
        if (session.getInputNames().contains("token_type_ids")) inputs.put("token_type_ids", types);
        return inputs;
    }

    @Override
    public void close() {
        tokenizer.close();
        try {
            session.close();
        } catch (OrtException error) {
            throw new IllegalStateException("Could not close the ONNX model.", error);
        }
    }
}
