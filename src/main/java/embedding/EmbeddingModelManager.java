package embedding;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

// Keeps preset selection and model-file checks out of the Swing screen.
public class EmbeddingModelManager {
    private EmbeddingPreset active = EmbeddingPreset.FAST;
    private final Path modelsDirectory;

    public EmbeddingModelManager(Path modelsDirectory) {
        this.modelsDirectory = modelsDirectory;
    }

    public EmbeddingPreset active() { return active; }
    public void activate(EmbeddingPreset preset) { active = preset; }
    public Path modelPath(EmbeddingPreset preset) {
        return modelsDirectory.resolve(preset.folder()).resolve(preset.fileName());
    }

    // A GPU choice is accepted only after a real CUDA-backed inference succeeds.
    public void validate(EmbeddingPreset preset) throws IOException, OrtException {
        checkFiles(preset);
        if (preset.device().equals("GPU")) {
            try (OnnxEmbeddingModel ignored = new OnnxEmbeddingModel(modelPath(preset).getParent(), preset)) {
                // Constructor profiles a short inference to reject silent CPU fallback.
            } catch (OrtException | IOException | RuntimeException error) {
                throw new IOException("GPU execution is unavailable. Select Fast or Balanced.", error);
            }
        }
    }

    private void checkFiles(EmbeddingPreset preset) throws IOException {
        Path model = modelPath(preset);
        Path tokenizer = model.getParent().resolve("tokenizer.json");
        if (!Files.isRegularFile(model) || !Files.isRegularFile(tokenizer)) {
            throw new IOException(preset.model() + " model files are missing.");
        }
        if (preset.device().equals("GPU")
                && !OrtEnvironment.getAvailableProviders().contains(OrtProvider.CUDA)) {
            throw new IOException("GPU execution is unavailable. Select Fast or Balanced.");
        }
    }

    // The caller retains this session for all indexing batches and later queries.
    public OnnxEmbeddingModel loadActive() throws IOException, OrtException {
        checkFiles(active);
        return new OnnxEmbeddingModel(modelPath(active).getParent(), active);
    }
}
