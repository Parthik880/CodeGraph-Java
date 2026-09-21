package search;

import ai.onnxruntime.OrtException;
import embedding.OnnxEmbeddingModel;
import model.IndexedFile;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.HexFormat;

public class EmbeddingSearch {
    private static final int CHUNK_LINES = 80;
    private static final int OVERLAP_LINES = 12;
    private static final String CACHE_VERSION = "80x12-v2";
    private final OnnxEmbeddingModel model;
    private final List<CodeChunk> chunks = new ArrayList<>();

    public EmbeddingSearch(OnnxEmbeddingModel model) {
        this.model = model;
    }

    // Reuse unchanged files' vectors across runs; only new chunks need ONNX inference.
    public void buildIndex(List<IndexedFile> files, Path repository, Consumer<String> progress)
            throws OrtException {
        chunks.clear();
        Path cacheFile = Path.of("cache", sha256(repository.toAbsolutePath().normalize().toString())
                + "-" + model.preset().name().toLowerCase() + ".bin");
        Map<String, CachedFile> cached = readCache(cacheFile);
        Map<String, String> fingerprints = new HashMap<>();
        List<Integer> missing = new ArrayList<>();
        int reused = 0;
        for (IndexedFile file : files) {
            if (file.content().isBlank()) continue;
            String fingerprint = sha256(file.content());
            fingerprints.put(file.relativePath(), fingerprint);
            CachedFile old = cached.get(file.relativePath());
            String[] lines = file.content().split("\\R", -1);
            List<CodeChunk> fileChunks = new ArrayList<>();
            for (int start = 0; start < lines.length; start += CHUNK_LINES - OVERLAP_LINES) {
                int end = Math.min(lines.length, start + CHUNK_LINES);
                StringBuilder content = new StringBuilder(file.relativePath()).append('\n');
                for (int i = start; i < end; i++) content.append(lines[i]).append('\n');
                fileChunks.add(new CodeChunk(file, file.path().toString(), file.relativePath(),
                        content.toString(), start + 1, end, null));
                if (end == lines.length) break;
            }
            boolean unchanged = old != null && old.fingerprint().equals(fingerprint)
                    && old.chunks().size() == fileChunks.size();
            if (unchanged) for (int i = 0; i < fileChunks.size(); i++) {
                CodeChunk chunk = fileChunks.get(i);
                CachedChunk previous = old.chunks().get(i);
                if (chunk.startLine() != previous.startLine() || chunk.endLine() != previous.endLine()) {
                    unchanged = false;
                    break;
                }
            }
            for (int i = 0; i < fileChunks.size(); i++) {
                CodeChunk chunk = fileChunks.get(i);
                if (unchanged) {
                    chunk = new CodeChunk(file, chunk.absolutePath(), chunk.relativePath(), chunk.content(),
                            chunk.startLine(), chunk.endLine(), old.chunks().get(i).embedding());
                    reused++;
                } else {
                    missing.add(chunks.size());
                }
                chunks.add(chunk);
            }
        }
        progress.accept("● Reused " + reused + " cached embeddings; building " + missing.size() + "...");
        // Each preset batches chunks without changing the one-query-inference search path.
        for (int start = 0; start < missing.size(); start += model.batchSize()) {
            int end = Math.min(missing.size(), start + model.batchSize());
            List<String> texts = new ArrayList<>();
            for (int i = start; i < end; i++) texts.add(chunks.get(missing.get(i)).content());
            List<float[]> vectors = model.embedBatch(texts);
            for (int i = start; i < end; i++) {
                int index = missing.get(i);
                CodeChunk chunk = chunks.get(index);
                chunks.set(index, new CodeChunk(chunk.file(), chunk.absolutePath(), chunk.relativePath(),
                        chunk.content(), chunk.startLine(), chunk.endLine(), vectors.get(i - start)));
            }
            if (end % 128 == 0 || end == missing.size()) {
                progress.accept("● " + model.preset().displayName() + " / " + model.preset().device()
                        + " — " + end + " / " + missing.size() + " chunks");
            }
        }
        writeCache(cacheFile, fingerprints, progress);
    }

    private Map<String, CachedFile> readCache(Path file) {
        if (!Files.isRegularFile(file)) return Map.of();
        try (DataInputStream input = new DataInputStream(Files.newInputStream(file))) {
            if (!CACHE_VERSION.equals(input.readUTF()) || !model.cacheKey().equals(input.readUTF())) {
                return Map.of();
            }
            int count = input.readInt();
            if (count < 0 || count > 1_000_000) return Map.of();
            Map<String, CachedFile> cached = new HashMap<>();
            for (int i = 0; i < count; i++) {
                String path = input.readUTF();
                String fingerprint = input.readUTF();
                int chunkCount = input.readInt();
                if (chunkCount < 0 || chunkCount > 1_000_000) return Map.of();
                List<CachedChunk> saved = new ArrayList<>();
                for (int j = 0; j < chunkCount; j++) {
                    int start = input.readInt();
                    int end = input.readInt();
                    float[] embedding = new float[model.dimensions()];
                    for (int k = 0; k < embedding.length; k++) embedding[k] = input.readFloat();
                    saved.add(new CachedChunk(start, end, embedding));
                }
                cached.put(path, new CachedFile(fingerprint, saved));
            }
            return cached;
        } catch (IOException error) {
            return Map.of(); // A partial or old cache is safe to rebuild.
        }
    }

    private void writeCache(Path file, Map<String, String> fingerprints, Consumer<String> progress) {
        Map<String, List<CodeChunk>> byFile = new HashMap<>();
        for (CodeChunk chunk : chunks) {
            byFile.computeIfAbsent(chunk.file().relativePath(), ignored -> new ArrayList<>()).add(chunk);
        }
        Path temporary = null;
        try {
            Files.createDirectories(file.getParent());
            temporary = Files.createTempFile(file.getParent(), "embeddings-", ".tmp");
            try (DataOutputStream output = new DataOutputStream(Files.newOutputStream(temporary))) {
                output.writeUTF(CACHE_VERSION);
                output.writeUTF(model.cacheKey());
                output.writeInt(byFile.size());
                for (Map.Entry<String, List<CodeChunk>> entry : byFile.entrySet()) {
                    output.writeUTF(entry.getKey());
                    output.writeUTF(fingerprints.get(entry.getKey()));
                    output.writeInt(entry.getValue().size());
                    for (CodeChunk chunk : entry.getValue()) {
                        output.writeInt(chunk.startLine());
                        output.writeInt(chunk.endLine());
                        for (float value : chunk.embedding()) output.writeFloat(value);
                    }
                }
            }
            try {
                Files.move(temporary, file, StandardCopyOption.ATOMIC_MOVE,
                        StandardCopyOption.REPLACE_EXISTING);
            } catch (AtomicMoveNotSupportedException error) {
                Files.move(temporary, file, StandardCopyOption.REPLACE_EXISTING);
            }
        } catch (IOException error) {
            progress.accept("● Indexed, but could not save embedding cache: " + error.getMessage());
        } finally {
            if (temporary != null) try { Files.deleteIfExists(temporary); } catch (IOException ignored) {}
        }
    }

    private static String sha256(String text) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(text.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(digest);
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException(error);
        }
    }

    /*
     * Cosine similarity compares the directions of two embedding vectors.
     * Related code can score well even when the query uses different words.
     * Vectors are normalized by the model wrapper, so their dot product is cosine.
     */
    public Map<IndexedFile, SemanticHit> search(String query) throws OrtException {
        float[] queryVector = model.embed(query);
        Map<IndexedFile, SemanticHit> best = new HashMap<>();
        for (CodeChunk chunk : chunks) {
            double score = 0;
            for (int i = 0; i < queryVector.length; i++) score += queryVector[i] * chunk.embedding[i];
            SemanticHit previous = best.get(chunk.file);
            if (previous == null || score > previous.score()) {
                best.put(chunk.file, new SemanticHit(score, chunk.startLine, chunk.endLine));
            }
        }
        return best;
    }

    public record SemanticHit(double score, int startLine, int endLine) {}

    private record CodeChunk(IndexedFile file, String absolutePath, String relativePath,
                             String content, int startLine, int endLine, float[] embedding) {}
    private record CachedFile(String fingerprint, List<CachedChunk> chunks) {}
    private record CachedChunk(int startLine, int endLine, float[] embedding) {}
}
