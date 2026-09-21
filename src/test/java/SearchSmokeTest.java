import embedding.OnnxEmbeddingModel;
import embedding.EmbeddingPreset;
import model.IndexedFile;
import model.SearchResult;
import search.BM25Search;
import search.EmbeddingSearch;
import search.HybridSearch;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

// Run with assertions enabled to check MiniLM batching and cache reuse.
public class SearchSmokeTest {
    public static void main(String[] args) throws Exception {
        String source = "int refreshToken() { return 1; }\n".repeat(100);
        Path repository = Path.of("src/test").toAbsolutePath();
        IndexedFile file = new IndexedFile(repository.resolve("Example.java"), "Example.java",
                "Example.java", source);
        try (OnnxEmbeddingModel model = new OnnxEmbeddingModel(Path.of("models/minilm"), EmbeddingPreset.FAST)) {
            List<float[]> batch = model.embedBatch(Collections.nCopies(32, "refresh token"));
            assert batch.size() == 32 && batch.get(0).length == 384;

            EmbeddingSearch search = new EmbeddingSearch(model);
            List<String> progress = new ArrayList<>();
            search.buildIndex(List.of(file), repository, progress::add);
            progress.clear();
            search.buildIndex(List.of(file), repository, progress::add);
            assert progress.stream().anyMatch(message -> message.contains("Reused 2 cached embeddings"));
            assert search.search("refresh token").containsKey(file);

            BM25Search bm25 = new BM25Search();
            bm25.buildIndex(List.of(file));
            List<SearchResult> results = new HybridSearch(bm25, search).search("refresh token", 10);
            assert results.size() == 1;
            assert results.get(0).startLine() >= 1 && results.get(0).endLine() >= results.get(0).startLine();
            assert results.get(0).absolutePath().equals(file.path().toString());
            assert results.get(0).relativePath().equals(file.relativePath());

            progress.clear();
            IndexedFile changed = new IndexedFile(file.path(), file.relativePath(), file.name(),
                    source.replace("return 1", "return 2"));
            search.buildIndex(List.of(changed), repository, progress::add);
            assert progress.stream().anyMatch(message -> message.contains("building 2"));
        }
        System.out.println("MiniLM batch/cache and deduplicated hybrid line ranges OK");
    }
}
