package search;

import ai.onnxruntime.OrtException;
import model.IndexedFile;
import model.SearchResult;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class HybridSearch {
    private static final double BM25_WEIGHT = 0.55;
    private static final double EMBEDDING_WEIGHT = 0.45;
    private final BM25Search bm25;
    private final EmbeddingSearch embeddings;

    public HybridSearch(BM25Search bm25, EmbeddingSearch embeddings) {
        this.bm25 = bm25;
        this.embeddings = embeddings;
    }

    /*
     * BM25 rewards exact names and terms; embeddings find related ideas in
     * differently worded code. Their score ranges differ, so divide each by
     * its best positive score before applying the two visible weights.
     */
    public List<SearchResult> search(String query, int limit) throws OrtException {
        Map<IndexedFile, Double> lexical = bm25.search(query);
        Map<IndexedFile, EmbeddingSearch.SemanticHit> semantic = embeddings.search(query);
        double bestBm25 = lexical.values().stream().mapToDouble(Double::doubleValue).max().orElse(0);
        double bestSemantic = semantic.values().stream().mapToDouble(hit -> Math.max(0, hit.score())).max().orElse(0);
        Set<IndexedFile> files = new HashSet<>(semantic.keySet());
        files.addAll(lexical.keySet());
        List<SearchResult> results = new ArrayList<>();
        for (IndexedFile file : files) {
            double bm25Score = lexical.getOrDefault(file, 0.0);
            EmbeddingSearch.SemanticHit hit = semantic.get(file);
            double embeddingScore = hit == null ? 0 : hit.score();
            double normalizedBm25 = bestBm25 == 0 ? 0 : bm25Score / bestBm25;
            double normalizedEmbedding = bestSemantic == 0 ? 0 : Math.max(0, embeddingScore) / bestSemantic;
            double finalScore = BM25_WEIGHT * normalizedBm25 + EMBEDDING_WEIGHT * normalizedEmbedding;
            results.add(new SearchResult(file, bm25Score, embeddingScore, finalScore,
                    hit == null ? 1 : hit.startLine(), hit == null ? 1 : hit.endLine()));
        }
        results.sort(Comparator.comparingDouble(SearchResult::finalScore).reversed());
        return results.subList(0, Math.min(limit, results.size()));
    }
}
