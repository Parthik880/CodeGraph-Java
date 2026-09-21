package search;

import model.IndexedFile;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class BM25Search {
    private static final double K1 = 1.5;
    private static final double B = 0.75;
    private final Map<IndexedFile, Map<String, Integer>> frequencies = new HashMap<>();
    private final Map<String, Integer> documentFrequency = new HashMap<>();
    private final Map<IndexedFile, Integer> lengths = new HashMap<>();
    private double averageDocumentLength;

    // Split camelCase, snake_case, and hyphenated names into searchable words.
    public static List<String> terms(String text) {
        String separated = text.replaceAll("([A-Z])([A-Z][a-z])", "$1 $2")
                .replaceAll("([a-z0-9])([A-Z])", "$1 $2");
        List<String> words = new ArrayList<>();
        for (String word : separated.toLowerCase(java.util.Locale.ROOT).split("[^a-z0-9]+")) {
            if (!word.isEmpty()) words.add(word);
        }
        return words;
    }

    /*
     * Store each file's term counts and the number of files containing each term.
     * These are all BM25 needs later, so no separate index database is required.
     */
    public void buildIndex(List<IndexedFile> files) {
        frequencies.clear();
        documentFrequency.clear();
        lengths.clear();
        long totalLength = 0;
        for (IndexedFile file : files) {
            List<String> words = terms(file.relativePath() + " " + file.content());
            Map<String, Integer> counts = new HashMap<>();
            for (String word : words) counts.merge(word, 1, Integer::sum);
            frequencies.put(file, counts);
            lengths.put(file, words.size());
            totalLength += words.size();
            for (String word : counts.keySet()) documentFrequency.merge(word, 1, Integer::sum);
        }
        averageDocumentLength = files.isEmpty() ? 1 : (double) totalLength / files.size();
    }

    /*
     * BM25 measures how strongly query words identify a file. Words found in
     * many files receive less weight, while repeated words help with diminishing
     * returns. Document length normalization prevents long source files from
     * winning just because they contain more text. This suits exact code terms.
     */
    public Map<IndexedFile, Double> search(String query) {
        Map<IndexedFile, Double> scores = new HashMap<>();
        Set<String> queryWords = new HashSet<>(terms(query));
        for (var entry : frequencies.entrySet()) {
            IndexedFile file = entry.getKey();
            double score = 0;
            for (String word : queryWords) {
                int termFrequency = entry.getValue().getOrDefault(word, 0);
                if (termFrequency == 0) continue;
                int df = documentFrequency.get(word);
                double inverseDocumentFrequency = Math.log(1 +
                        (frequencies.size() - df + 0.5) / (df + 0.5));
                double documentLength = lengths.get(file);
                double denominator = termFrequency + K1 *
                        (1 - B + B * documentLength / averageDocumentLength);
                score += inverseDocumentFrequency * termFrequency * (K1 + 1) / denominator;
            }
            if (score > 0) scores.put(file, score);
        }
        return scores;
    }
}
