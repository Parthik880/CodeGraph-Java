package model;

// Scores and the best matching code section for one file.
public record SearchResult(IndexedFile file, double bm25Score, double embeddingScore,
                           double finalScore, int startLine, int endLine) {
    public String absolutePath() { return file.path().toString(); }
    public String relativePath() { return file.relativePath(); }
}
