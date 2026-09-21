package model;

import java.nio.file.Path;

// One readable source file in the selected repository.
public class IndexedFile {
    private final Path path;
    private final String relativePath;
    private final String name;
    private final String content;

    public IndexedFile(Path path, String relativePath, String name, String content) {
        this.path = path;
        this.relativePath = relativePath;
        this.name = name;
        this.content = content;
    }

    public Path path() { return path; }
    public String relativePath() { return relativePath; }
    public String name() { return name; }
    public String content() { return content; }
}
