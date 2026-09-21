package scanner;

import model.IndexedFile;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.function.Consumer;

public class RepositoryScanner {
    private static final Set<String> IGNORED = Set.of(
            ".git", "node_modules", "build", "dist", "target", "venv", ".venv",
            "__pycache__", ".idea", ".vscode");
    private static final Set<String> EXTENSIONS = Set.of(
            ".java", ".py", ".js", ".ts", ".tsx", ".jsx", ".cpp",
            ".c", ".h", ".hpp", ".go", ".rs");
    private static final long MAX_FILE_BYTES = 1_000_000;

    /*
     * walkFileTree visits a directory before its children. Returning SKIP_SUBTREE
     * here means large dependency and build folders are never opened at all.
     * Unreadable or oversized files are skipped so one bad file cannot stop indexing.
     */
    public List<IndexedFile> scan(Path root, Consumer<String> progress) throws IOException {
        if (!Files.isDirectory(root)) {
            throw new IOException("Select an existing repository folder.");
        }
        List<IndexedFile> files = new ArrayList<>();
        Files.walkFileTree(root, new SimpleFileVisitor<>() {
            @Override
            public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) {
                if (!dir.equals(root) && IGNORED.contains(dir.getFileName().toString().toLowerCase(Locale.ROOT))) {
                    return FileVisitResult.SKIP_SUBTREE;
                }
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
                String name = file.getFileName().toString();
                int dot = name.lastIndexOf('.');
                if (dot >= 0 && EXTENSIONS.contains(name.substring(dot).toLowerCase(Locale.ROOT))
                        && attrs.size() <= MAX_FILE_BYTES) {
                    try {
                        String content = Files.readString(file, StandardCharsets.UTF_8);
                        files.add(new IndexedFile(file.toAbsolutePath(), root.relativize(file).toString(), name, content));
                        if (files.size() % 100 == 0) progress.accept("● Scanned " + files.size() + " files...");
                    } catch (IOException ignored) {
                        // Keep scanning when one file is locked or not UTF-8 text.
                    }
                }
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult visitFileFailed(Path file, IOException error) {
                return FileVisitResult.CONTINUE;
            }
        });
        return files;
    }
}
