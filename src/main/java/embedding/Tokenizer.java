package embedding;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

public class Tokenizer implements AutoCloseable {
    private final int maxTokens;
    private final HuggingFaceTokenizer tokenizer;

    public Tokenizer(Path file, int maxTokens) throws IOException {
        tokenizer = HuggingFaceTokenizer.newInstance(file);
        this.maxTokens = maxTokens;
    }

    // Each preset keeps its own tokenizer vocabulary and special tokens.
    public Tokens encode(String text) {
        Encoding encoding = tokenizer.encode(text);
        long[] ids = encoding.getIds();
        long[] mask = encoding.getAttentionMask();
        if (ids.length > maxTokens) {
            long lastToken = ids[ids.length - 1];
            ids = Arrays.copyOf(ids, maxTokens);
            mask = Arrays.copyOf(mask, maxTokens);
            ids[maxTokens - 1] = lastToken; // Keep the closing special token.
        }
        return new Tokens(ids, mask);
    }

    public record Tokens(long[] ids, long[] attentionMask) {}

    @Override
    public void close() {
        tokenizer.close();
    }
}
