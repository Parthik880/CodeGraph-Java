package ui;

import javax.swing.*;
import javax.swing.border.*;
import javax.swing.text.BadLocationException;
import javax.swing.text.DefaultHighlighter;

import embedding.EmbeddingModelManager;
import embedding.EmbeddingPreset;
import embedding.OnnxEmbeddingModel;
import model.IndexedFile;
import model.SearchResult;
import scanner.RepositoryScanner;
import search.BM25Search;
import search.EmbeddingSearch;
import search.HybridSearch;

import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.nio.file.Path;
import java.util.List;

public class CodeExplorerGUI extends JFrame {

    // -------------------- Colors --------------------
    private static final Color BG = new Color(5, 7, 10);
    private static final Color TOPBAR = new Color(16, 19, 24);
    private static final Color CARD = new Color(17, 20, 26);
    private static final Color TEXT = new Color(245, 245, 247);
    private static final Color MUTED = new Color(165, 170, 185);
    private static final Color NAV_TEXT = new Color(225, 230, 240);
    private static final Color ACCENT = new Color(200, 255, 0);
    private static final Color ACCENT_TEXT = new Color(10, 10, 10);
    private static final Color BORDER = new Color(55, 60, 70);
    private static final Color ERROR = new Color(248, 113, 113);

    // -------------------- Main UI state --------------------
    private CardLayout pageLayout;
    private JPanel pages;

    private JTextField repositoryField;
    private JTextArea queryArea;
    private JLabel statusLabel;
    // Stores the file results shown in the left panel.
    private DefaultListModel<SearchResult> resultModel;
    // Stores the source code shown in the right panel.
    private JTextArea codePreview;
    private JList<SearchResult> resultList;
    private JButton indexButton;
    private JButton searchButton;
    private JComboBox<EmbeddingPreset> presetSelector;
    private List<SearchResult> results = List.of();
    private HybridSearch search;
    private OnnxEmbeddingModel embeddingModel;
    private final EmbeddingModelManager modelManager = new EmbeddingModelManager(modelDirectory());

    public CodeExplorerGUI() {
        setTitle("CodeGraph");
        setSize(1300, 800);
        setMinimumSize(new Dimension(1000, 650));
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);
        getContentPane().setBackground(BG);
        setLayout(new BorderLayout());

        add(createTopBar(), BorderLayout.NORTH);

        pageLayout = new CardLayout();
        pages = new JPanel(pageLayout);
        pages.setBackground(BG);

        pages.add(createHomePage(), "HOME");
        pages.add(createExplorerPage(), "EXPLORER");

        add(pages, BorderLayout.CENTER);
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                if (embeddingModel != null) embeddingModel.close();
            }
        });
    }

    // -------------------- Top navigation --------------------
    private JPanel createTopBar() {
        JPanel bar = new JPanel(new BorderLayout());
        bar.setBackground(TOPBAR);
        bar.setPreferredSize(new Dimension(0, 72));
        bar.setBorder(new CompoundBorder(
                new MatteBorder(0, 0, 1, 0, BORDER),
                new EmptyBorder(0, 28, 0, 28)
        ));

        JPanel left = new JPanel(new FlowLayout(FlowLayout.LEFT, 14, 14));
        left.setOpaque(false);

        JLabel logo = new JLabel("◆");
        logo.setForeground(ACCENT);
        logo.setFont(new Font("SansSerif", Font.BOLD, 28));

        JLabel brand = new JLabel("CodeGraph");
        brand.setForeground(TEXT);
        brand.setFont(new Font("SansSerif", Font.BOLD, 19));

        left.add(logo);
        left.add(brand);
        left.add(Box.createHorizontalStrut(28));
        left.add(createNavButton("Search", "HOME"));
        left.add(createNavButton("Files", "EXPLORER"));

        JPanel right = new JPanel(new FlowLayout(FlowLayout.RIGHT, 12, 14));
        right.setOpaque(false);

        statusLabel = new JLabel("● Ready");
        statusLabel.setForeground(ACCENT);
        statusLabel.setFont(new Font("SansSerif", Font.BOLD, 12));

        JButton settingsButton = createDarkButton("Settings");
        settingsButton.addActionListener(e -> showSettings());

        JLabel modeLabel = new JLabel("Mode:");
        modeLabel.setForeground(MUTED);
        modeLabel.setFont(new Font("SansSerif", Font.BOLD, 12));
        presetSelector = new JComboBox<>(EmbeddingPreset.values());
        presetSelector.setSelectedItem(EmbeddingPreset.FAST);
        presetSelector.setBackground(new Color(25, 28, 35));
        presetSelector.setForeground(TEXT);
        presetSelector.setFont(new Font("SansSerif", Font.BOLD, 12));
        presetSelector.setFocusable(false);
        presetSelector.setToolTipText(presetTooltip(EmbeddingPreset.FAST));
        presetSelector.addActionListener(e -> changePreset());

        right.add(modeLabel);
        right.add(presetSelector);
        right.add(statusLabel);
        right.add(settingsButton);

        bar.add(left, BorderLayout.WEST);
        bar.add(right, BorderLayout.EAST);
        // At the minimum window width, keep the full status on a second row.
        bar.addComponentListener(new ComponentAdapter() {
            @Override public void componentResized(ComponentEvent event) {
                boolean stacked = bar.getWidth() < 1150;
                if (stacked == (bar.getPreferredSize().height == 136)) return;
                bar.remove(right);
                bar.setPreferredSize(new Dimension(0, stacked ? 136 : 72));
                bar.add(right, stacked ? BorderLayout.SOUTH : BorderLayout.EAST);
                bar.revalidate();
                bar.repaint();
            }
        });

        return bar;
    }

    // -------------------- Home page --------------------
    private JPanel createHomePage() {
        JPanel page = new JPanel(new GridBagLayout());
        page.setBackground(BG);

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.insets = new Insets(10, 20, 10, 20);

        JLabel heroIcon = new JLabel("◇");
        heroIcon.setForeground(ACCENT);
        heroIcon.setFont(new Font("SansSerif", Font.BOLD, 76));
        page.add(heroIcon, gbc);

        gbc.gridy++;

        JLabel title = new JLabel("CODEGRAPH");
        title.setForeground(TEXT);
        title.setFont(new Font("SansSerif", Font.BOLD, 48));
        page.add(title, gbc);

        gbc.gridy++;

        JLabel subtitle = new JLabel(
                "Explore large codebases with hybrid lexical and semantic search"
        );
        subtitle.setForeground(MUTED);
        subtitle.setFont(new Font("SansSerif", Font.PLAIN, 16));
        page.add(subtitle, gbc);

        gbc.gridy++;
        gbc.insets = new Insets(35, 20, 10, 20);
        page.add(createSearchCard(), gbc);

        gbc.gridy++;
        gbc.insets = new Insets(18, 20, 10, 20);

        JLabel hint = new JLabel(
                "Select a repository, index it, then ask a question"
        );
        hint.setForeground(MUTED);
        hint.setFont(new Font("SansSerif", Font.PLAIN, 12));
        page.add(hint, gbc);

        return page;
    }

    // -------------------- Main search card --------------------
    private JPanel createSearchCard() {
        JPanel card = new JPanel(new BorderLayout(0, 12));
        card.setPreferredSize(new Dimension(850, 210));
        card.setBackground(CARD);
        card.setBorder(new CompoundBorder(
                new LineBorder(BORDER, 1),
                new EmptyBorder(20, 24, 18, 24)
        ));

        // Repository row
        JPanel repoPanel = new JPanel(new BorderLayout(8, 0));
        repoPanel.setOpaque(false);

        repositoryField = new JTextField();
        styleInput(repositoryField);
        repositoryField.setToolTipText("Repository path");

        JButton browseButton = createDarkButton("Browse");
        indexButton = createDarkButton("Index");

        browseButton.addActionListener(e -> browseRepository());
        indexButton.addActionListener(e -> indexRepository());

        JPanel repoButtons = new JPanel(new FlowLayout(FlowLayout.RIGHT, 8, 0));
        repoButtons.setOpaque(false);
        repoButtons.add(browseButton);
        repoButtons.add(indexButton);

        repoPanel.add(repositoryField, BorderLayout.CENTER);
        repoPanel.add(repoButtons, BorderLayout.EAST);
        card.add(repoPanel, BorderLayout.NORTH);

        // Query area
        queryArea = new JTextArea();
        queryArea.setRows(3);
        queryArea.setLineWrap(true);
        queryArea.setWrapStyleWord(true);
        queryArea.setBackground(CARD);
        queryArea.setForeground(TEXT);
        queryArea.setCaretColor(TEXT);
        queryArea.setFont(new Font("SansSerif", Font.PLAIN, 16));
        queryArea.setBorder(new EmptyBorder(12, 4, 12, 4));
        queryArea.setText("Ask anything about your codebase...");

        queryArea.addFocusListener(new FocusAdapter() {
            @Override
            public void focusGained(FocusEvent e) {
                if (queryArea.getText().equals("Ask anything about your codebase...")) {
                    queryArea.setText("");
                }
            }

            @Override
            public void focusLost(FocusEvent e) {
                if (queryArea.getText().trim().isEmpty()) {
                    queryArea.setText("Ask anything about your codebase...");
                }
            }
        });

        card.add(queryArea, BorderLayout.CENTER);

        // Bottom row
        JPanel bottom = new JPanel(new BorderLayout());
        bottom.setOpaque(false);
        bottom.setBorder(new MatteBorder(1, 0, 0, 0, BORDER));

        JPanel modes = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 12));
        modes.setOpaque(false);
        modes.add(createChip("BM25"));
        modes.add(createChip("Embedding"));
        modes.add(createChip("ONNX"));

        searchButton = createAccentButton("Explore Code");
        searchButton.addActionListener(e -> runSearch());

        JPanel right = new JPanel(new FlowLayout(FlowLayout.RIGHT, 0, 10));
        right.setOpaque(false);
        right.add(searchButton);

        bottom.add(modes, BorderLayout.WEST);
        bottom.add(right, BorderLayout.EAST);
        card.add(bottom, BorderLayout.SOUTH);

        return card;
    }

    // -------------------- Explorer page --------------------
    private JPanel createExplorerPage() {
        JPanel page = new JPanel(new BorderLayout(12, 12));
        page.setBackground(BG);
        page.setBorder(new EmptyBorder(18, 22, 18, 22));

        JLabel queryLabel = new JLabel("Search Results");
        queryLabel.setForeground(TEXT);
        queryLabel.setFont(new Font("SansSerif", Font.BOLD, 20));
        page.add(queryLabel, BorderLayout.NORTH);

        JSplitPane split = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
        split.setResizeWeight(0.30);
        split.setDividerLocation(320);
        split.setDividerSize(6);
        split.setBorder(null);
        split.setBackground(BG);

        split.setLeftComponent(createResultPanel());
        split.setRightComponent(createCodePanel());

        page.add(split, BorderLayout.CENTER);
        return page;
    }

    private JPanel createResultPanel() {
        JPanel panel = new JPanel(new BorderLayout(0, 10));
        panel.setBackground(CARD);
        panel.setBorder(createPanelBorder());

        JLabel title = new JLabel("Relevant Files");
        title.setForeground(TEXT);
        title.setFont(new Font("SansSerif", Font.BOLD, 15));

        resultModel = new DefaultListModel<>();

        resultList = new JList<>(resultModel);
        resultList.setBackground(new Color(12, 15, 20));
        resultList.setForeground(TEXT);
        resultList.setSelectionBackground(new Color(40, 50, 24));
        resultList.setSelectionForeground(ACCENT);
        resultList.setCellRenderer(new SearchResultRenderer());
        resultList.setFixedCellHeight(76);
        resultList.addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting()) showPreview(resultList.getSelectedIndex());
        });

        JScrollPane scroll = new JScrollPane(resultList);
        scroll.setBorder(null);

        panel.add(title, BorderLayout.NORTH);
        panel.add(scroll, BorderLayout.CENTER);

        return panel;
    }

    // A compact three-line card keeps filename, real chunk range, and score readable.
    private class SearchResultRenderer extends JPanel implements ListCellRenderer<SearchResult> {
        private final JLabel name = new JLabel();
        private final JLabel path = new JLabel();
        private final JLabel details = new JLabel();

        SearchResultRenderer() {
            setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
            setBorder(new EmptyBorder(8, 10, 7, 10));
            name.setFont(new Font("SansSerif", Font.BOLD, 13));
            path.setFont(new Font("SansSerif", Font.PLAIN, 11));
            details.setFont(new Font("SansSerif", Font.BOLD, 11));
            add(name);
            add(Box.createVerticalStrut(3));
            add(path);
            add(Box.createVerticalStrut(3));
            add(details);
        }

        @Override public Component getListCellRendererComponent(JList<? extends SearchResult> list,
                SearchResult result, int index, boolean selected, boolean focused) {
            setBackground(selected ? new Color(40, 50, 24) : new Color(12, 15, 20));
            name.setForeground(selected ? ACCENT : TEXT);
            path.setForeground(MUTED);
            details.setForeground(selected ? ACCENT : new Color(190, 220, 110));
            name.setText(result.file().name());
            path.setText(result.relativePath());
            String range = result.startLine() == result.endLine()
                    ? "Line " + result.startLine()
                    : "Lines " + result.startLine() + "-" + result.endLine();
            details.setText(String.format("%s        Score %.2f", range, result.finalScore()));
            return this;
        }
    }

    private JPanel createCodePanel() {
        JPanel panel = new JPanel(new BorderLayout(0, 10));
        panel.setBackground(CARD);
        panel.setBorder(createPanelBorder());

        JLabel title = new JLabel("Code Preview");
        title.setForeground(TEXT);
        title.setFont(new Font("SansSerif", Font.BOLD, 15));

        codePreview = new JTextArea();
        codePreview.setEditable(false);
        codePreview.setBackground(new Color(8, 10, 14));
        codePreview.setForeground(new Color(220, 225, 235));
        codePreview.setCaretColor(TEXT);
        codePreview.setFont(new Font("Monospaced", Font.PLAIN, 14));
        codePreview.setMargin(new Insets(18, 18, 18, 18));

        codePreview.setText("Select a search result to preview its source code.");

        JScrollPane scroll = new JScrollPane(codePreview);
        scroll.setBorder(null);

        panel.add(title, BorderLayout.NORTH);
        panel.add(scroll, BorderLayout.CENTER);

        return panel;
    }

    // -------------------- Reusable UI components --------------------
    private JButton createNavButton(String text, String page) {
        JButton button = new JButton(text);
        button.setForeground(NAV_TEXT);
        button.setBackground(TOPBAR);
        button.setFocusPainted(false);
        button.setBorderPainted(false);
        button.setOpaque(true);
        button.setFont(new Font("SansSerif", Font.BOLD, 13));
        button.setCursor(new Cursor(Cursor.HAND_CURSOR));
        button.setBorder(new EmptyBorder(10, 15, 10, 15));

        button.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseEntered(MouseEvent e) {
                button.setForeground(ACCENT);
                button.setBackground(new Color(24, 28, 35));
            }

            @Override
            public void mouseExited(MouseEvent e) {
                button.setForeground(NAV_TEXT);
                button.setBackground(TOPBAR);
            }
        });

        button.addActionListener(e -> pageLayout.show(pages, page));
        return button;
    }

    private JButton createDarkButton(String text) {
        JButton button = new JButton(text);
        button.setBackground(new Color(25, 28, 35));
        button.setForeground(TEXT);
        button.setFocusPainted(false);
        button.setFont(new Font("SansSerif", Font.BOLD, 12));
        button.setCursor(new Cursor(Cursor.HAND_CURSOR));
        button.setBorder(new CompoundBorder(
                new LineBorder(BORDER),
                new EmptyBorder(9, 16, 9, 16)
        ));
        return button;
    }

    private JButton createAccentButton(String text) {
        JButton button = new JButton(text);
        button.setBackground(ACCENT);
        button.setForeground(ACCENT_TEXT);
        button.setFocusPainted(false);
        button.setBorderPainted(false);
        button.setFont(new Font("SansSerif", Font.BOLD, 14));
        button.setPreferredSize(new Dimension(155, 46));
        button.setCursor(new Cursor(Cursor.HAND_CURSOR));

        button.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseEntered(MouseEvent e) {
                button.setBackground(new Color(220, 255, 55));
            }

            @Override
            public void mouseExited(MouseEvent e) {
                button.setBackground(ACCENT);
            }
        });

        return button;
    }

    private JLabel createChip(String text) {
        JLabel chip = new JLabel(text);
        chip.setForeground(new Color(205, 210, 220));
        chip.setFont(new Font("SansSerif", Font.BOLD, 11));
        chip.setOpaque(true);
        chip.setBackground(new Color(29, 32, 40));
        chip.setBorder(new CompoundBorder(
                new LineBorder(BORDER),
                new EmptyBorder(6, 11, 6, 11)
        ));
        return chip;
    }

    private void styleInput(JTextField field) {
        field.setBackground(new Color(12, 15, 20));
        field.setForeground(TEXT);
        field.setCaretColor(TEXT);
        field.setFont(new Font("SansSerif", Font.PLAIN, 14));
        field.setBorder(new CompoundBorder(
                new LineBorder(BORDER),
                new EmptyBorder(10, 12, 10, 12)
        ));
    }

    private Border createPanelBorder() {
        return new CompoundBorder(
                new LineBorder(BORDER),
                new EmptyBorder(16, 16, 16, 16)
        );
    }

    private String presetTooltip(EmbeddingPreset preset) {
        return "<html>" + preset.model() + " / " + preset.precision() + " / " + preset.device()
                + "<br>" + preset.description() + "</html>";
    }

    /*
     * Preset validation runs off the UI thread. Switching discards the old
     * semantic search because vectors from different models cannot be mixed.
     * A failed GPU check keeps the old preset and index available.
     */
    private void changePreset() {
        EmbeddingPreset requested = (EmbeddingPreset) presetSelector.getSelectedItem();
        if (requested == null || requested == modelManager.active()) return;
        setWorking(true);
        presetSelector.setEnabled(false);
        statusLabel.setText("● Checking " + requested.displayName() + "...");
        new SwingWorker<Void, Void>() {
            @Override protected Void doInBackground() throws Exception {
                modelManager.validate(requested);
                return null;
            }
            @Override protected void done() {
                try {
                    get();
                    modelManager.activate(requested);
                    presetSelector.setToolTipText(presetTooltip(requested));
                    if (search != null) {
                        search = null;
                        results = List.of();
                        resultModel.clear();
                        codePreview.setText("Mode changed. Click Index before searching again.");
                        if (embeddingModel != null) embeddingModel.close();
                        embeddingModel = null;
                        statusLabel.setText("● Mode changed — re-index required");
                    } else {
                        statusLabel.setText("● " + requested.displayName() + " selected");
                    }
                    statusLabel.setForeground(ACCENT);
                } catch (Exception error) {
                    presetSelector.setSelectedItem(modelManager.active());
                    Throwable cause = error.getCause() == null ? error : error.getCause();
                    String message = cause.getMessage();
                    statusLabel.setText("● " + (requested.device().equals("GPU") ? "GPU unavailable" : "Model missing"));
                    statusLabel.setForeground(ERROR);
                    JOptionPane.showMessageDialog(CodeExplorerGUI.this, message,
                            "Mode unavailable", JOptionPane.WARNING_MESSAGE);
                } finally {
                    presetSelector.setEnabled(true);
                    setWorking(false);
                }
            }
        }.execute();
    }

    private void showSettings() {
        EmbeddingPreset preset = modelManager.active();
        JOptionPane.showMessageDialog(this, "Active Mode: " + preset.displayName()
                + "\nModel: " + preset.model() + "\nPrecision: " + preset.precision()
                + "\nDevice: " + preset.device() + "\nEmbedding Dimension: " + preset.dimensions()
                + "\nModel file: " + modelManager.modelPath(preset)
                + "\nExecution Provider: " + (preset.device().equals("GPU") ? "CUDA" : "CPU"),
                "Settings", JOptionPane.INFORMATION_MESSAGE);
    }

    // -------------------- Local repository actions --------------------
    private void browseRepository() {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            File folder = chooser.getSelectedFile();
            repositoryField.setText(folder.getAbsolutePath());
            statusLabel.setText("● Repository selected");
            statusLabel.setForeground(ACCENT);
        }
    }

    /*
     * Scanning, BM25 indexing, and ONNX inference run off the Swing event thread.
     * publish/process update the status label while the buttons and window stay
     * responsive. A failed new index leaves the previous index usable.
     */
    private void indexRepository() {
        String path = repositoryField.getText().trim();
        if (path.isEmpty()) {
            JOptionPane.showMessageDialog(this, "Select a repository first.");
            return;
        }
        setWorking(true);
        statusLabel.setText("● Scanning repository...");

        new SwingWorker<IndexData, String>() {
            @Override
            protected IndexData doInBackground() throws Exception {
                RepositoryScanner scanner = new RepositoryScanner();
                List<IndexedFile> files = scanner.scan(Path.of(path).toAbsolutePath().normalize(),
                        message -> publish(message));
                if (files.isEmpty()) throw new IllegalArgumentException("No readable source files found.");

                EmbeddingPreset preset = modelManager.active();
                publish("● Building " + preset.model() + " embeddings...");
                OnnxEmbeddingModel model = modelManager.loadActive();
                try {
                    BM25Search bm25 = new BM25Search();
                    bm25.buildIndex(files);
                    EmbeddingSearch embeddings = new EmbeddingSearch(model);
                    embeddings.buildIndex(files, Path.of(path), message -> publish(message));
                    return new IndexData(files.size(), model, new HybridSearch(bm25, embeddings));
                } catch (Exception error) {
                    model.close();
                    throw error;
                }
            }

            @Override
            protected void process(List<String> updates) {
                statusLabel.setText(updates.get(updates.size() - 1));
            }

            @Override
            protected void done() {
                try {
                    IndexData data = get();
                    if (embeddingModel != null) embeddingModel.close();
                    embeddingModel = data.model();
                    search = data.search();
                    results = List.of();
                    resultModel.clear();
                    codePreview.setText("Select a search result to preview its source code.");
                    statusLabel.setText("● Indexed " + data.fileCount() + " files");
                    statusLabel.setForeground(ACCENT);
                } catch (Exception error) {
                    showError("Index failed", error);
                } finally {
                    setWorking(false);
                }
            }
        }.execute();
    }

    private void runSearch() {
        String query = queryArea.getText().trim();
        if (query.isEmpty() || query.equals("Ask anything about your codebase...")) {
            JOptionPane.showMessageDialog(this, "Enter a codebase question.");
            return;
        }
        if (search == null) {
            JOptionPane.showMessageDialog(this, "Index a repository before searching.");
            return;
        }
        setWorking(true);
        statusLabel.setText("● Searching...");

        new SwingWorker<List<SearchResult>, Void>() {
            @Override
            protected List<SearchResult> doInBackground() throws Exception {
                return search.search(query, 20);
            }

            @Override
            protected void done() {
                try {
                    results = get();
                    resultModel.clear();
                    for (SearchResult result : results) resultModel.addElement(result);
                    codePreview.setText(results.isEmpty()
                            ? "No matching source files found."
                            : "Select a search result to preview its source code.");
                    pageLayout.show(pages, "EXPLORER");
                    statusLabel.setText("● Found " + results.size() + " results");
                    statusLabel.setForeground(ACCENT);
                } catch (Exception error) {
                    showError("Search failed", error);
                } finally {
                    setWorking(false);
                }
            }
        }.execute();
    }

    private void showPreview(int selectedIndex) {
        if (selectedIndex < 0 || selectedIndex >= results.size()) return;
        SearchResult result = results.get(selectedIndex);
        codePreview.setText(result.file().content());
        codePreview.getHighlighter().removeAllHighlights();
        try {
            int firstLine = Math.min(result.startLine() - 1, codePreview.getLineCount() - 1);
            int lastLine = Math.min(result.endLine() - 1, codePreview.getLineCount() - 1);
            int start = codePreview.getLineStartOffset(Math.max(0, firstLine));
            int end = codePreview.getLineEndOffset(Math.max(0, lastLine));
            codePreview.getHighlighter().addHighlight(start, end,
                    new DefaultHighlighter.DefaultHighlightPainter(new Color(58, 72, 26)));
            codePreview.setCaretPosition(start);
        } catch (BadLocationException ignored) {
            codePreview.setCaretPosition(0);
        }
    }

    private Path modelDirectory() {
        return Path.of(System.getProperty("codegraph.models.dir", "models"));
    }

    private void setWorking(boolean working) {
        indexButton.setEnabled(!working);
        searchButton.setEnabled(!working);
    }

    private void showError(String title, Exception error) {
        Throwable cause = error.getCause() == null ? error : error.getCause();
        cause.printStackTrace();
        statusLabel.setText("● " + title);
        statusLabel.setForeground(ERROR);
        JOptionPane.showMessageDialog(this, cause.getMessage(), title, JOptionPane.ERROR_MESSAGE);
    }

    private record IndexData(int fileCount, OnnxEmbeddingModel model, HybridSearch search) {}
    // -------------------- App entry point --------------------
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            CodeExplorerGUI gui = new CodeExplorerGUI();
            gui.setVisible(true);
        });
    }
}
