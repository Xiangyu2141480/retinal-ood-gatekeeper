# Dissertation Submission Source Snapshot

This directory preserves the contents of the supplied final dissertation source archive while
keeping its internal `docs/` and `reports/` paths intact. It is a manuscript source snapshot and
does not replace the repository's canonical experiment evidence or documentation.

The archive contains the LaTeX entry point (`main.tex`), bibliography (`sample.bib`), document
class, supporting documentation, figures, and the supplied compile log. The supplied archive did
not contain the generated dissertation PDF, so no PDF was added beyond the included figure files.

Original archive SHA-256:

```text
E66EF761454B2638933BE22ED79FEF70AE89E2C1807DAC01FD3FF10E97A27DD3
```

To compile from this directory with Tectonic:

```bash
tectonic main.tex
```

For repository privacy and hygiene, one machine-specific executable path in
`latest_main_compile.log` was replaced with `<local-tectonic-path>`, and trailing whitespace was
normalised in the compile log, document class, and bibliography. The textual content is otherwise
preserved as provided.
