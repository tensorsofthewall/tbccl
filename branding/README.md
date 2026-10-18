# TBCCL brand assets

Canonical source of the TBCCL "Geometric Crossing" mark. The other TBCCL repositories keep their own copy of the few files their documentation uses (`docs/_static/brand/`), so each documentation site builds on its own, offline.

Design source: <https://www.figma.com/design/uOP5Ilb13nZ0ejLSnLlpuT>. The files here are SVG exports prepared from the approved path definitions, not native Figma exports. Compare them with the Figma file before using them anywhere new (print, a store listing, a trademark filing).

| File | Use |
|---|---|
| `logo/tbccl-mark.svg` | Primary colour mark on a light background. Transparent. |
| `logo/tbccl-mark-white.svg` | Single-colour mark on a dark background. Transparent. |
| `logo/tbccl-mark-black.svg` | Single-colour mark (navy `#0F172A`) for print and light backgrounds. |
| `logo/tbccl-horizontal.svg`, `logo/tbccl-stacked.svg` | Mark with the wordmark, text converted to outlines. The wordmark is dark navy: use on light backgrounds only. |
| `logo/*-editable.svg` | The same lockups with live text (font Inter). For editing, not for publishing. |
| `icons/favicon.svg`, `favicon.ico`, `favicon-{32,64,256}.png` | Transparent favicon formats. |
| `icons/github-avatar-512.png` | White mark on a navy square, for profile pictures. |

## Rules

* Do not redraw, recolour, skew or recombine the mark. Use the files as they are.
* Keep clear space of at least one strand height around the mark.
* Do not place the mark inside another logo or add a framework logo to it. A project is named in text next to the mark.
* In documentation the colour mark goes on light backgrounds and the white mark on dark ones (the darkest strand of the colour mark has about 2.2:1 contrast on a dark page, below the 3:1 that graphics need).

## Project lockups (mark + project name)

`lockups/` holds the horizontal, compact and stacked lockups of the four projects (`tbccl`, `torch-tbccl`, `vllm-tbccl`, `exo-tbccl`), each in light, dark, mono-black and mono-white; `lockups/source/` holds light-theme sources with live text for editing. The mark is the unchanged Geometric Crossing artwork. In the adapter names the hyphen is a small connector in the project accent colour (see `lockups.json`); the monochrome files use one colour throughout. No framework logo or symbol is used.

The lettering is Inter SemiBold (600), converted to paths with real shaping and kerning, so the files need no font. `lockups/MANIFEST.json` records the typeface, font version, file checksum, tool versions, the generation command and a checksum of every output. The fonts are SIL Open Font License 1.1 (Inter: copyright The Inter Project Authors); outlined lettering does not embed or redistribute them and no font binary is committed.

Regenerate (needs `pip install uharfbuzz fonttools` and the Inter SemiBold file; the tool stops with an error when the font is missing or has the wrong weight, it never substitutes one):

```sh
python3 branding/tools/build_lockups.py --editable [--font-dir DIR]
python3 branding/tools/build_lockups.py --check --editable   # compare with the committed files
python3 branding/tools/build_lockups.py --project torch-tbccl --out ../torch-tbccl/branding/lockups   # an adapter's copy
```

`--typeface space-grotesk-600` or `plex-sans-600` builds the same design in another face for comparison; the committed set uses `inter-600`. Each adapter repository keeps its own copy of its twelve files; the copies must stay byte-identical to `lockups/` here (`tests/branding_assets_test.py` checks this in every repository against its manifest).

## Palette

| Token | Value |
|---|---|
| Strand blues (dark to light) | `#075AD7`, `#1678DC`, `#239CF5`, `#10ACFB`, `#22AAFB`, `#27C2F5` |
| Wordmark and mono navy | `#0F172A` |

## Known limits of this export

* The favicon files place the unchanged mark, centred by its bounding box, in a 256 px square (scale 0.29). The mark is wide (about 1.56:1), so it fills only the middle 57% of the height. It is legible at 32 px and blurry at 16 px.
* The artwork has uneven padding in its view box (9 units above, 21 below) and the `width`/`height` do not match the `viewBox` ratio exactly (830x530 against 825x530).
* The older `logo/tbccl-horizontal*.svg` and `logo/tbccl-stacked*.svg` lockups (wordmark `TBCCL`) have no dark-background variant; the project lockups in `lockups/` do.
