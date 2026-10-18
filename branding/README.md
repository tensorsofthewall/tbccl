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

## Palette

| Token | Value |
|---|---|
| Strand blues (dark to light) | `#075AD7`, `#1678DC`, `#239CF5`, `#10ACFB`, `#22AAFB`, `#27C2F5` |
| Wordmark and mono navy | `#0F172A` |

## Known limits of this export

* The favicon files place the unchanged mark, centred by its bounding box, in a 256 px square (scale 0.29). The mark is wide (about 1.56:1), so it fills only the middle 57% of the height. It is legible at 32 px and blurry at 16 px.
* The artwork has uneven padding in its view box (9 units above, 21 below) and the `width`/`height` do not match the `viewBox` ratio exactly (830x530 against 825x530).
* The lockups have no dark-background variant.
