# PPT Test Result Media

`test_result_media/` contains copied PNG and GIF outputs from `visualizations/`
for PowerPoint use. The folder preserves the original visualization subfolder
names so every asset remains traceable to its benchmark or simulation source.

`test_result_media_manifest.csv` maps each copied file back to its original
source path.

Current verified copy:

- 60 PNG files
- 8 GIF files
- 68 media files total

## LaTeX Presentation

`hamiltonian_geometric_results_deck.tex` is a Beamer presentation that uses
paths beginning with `test_result_media/`.

Compile from this folder:

```powershell
.\build_deck.ps1
```

Animated GIFs are listed in the deck as PPT-ready media assets. Standard
Beamer/PDF workflows do not reliably embed animated GIF playback, so insert
the GIF files directly into PowerPoint if you want animation on those slides.
