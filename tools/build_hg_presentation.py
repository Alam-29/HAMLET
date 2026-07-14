from __future__ import annotations

import html
import shutil
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "hamiltonian_geometric_presentation.pptx"
NOTES = ROOT / "docs" / "hamiltonian_geometric_presentation_notes.md"

SLIDE_W = 13_333_333
SLIDE_H = 7_500_000

BG = "F8FAFC"
INK = "172033"
MUTED = "556070"
BLUE = "2563EB"
TEAL = "0F766E"
GREEN = "15803D"
ORANGE = "C2410C"
PURPLE = "7C3AED"
RED = "B91C1C"
LINE = "CBD5E1"


def emu(inches: float) -> int:
    return int(inches * 914400)


def x(x_in: float) -> int:
    return emu(x_in)


def y(y_in: float) -> int:
    return emu(y_in)


def w(w_in: float) -> int:
    return emu(w_in)


def h(h_in: float) -> int:
    return emu(h_in)


def tx(text: str) -> str:
    return escape(text)


def shape_text(
    shape_id: int,
    text: str,
    left: int,
    top: int,
    width: int,
    height: int,
    font_size: int = 24,
    color: str = INK,
    bold: bool = False,
    fill: str | None = None,
    line: str | None = None,
    radius: bool = True,
    align: str = "l",
) -> str:
    prst = "roundRect" if radius else "rect"
    fill_xml = (
        f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'
        if fill
        else "<a:noFill/>"
    )
    line_xml = (
        f'<a:ln w="12000"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
        if line
        else "<a:ln><a:noFill/></a:ln>"
    )
    bold_xml = ' b="1"' if bold else ""
    body = "".join(
        f'<a:p><a:pPr algn="{align}"/><a:r><a:rPr lang="en-US" sz="{font_size * 100}"{bold_xml}>'
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:rPr><a:t>{tx(line_text)}</a:t></a:r></a:p>'
        for line_text in text.split("\n")
    )
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{shape_id}" name="Text {shape_id}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{left}" y="{top}"/><a:ext cx="{width}" cy="{height}"/></a:xfrm>
        <a:prstGeom prst="{prst}"><a:avLst/></a:prstGeom>
        {fill_xml}{line_xml}
      </p:spPr>
      <p:txBody><a:bodyPr wrap="square" lIns="110000" rIns="110000" tIns="70000" bIns="70000"/><a:lstStyle/>{body}</p:txBody>
    </p:sp>
    """


def title_block(title: str, subtitle: str | None = None) -> str:
    out = shape_text(10, title, x(0.55), y(0.22), w(12.1), h(0.55), 28, INK, True, None, None, False)
    if subtitle:
        out += shape_text(11, subtitle, x(0.58), y(0.78), w(11.9), h(0.36), 13, MUTED, False, None, None, False)
    out += f'<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="12" name="line"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr><p:spPr><a:xfrm><a:off x="{x(0.58)}" y="{y(1.18)}"/><a:ext cx="{w(12.1)}" cy="0"/></a:xfrm><a:prstGeom prst="line"><a:avLst/></a:prstGeom><a:ln w="14000"><a:solidFill><a:srgbClr val="{LINE}"/></a:solidFill></a:ln></p:spPr></p:cxnSp>'
    return out


def image_xml(shape_id: int, rid: str, left: int, top: int, width: int, height: int) -> str:
    return f"""
    <p:pic>
      <p:nvPicPr><p:cNvPr id="{shape_id}" name="Picture {shape_id}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>
      <p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
      <p:spPr><a:xfrm><a:off x="{left}" y="{top}"/><a:ext cx="{width}" cy="{height}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>
    </p:pic>
    """


def fit_image(path: Path, left: float, top: float, width: float, height: float) -> tuple[int, int, int, int]:
    with Image.open(path) as img:
        iw, ih = img.size
    box_w, box_h = w(width), h(height)
    scale = min(box_w / iw, box_h / ih)
    rw, rh = int(iw * scale), int(ih * scale)
    return x(left) + (box_w - rw) // 2, y(top) + (box_h - rh) // 2, rw, rh


def diagram_pipeline(shape_id: int, labels: list[str], colors: list[str], top: float = 2.2) -> str:
    xml = ""
    left = 0.75
    box_w = 2.05
    gap = 0.42
    for i, label in enumerate(labels):
        lx = left + i * (box_w + gap)
        xml += shape_text(shape_id + i * 3, label, x(lx), y(top), w(box_w), h(0.78), 16, "FFFFFF", True, colors[i], None, True, "ctr")
        if i < len(labels) - 1:
            xml += shape_text(shape_id + i * 3 + 1, "→", x(lx + box_w + 0.06), y(top + 0.12), w(0.32), h(0.45), 24, MUTED, True, None, None, False, "ctr")
    return xml


def slide_xml(body: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg><p:bgPr><a:solidFill><a:srgbClr val="{BG}"/></a:solidFill></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {body}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


class Deck:
    def __init__(self) -> None:
        self.slides: list[dict] = []
        self.media: dict[Path, str] = {}

    def add(self, body: str, images: list[Path] | None = None, notes: str = "") -> None:
        self.slides.append({"body": body, "images": images or [], "notes": notes})

    def media_name(self, path: Path) -> str:
        if path not in self.media:
            self.media[path] = f"image{len(self.media) + 1}{path.suffix.lower()}"
        return self.media[path]

    def write(self) -> None:
        if OUT.exists():
            OUT.unlink()
        for slide in self.slides:
            for p in slide["images"]:
                self.media_name(p)
        with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", self.content_types())
            z.writestr("_rels/.rels", self.root_rels())
            z.writestr("ppt/presentation.xml", self.presentation())
            z.writestr("ppt/_rels/presentation.xml.rels", self.presentation_rels())
            z.writestr("ppt/slideMasters/slideMaster1.xml", self.slide_master())
            z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", self.slide_master_rels())
            z.writestr("ppt/slideLayouts/slideLayout1.xml", self.slide_layout())
            z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", self.slide_layout_rels())
            z.writestr("ppt/theme/theme1.xml", self.theme())
            z.writestr("docProps/core.xml", self.core())
            z.writestr("docProps/app.xml", self.app())
            for i, slide in enumerate(self.slides, 1):
                z.writestr(f"ppt/slides/slide{i}.xml", slide_xml(slide["body"]))
                z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", self.slide_rels(slide))
            for path, name in self.media.items():
                z.write(path, f"ppt/media/{name}")
        NOTES.write_text(self.notes_markdown(), encoding="utf-8")

    def content_types(self) -> str:
        slide_overrides = "\n".join(
            f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
            for i in range(1, len(self.slides) + 1)
        )
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Default Extension="gif" ContentType="image/gif"/>
<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
{slide_overrides}
</Types>"""

    def root_rels(self) -> str:
        return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""

    def presentation(self) -> str:
        ids = "\n".join(f'<p:sldId id="{255+i}" r:id="rId{i}"/>' for i in range(1, len(self.slides) + 1))
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{len(self.slides)+1}"/></p:sldMasterIdLst>
<p:sldIdLst>{ids}</p:sldIdLst>
<p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
<p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>"""

    def presentation_rels(self) -> str:
        rels = [
            f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
            for i in range(1, len(self.slides) + 1)
        ]
        rels.append(f'<Relationship Id="rId{len(self.slides)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>')
        rels.append(f'<Relationship Id="rId{len(self.slides)+2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>')
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{''.join(rels)}</Relationships>"""

    def slide_rels(self, slide: dict) -> str:
        rels = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>']
        for i, path in enumerate(slide["images"], 2):
            rels.append(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{self.media_name(path)}"/>')
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{''.join(rels)}</Relationships>"""

    def slide_master(self) -> str:
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/><p:sldLayoutIdLst><p:sldLayoutId id="1" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>"""

    def slide_master_rels(self) -> str:
        return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/></Relationships>"""

    def slide_layout(self) -> str:
        return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1"><p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>"""

    def slide_layout_rels(self) -> str:
        return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>"""

    def theme(self) -> str:
        return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="HG Theme"><a:themeElements><a:clrScheme name="HG"><a:dk1><a:srgbClr val="172033"/></a:dk1><a:lt1><a:srgbClr val="F8FAFC"/></a:lt1><a:dk2><a:srgbClr val="334155"/></a:dk2><a:lt2><a:srgbClr val="E2E8F0"/></a:lt2><a:accent1><a:srgbClr val="2563EB"/></a:accent1><a:accent2><a:srgbClr val="0F766E"/></a:accent2><a:accent3><a:srgbClr val="C2410C"/></a:accent3><a:accent4><a:srgbClr val="7C3AED"/></a:accent4><a:accent5><a:srgbClr val="15803D"/></a:accent5><a:accent6><a:srgbClr val="B91C1C"/></a:accent6><a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink></a:clrScheme><a:fontScheme name="Aptos"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme><a:fmtScheme name="Clean"><a:fillStyleLst><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="12000"><a:solidFill><a:srgbClr val="CBD5E1"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:srgbClr val="F8FAFC"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>"""

    def core(self) -> str:
        return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>Hamiltonian-Geometric Optimization Presentation</dc:title><dc:creator>HAMLET workspace</dc:creator></cp:coreProperties>"""

    def app(self) -> str:
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Codex</Application><Slides>{len(self.slides)}</Slides></Properties>"""

    def notes_markdown(self) -> str:
        chunks = ["# Hamiltonian-Geometric Optimization: Speaker Notes\n"]
        for i, slide in enumerate(self.slides, 1):
            chunks.append(f"## Slide {i}\n\n{slide['notes'].strip()}\n")
        return "\n".join(chunks)


def add_image(deck: Deck, path: str, shape_id: int, left: float, top: float, width: float, height: float) -> tuple[str, Path]:
    p = ROOT / path
    rid = f"rId{shape_id}"
    lx, ty, ww, hh = fit_image(p, left, top, width, height)
    return image_xml(shape_id + 100, rid, lx, ty, ww, hh), p


def main() -> None:
    d = Deck()

    # 1
    body = (
        shape_text(20, "Hamiltonian-Geometric Optimization", x(0.75), y(0.55), w(12), h(0.85), 36, INK, True, None, None, False)
        + shape_text(21, "Optimization as motion on a curved landscape", x(0.78), y(1.42), w(11.5), h(0.38), 18, MUTED, False, None, None, False)
        + diagram_pipeline(30, ["Loss\nenergy", "Metric\ngeometry", "Momentum\nstate", "Damping\ntraining", "Evidence\nplots"], [BLUE, TEAL, PURPLE, ORANGE, GREEN], 2.35)
        + shape_text(51, "Not a magic optimizer. A geometry-first framework for structured learning problems.", x(1.0), y(4.9), w(11.2), h(0.78), 25, INK, True, "FFFFFF", LINE, True, "ctr")
    )
    d.add(body, notes="Open with the high-level claim. The project is a framework, not a magic replacement for Adam. Its value is that it gives a principled language for curvature, momentum, memory, and phase-space control.")

    # 2
    img, p = add_image(d, "visualizations/geometric_evidence/condition_number_scaling.png", 2, 0.55, 1.25, 5.85, 4.75)
    img2, p2 = add_image(d, "visualizations/geometric_evidence/rotation_invariance.png", 3, 6.85, 1.25, 5.85, 4.75)
    body = title_block("Why Build This?", "Because many hard landscapes are curved, rotated, stiff, or structured.")
    body += img + img2
    body += shape_text(50, "Core bet: use geometry instead of pretending the landscape is flat.", x(1.15), y(6.18), w(11.0), h(0.46), 19, "FFFFFF", True, INK, None, True, "ctr")
    d.add(body, notes="Explain that we are trying to make optimization match the geometry of the problem. This is most compelling for physics-informed models, quantum control, spin systems, and tensor networks.")

    # 3
    body = title_block("The Core Idea", "A compact map from mechanics to learning.")
    body += diagram_pipeline(30, ["θ\nposition", "L(θ)\npotential", "g(θ)\nmetric", "p\nmomentum", "γ\ndamping"], [BLUE, ORANGE, TEAL, PURPLE, GREEN], 1.55)
    body += shape_text(60, "H(θ,p) = kinetic geometry + loss energy", x(1.45), y(3.25), w(10.3), h(0.85), 30, INK, True, "FFFFFF", LINE, True, "ctr")
    body += shape_text(61, "Hamiltonian motion + dissipation = optimizer trajectory", x(1.45), y(4.45), w(10.3), h(0.75), 25, "FFFFFF", True, INK, None, True, "ctr")
    d.add(body, notes="This is the cleanest conceptual slide. Parameters become coordinates, the loss is potential energy, the metric tells us what a unit step means locally, and momentum carries state. Damping is essential because pure Hamiltonian mechanics conserves energy, while optimization should decrease loss.")

    # 4
    body = title_block("Derived vs Proposed", "Say this clearly in the talk.")
    body += shape_text(30, "STANDARD", x(0.85), y(1.55), w(3.55), h(0.7), 26, "FFFFFF", True, GREEN, None, True, "ctr")
    body += shape_text(31, "Hamilton's equations\nLegendre transform\nRayleigh damping\nNatural gradient", x(0.95), y(2.55), w(3.35), h(2.3), 22, INK, True, "FFFFFF", LINE, True, "ctr")
    body += shape_text(32, "OUR SYNTHESIS", x(4.9), y(1.55), w(3.55), h(0.7), 26, "FFFFFF", True, BLUE, None, True, "ctr")
    body += shape_text(33, "loss metric\nmemory force\nspectral term\nenergy safeguard", x(5.0), y(2.55), w(3.35), h(2.3), 22, INK, True, "FFFFFF", LINE, True, "ctr")
    body += shape_text(34, "HONEST CLAIM", x(8.95), y(1.55), w(3.55), h(0.7), 26, "FFFFFF", True, RED, None, True, "ctr")
    body += shape_text(35, "If no exact reference exists,\nwe call it proposed here.", x(9.05), y(2.85), w(3.35), h(1.45), 24, INK, True, "FFFFFF", LINE, True, "ctr")
    d.add(body, notes="Be very honest here. The math tools are standard. The contribution is the assembly into an optimizer and the experiments. Say: if we do not have a direct reference for a specific design choice, we label it as our proposed modeling choice.")

    # 5
    body = title_block("Optimizer Family Tree", "One visual lens for familiar methods.")
    body += diagram_pipeline(30, ["GD\nflat", "Momentum\ninertia", "Adam\ndiagonal", "Natural\nfull metric", "HG\nmetric + p"], [MUTED, PURPLE, BLUE, TEAL, GREEN], 2.15)
    body += shape_text(65, "Correspondence, not historical authorship.", x(2.15), y(4.55), w(9.0), h(0.75), 27, "FFFFFF", True, INK, None, True, "ctr")
    d.add(body, notes="Use this slide to answer 'how do you know it is derived like this?' We derive by taking limits or choices: flat metric, diagonal metric, no momentum, full metric. Where it is a correspondence rather than a historical derivation, say correspondence.")

    # 6 GIF
    img, p = add_image(d, "visualizations/phase_space/phase_space_rotation.gif", 2, 1.25, 1.2, 10.8, 5.35)
    body = title_block("Motion Matters", "Phase-space trajectory as a moving object, not just a final number.")
    body += img
    body += shape_text(70, "GIF: optimizer trajectory projected into shared phase space", x(2.25), y(6.42), w(8.8), h(0.36), 15, MUTED, False, None, None, False, "ctr")
    d.add(body, [p], notes="This is a visual evidence slide. Emphasize that the method produces a trajectory in phase space, not only a scalar loss curve.")

    # 7
    img, p = add_image(d, "visualizations/pinn_benchmark/optimizer_convergence.png", 2, 0.7, 1.25, 6.0, 4.75)
    img2, p2 = add_image(d, "visualizations/phase_space/phase_space.png", 3, 7.0, 1.25, 5.65, 4.75)
    body = title_block("Physics-Informed Learning", "PINN capacitor benchmark + phase-space path.")
    body += img + img2
    body += shape_text(72, "Strongest current story: structured physics loss + metric-aware motion.", x(1.0), y(6.15), w(11.4), h(0.48), 18, "FFFFFF", True, TEAL, None, True, "ctr")
    d.add(body, [p, p2], notes="For the PINN result, say this is the strongest argument for the framework: physics problems have operators, boundary conditions, and curvature. The phase-space picture makes the optimizer behavior visible.")

    # 8
    img, p = add_image(d, "visualizations/algoperf_style_benchmark/algoperf_style_convergence.gif", 2, 0.75, 1.2, 5.9, 4.9)
    img2, p2 = add_image(d, "visualizations/deepobs_style_benchmark/deepobs_style_convergence.gif", 3, 6.9, 1.2, 5.75, 4.9)
    body = title_block("Training Dynamics", "Two animated local benchmark runs.")
    body += img + img2
    body += shape_text(74, "GIFs: convergence behavior is easier to explain than tables.", x(1.2), y(6.25), w(10.8), h(0.38), 16, MUTED, False, None, None, False, "ctr")
    d.add(body, [p, p2], notes="Be transparent. This builds credibility. Say that the diagonal reduction of our method behaves close to Adam, which is expected. The full framework matters more where full metric structure is available.")

    # 9
    img, p = add_image(d, "visualizations/qaoa_benchmark/qaoa_approximation_ratio.png", 2, 0.65, 1.25, 5.8, 4.75)
    img2, p2 = add_image(d, "visualizations/quantum_chaos_benchmark/quantum_chaos_fidelity.png", 3, 6.9, 1.25, 5.8, 4.75)
    body = title_block("Quantum Optimization Evidence", "QAOA + chaotic kicked-top control.")
    body += img + img2
    body += shape_text(76, "This is the bridge to spin chains and tensor networks.", x(1.25), y(6.15), w(10.8), h(0.48), 19, "FFFFFF", True, PURPLE, None, True, "ctr")
    d.add(body, [p, p2], notes="Connect to spin chains and tensor networks here. Say the current quantum tests are small but directionally relevant: they are closer to the target future use cases than plain image classification.")

    # 10
    img, p = add_image(d, "visualizations/modern_optimizer_benchmark/modern_optimizer_loss.png", 2, 0.65, 1.25, 5.8, 4.7)
    img2, p2 = add_image(d, "visualizations/modern_optimizer_benchmark/modern_optimizer_seed_sweep.png", 3, 6.9, 1.25, 5.8, 4.7)
    body = title_block("Modern Optimizers", "Competitive, not universally dominant.")
    body += img + img2
    body += shape_text(80, "Honest result: Adam / HG / Muon are close; seed variance matters.", x(1.0), y(6.12), w(11.4), h(0.48), 18, "FFFFFF", True, ORANGE, None, True, "ctr")
    d.add(body, [p, p2], notes="This is your progress slide. Use it to show the project is not just an idea. Mention that negative results were kept, including memory metric hurting in the tested quantum benchmark.")

    # 11
    img, p = add_image(d, "visualizations/modern_optimizer_benchmark/modern_optimizer_scaling_sweep.png", 2, 0.7, 1.25, 5.8, 4.7)
    img2, p2 = add_image(d, "visualizations/modern_optimizer_benchmark/modern_optimizer_pareto.png", 3, 6.9, 1.25, 5.8, 4.7)
    body = title_block("Scaling And Cost", "The limitations are visible.")
    body += img + img2
    body += shape_text(86, "Do not oversell: geometry helps when structure justifies the cost.", x(1.2), y(6.12), w(10.9), h(0.48), 18, "FFFFFF", True, RED, None, True, "ctr")
    d.add(body, notes="This slide protects you. Audiences trust you more when you state limitations clearly. It also sets up the next slide: where the method should go next.")

    # 12
    img, p = add_image(d, "visualizations/ablation_study/ablation_study_figure.png", 2, 0.65, 1.25, 5.9, 4.75)
    img2, p2 = add_image(d, "visualizations/quantum_ablation_study/quantum_ablation_fidelity.png", 3, 6.95, 1.25, 5.7, 4.75)
    body = title_block("What We Have Done", "Framework + implementation + ablations + negative results.")
    body += img + img2
    body += shape_text(87, "Current status: working prototype and evidence, not final theory.", x(1.2), y(6.12), w(10.9), h(0.48), 18, "FFFFFF", True, GREEN, None, True, "ctr")
    d.add(body, notes="Tensor networks are probably the strongest future direction. Stress that they already have local geometry and gauge structure, so a geometric optimizer is not artificial there.")

    # 13
    body = title_block("Next: Tensor Networks", "Visual plan: local geometry, local updates, global energy.")
    body += diagram_pipeline(30, ["MPS\nstate", "left/right\nenvironments", "block\nmetric", "HG\nupdate", "gauge +\nenergy check"], [BLUE, TEAL, PURPLE, ORANGE, GREEN], 2.0)
    body += shape_text(60, "Gauge freedom • bond dimension • entanglement spectrum • contraction curvature", x(1.05), y(4.2), w(11.15), h(0.7), 24, INK, True, "FFFFFF", LINE, True, "ctr")
    body += shape_text(61, "First integration target: MPS / MPO variational optimization.", x(1.75), y(5.35), w(9.7), h(0.62), 22, "FFFFFF", True, TEAL, None, True, "ctr")
    d.add(body, notes="This gives a credible physics roadmap. Mention TDVP and DMRG as baselines. Do not claim we have done this yet unless asked; say this is planned integration.")

    # 14
    body = title_block("Next: Spin Chains", "Physical Hamiltonian meets optimizer Hamiltonian.")
    body += shape_text(30, "Ising", x(0.95), y(1.55), w(2.25), h(1.05), 28, "FFFFFF", True, PURPLE, None, True, "ctr")
    body += shape_text(31, "XXZ", x(3.45), y(1.55), w(2.25), h(1.05), 28, "FFFFFF", True, BLUE, None, True, "ctr")
    body += shape_text(32, "J1-J2", x(5.95), y(1.55), w(2.25), h(1.05), 28, "FFFFFF", True, ORANGE, None, True, "ctr")
    body += shape_text(33, "VQE", x(8.45), y(1.55), w(2.25), h(1.05), 28, "FFFFFF", True, GREEN, None, True, "ctr")
    body += shape_text(34, "TDVP", x(10.95), y(1.55), w(1.65), h(1.05), 24, "FFFFFF", True, TEAL, None, True, "ctr")
    body += shape_text(35, "Measure: ground-state energy, fidelity, symmetry-sector stability, entanglement growth, runtime.", x(1.0), y(3.45), w(11.35), h(0.75), 23, INK, True, "FFFFFF", LINE, True, "ctr")
    body += shape_text(36, "First experiment: transverse-field Ising chain with MPS baseline.", x(1.65), y(5.25), w(10.0), h(0.72), 24, "FFFFFF", True, PURPLE, None, True, "ctr")
    d.add(body, notes="This answers the user's request for other higher-order optimization problems. Keep it grounded: the method is worth trying when curvature or structure matters enough to pay extra compute.")

    # 15
    body = title_block("Higher-Order Targets", "Use HG where geometry pays rent.")
    items = ["PDE learning", "Quantum circuits", "Neural ODEs", "Implicit layers", "Meta-learning", "Scientific inverse problems"]
    colors = [BLUE, PURPLE, TEAL, GREEN, ORANGE, RED]
    for i, item in enumerate(items):
        col = i % 3
        row = i // 3
        body += shape_text(30 + i, item, x(0.95 + col * 4.15), y(1.75 + row * 2.15), w(3.45), h(1.2), 25, "FFFFFF", True, colors[i], None, True, "ctr")
    body += shape_text(61, "Rule: if curvature, constraints, or symmetries matter, try the geometric optimizer.", x(1.0), y(6.0), w(11.35), h(0.58), 19, "FFFFFF", True, INK, None, True, "ctr")
    d.add(body, notes="End with a practical plan. The audience should leave knowing what is done, what is honest uncertainty, and what the next milestone is.")

    # 16
    body = title_block("Roadmap", "What we do after this presentation.")
    body += diagram_pipeline(30, ["clean\nAPI", "spin-chain\nbenchmark", "tensor\nbackend", "runtime\nnormalization", "publish\ncarefully"], [GREEN, PURPLE, TEAL, ORANGE, BLUE], 2.0)
    body += shape_text(34, "North star: structured scientific and quantum optimization.", x(1.35), y(4.75), w(10.6), h(0.78), 28, "FFFFFF", True, INK, None, True, "ctr")
    d.add(body, notes="This is a backup or closing slide. It directly addresses the concern about references. Use it if the audience asks what is genuinely novel.")

    # 17
    body = title_block("Attribution Rule", "Use this if someone asks what is new.")
    body += shape_text(30, "Derived from standard mechanics", x(0.9), y(1.65), w(5.35), h(1.25), 25, "FFFFFF", True, GREEN, None, True, "ctr")
    body += shape_text(31, "Proposed here as an optimizer package", x(7.05), y(1.65), w(5.35), h(1.25), 25, "FFFFFF", True, BLUE, None, True, "ctr")
    body += shape_text(32, "equations", x(1.75), y(3.45), w(3.65), h(0.75), 28, INK, True, "FFFFFF", LINE, True, "ctr")
    body += shape_text(33, "design + evidence", x(7.85), y(3.45), w(3.65), h(0.75), 28, INK, True, "FFFFFF", LINE, True, "ctr")
    body += shape_text(34, "If no exact prior reference exists, say it plainly.", x(2.15), y(5.45), w(9.0), h(0.72), 24, "FFFFFF", True, RED, None, True, "ctr")
    d.add(body, notes="This is a backup or closing slide. It directly addresses the concern about references. Use it if the audience asks what is genuinely novel.")

    d.write()


if __name__ == "__main__":
    main()
