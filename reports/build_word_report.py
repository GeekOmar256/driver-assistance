import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image

ROOT = Path(__file__).resolve().parent
doc = Document()
section = doc.sections[0]
section.page_width, section.page_height = Inches(8.5), Inches(11)
section.top_margin = section.bottom_margin = Inches(.7)
section.left_margin = section.right_margin = Inches(.75)
for style_name in ('Normal', 'Title', 'Heading 1', 'Heading 2', 'Heading 3'):
    style = doc.styles[style_name]
    style.font.name = 'Calibri'
    style.font.color.rgb = RGBColor(0, 0, 0)
normal = doc.styles['Normal']
normal.font.size = Pt(11)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.08
for style in doc.styles:
    for border in list(style.element.iter(qn('w:pBdr'))):
        border.getparent().remove(border)
doc.styles['Title'].font.size = Pt(23)
doc.styles['Title'].paragraph_format.space_after = Pt(12)
for name, size in [('Heading 1', 15), ('Heading 2', 12)]:
    style = doc.styles[name]
    style.font.size = Pt(size)
    style.font.bold = True
    style.paragraph_format.space_before = Pt(14)
    style.paragraph_format.space_after = Pt(7)
    style.paragraph_format.keep_with_next = True


def hyperlink(paragraph, label, target):
    if target.startswith('E:/'):
        target = Path(target).as_uri()
    rid = paragraph.part.relate_to(target, 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink', is_external=True)
    link = OxmlElement('w:hyperlink')
    link.set(qn('r:id'), rid)
    run = OxmlElement('w:r')
    properties = OxmlElement('w:rPr')
    color = OxmlElement('w:color'); color.set(qn('w:val'), '244B70'); properties.append(color)
    run.append(properties)
    text = OxmlElement('w:t'); text.text = label; run.append(text)
    link.append(run); paragraph._p.append(link)


TOKEN = re.compile(r'(\[[^\]]+\]\((?:<[^>]+>|[^)]+)\)|\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)')


def inline(paragraph, text):
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith('['):
            match = re.match(r'\[([^]]+)\]\((.*)\)', part)
            hyperlink(paragraph, match[1], match[2].strip('<>'))
        elif part.startswith('**'):
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith('`'):
            run = paragraph.add_run(part[1:-1]); run.font.name = 'Consolas'; run.font.size = Pt(9.5)
        elif part.startswith('*'):
            paragraph.add_run(part[1:-1]).italic = True
        else:
            paragraph.add_run(part)


def table(rows):
    count = len(rows[0])
    widths = {2: [2.5, 4.5], 3: [3.8, 1.6, 1.6],
              5: [.35, 2.45, 1.4, 1.4, 1.4], 6: [2.0, .7, 1.1, 1.0, 1.1, 1.1]}.get(count, [7/count]*count)
    if rows[0][0] == 'Prepared split':
        widths = [1.0, .7, 1.8, 1.7, 1.8]
    result = doc.add_table(rows=0, cols=count)
    result.autofit = False
    for col, width in zip(result.columns, widths):
        col.width = Inches(width)
    props = result._tbl.tblPr
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        border = OxmlElement('w:' + edge)
        for key, value in [('val', 'single'), ('sz', '4'), ('color', 'D9D9D9')]:
            border.set(qn('w:' + key), value)
        borders.append(border)
    props.append(borders)
    for i, row_data in enumerate(rows):
        row = result.add_row()
        trpr = row._tr.get_or_add_trPr()
        trpr.append(OxmlElement('w:cantSplit'))
        if i == 0:
            header = OxmlElement('w:tblHeader'); header.set(qn('w:val'), 'true'); trpr.append(header)
        for j, (cell, text) in enumerate(zip(row.cells, row_data)):
            cell.width = Inches(widths[j])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcpr = cell._tc.get_or_add_tcPr()
            margin = OxmlElement('w:tcMar')
            for side in ('top', 'bottom', 'left', 'right'):
                element = OxmlElement('w:' + side)
                element.set(qn('w:w'), '85'); element.set(qn('w:type'), 'dxa'); margin.append(element)
            tcpr.append(margin)
            shading = OxmlElement('w:shd')
            shading.set(qn('w:fill'), '244B70' if i == 0 else ('F2F5F8' if i % 2 else 'FFFFFF'))
            tcpr.append(shading)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.keep_with_next = i == 0
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if count == 2 or j == 0 or (count == 5 and j == 1 and rows[0][0]=='ID') else WD_ALIGN_PARAGRAPH.CENTER
            inline(p, text)
            for run in p.runs:
                run.font.size = Pt(10)
                if i == 0:
                    run.bold = True; run.font.color.rgb = RGBColor(255, 255, 255)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)
    spacer.paragraph_format.space_before = Pt(0)
    spacer.paragraph_format.line_spacing = .5
    spacer.add_run().font.size = Pt(3)


lines = (ROOT / 'Dataset_Training_and_Testing_Report.md').read_text(encoding='utf-8').splitlines()
i = 0
while i < len(lines):
    line = lines[i].strip()
    if not line:
        i += 1; continue
    if line.startswith('```'):
        i += 1
        while i < len(lines) and not lines[i].startswith('```'):
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(1)
            r = p.add_run(lines[i]); r.font.name = 'Consolas'; r.font.size = Pt(9)
            i += 1
        i += 1; continue
    if line.startswith('|'):
        rows = []
        while i < len(lines) and lines[i].strip().startswith('|'):
            values = [v.strip() for v in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?', v) for v in values):
                rows.append(values)
            i += 1
        table(rows); continue
    if line.startswith('!['):
        match = re.match(r'!\[([^]]*)\]\(<([^>]+)>\)', line)
        path = Path(match[2])
        with Image.open(path) as img:
            width, height = img.size
        max_height = 4.25 if path.name == 'test_confusion_matrix.png' else 5.5
        width_inches = min(7.0, max_height*width/height)
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        picture = p.add_run().add_picture(str(path), width=Inches(width_inches))
        picture._inline.docPr.set('descr', match[1])
        i += 1; continue
    if line.startswith('#'):
        level = len(line) - len(line.lstrip('#'))
        title = line[level:].strip().replace(':', '').replace(',', '')
        if level == 1:
            doc.add_paragraph(title, style='Title')
        else:
            doc.add_heading(title, level=level-1)
        i += 1; continue
    if re.match(r'^\d+\. ', line):
        p = doc.add_paragraph(style='List Number')
        inline(p, re.sub(r'^\d+\. ', '', line)); i += 1; continue
    parts = [line]
    if not line.endswith('  ') and not lines[i].endswith('  '):
        while i+1 < len(lines) and lines[i+1].strip() and not lines[i+1].startswith(('#', '|', '![', '```')) and not re.match(r'^\d+\. ', lines[i+1]):
            i += 1; parts.append(lines[i].strip())
    p = doc.add_paragraph()
    if line.startswith('*Figure'):
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    inline(p, ' '.join(parts))
    i += 1

footer = section.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
run = footer.add_run('Page '); run.font.size = Pt(9)
field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'), 'PAGE'); footer._p.append(field)
doc.core_properties.title = 'VisionGuard Dataset Training and Testing Report'
doc.core_properties.subject = 'Nine class YOLOv8n fine tuning and held out test results'
doc.save(ROOT / 'Dataset_Training_and_Testing_Report.docx')
print('Saved Word report')
