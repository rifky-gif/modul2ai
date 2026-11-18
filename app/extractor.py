import csv
import io
import re
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from pypdf import PdfReader
from docx import Document as DocxDocument

LIGATURE_MAP = {
    "": "ft",
    "": "ti",
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}


def clean_extracted_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for bad, rep in LIGATURE_MAP.items():
        text = text.replace(bad, rep)
    # Hapus sisa karakter Private Use Area (PUA) yang tidak terbaca
    text = re.sub(r"[-]", "", text)
    # Rapikan pecahan rumus IKK yang terpecah antarbaris saat ekstraksi PDF
    text = re.sub(r"IKK\s*=\s*\n\s*Ni\s*\n\s*Nm\s*[×x]\s*4", "IKK = (Ni / Nm) × 4", text)
    text = re.sub(r"Ni\s*\n\s*IKK\s*=\s*[×x]\s*4\s*\n\s*Nm", "IKK = (Ni / Nm) × 4", text)
    return text.strip()


def _rows_to_markdown_table(rows: list[list[str]]) -> str:
    cleaned_rows = []
    for row in rows:
        cells = [re.sub(r"\s+", " ", str(c or "")).strip().replace("|", "\\|") for c in row]
        if any(cells):
            cleaned_rows.append(cells)
    if not cleaned_rows:
        return ""

    max_cols = max(len(r) for r in cleaned_rows)
    padded = [r + [""] * (max_cols - len(r)) for r in cleaned_rows]
    header = "| " + " | ".join(padded[0]) + " |"
    sep = "| " + " | ".join(["---"] * max_cols) + " |"
    body = ["| " + " | ".join(r) + " |" for r in padded[1:]]
    return "\n".join([header, sep, *body])


def _col_ref_to_index(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha()).upper()
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return max(0, idx - 1)


def _extract_xlsx_bytes(file_bytes: bytes) -> str:
    ns_main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ns_rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall(f"{{{ns_main}}}si"):
                text_parts = [t.text or "" for t in si.iter(f"{{{ns_main}}}t")]
                shared_strings.append("".join(text_parts))

        sheet_entries: list[tuple[str, str]] = []
        if "xl/workbook.xml" in zf.namelist() and "xl/_rels/workbook.xml.rels" in zf.namelist():
            wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
            rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
            rel_map = {
                rel.attrib.get("Id", ""): rel.attrib.get("Target", "").lstrip("/")
                for rel in rels_root
            }
            for sheet in wb_root.findall(f".//{{{ns_main}}}sheet"):
                name = sheet.attrib.get("name", "Sheet")
                rid = sheet.attrib.get(f"{{{ns_rel}}}id", "")
                target = rel_map.get(rid, "")
                if target and not target.startswith("xl/"):
                    target = f"xl/{target}"
                if target in zf.namelist():
                    sheet_entries.append((name, target))

        if not sheet_entries:
            for name in sorted(zf.namelist()):
                if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"):
                    sheet_entries.append((name.rsplit("/", 1)[-1].replace(".xml", ""), name))

        sections: list[str] = []
        for sheet_name, sheet_path in sheet_entries:
            s_root = ET.fromstring(zf.read(sheet_path))
            rows_data: list[list[str]] = []
            for row_el in s_root.findall(f".//{{{ns_main}}}row"):
                row_vals: dict[int, str] = {}
                for cell in row_el.findall(f"{{{ns_main}}}c"):
                    ref = cell.attrib.get("r", "")
                    col_idx = _col_ref_to_index(ref) if ref else len(row_vals)
                    c_type = cell.attrib.get("t", "")
                    val = ""
                    if c_type == "inlineStr":
                        val = "".join(t.text or "" for t in cell.iter(f"{{{ns_main}}}t"))
                    else:
                        v_el = cell.find(f"{{{ns_main}}}v")
                        if v_el is not None and v_el.text is not None:
                            raw_val = v_el.text
                            if c_type == "s":
                                try:
                                    val = shared_strings[int(raw_val)]
                                except (ValueError, IndexError):
                                    val = raw_val
                            elif c_type == "b":
                                val = "TRUE" if raw_val == "1" else "FALSE"
                            else:
                                val = raw_val
                    row_vals[col_idx] = val
                if row_vals:
                    max_c = max(row_vals.keys())
                    rows_data.append([row_vals.get(i, "") for i in range(max_c + 1)])

            md_table = _rows_to_markdown_table(rows_data)
            if md_table:
                if len(sheet_entries) > 1:
                    sections.append(f"### {sheet_name}\n\n{md_table}")
                else:
                    sections.append(md_table)

        return "\n\n".join(sections).strip()


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return clean_extracted_text("\n".join(pages))

    if ext == "docx":
        doc = DocxDocument(io.BytesIO(file_bytes))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            table_rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
            md_table = _rows_to_markdown_table(table_rows)
            if md_table:
                parts.append(md_table)
        return "\n\n".join(parts).strip()

    if ext == "xlsx":
        return _extract_xlsx_bytes(file_bytes)

    if ext == "csv":
        decoded = file_bytes.decode("utf-8-sig", errors="ignore").strip()
        if not decoded:
            return ""
        rows = list(csv.reader(io.StringIO(decoded)))
        md_table = _rows_to_markdown_table(rows)
        return md_table or decoded

    if ext in {"txt", "md"}:
        return file_bytes.decode("utf-8-sig", errors="ignore").strip()

    raise ValueError(f"Format file '.{ext}' belum didukung. Gunakan PDF, DOCX, XLSX, CSV, atau TXT.")
