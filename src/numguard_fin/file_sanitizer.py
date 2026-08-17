from __future__ import annotations

import struct
import tempfile
import zipfile
import zlib
from pathlib import Path
from xml.etree import ElementTree as ET


def sanitise_xlsx(path: str | Path) -> None:
    path = Path(path)
    rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    ct_ns = "http://schemas.openxmlformats.org/package/2006/content-types"
    with zipfile.ZipFile(path, "r") as src, tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        temp_path = Path(tmp.name)
    try:
        with zipfile.ZipFile(path, "r") as src, zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as dst:
            for info in src.infolist():
                name = info.filename
                if name.startswith("docProps/"):
                    continue
                data = src.read(name)
                if name == "_rels/.rels":
                    root = ET.fromstring(data)
                    for child in list(root):
                        if child.attrib.get("Type", "").endswith(("/metadata/core-properties", "/extended-properties", "/custom-properties")):
                            root.remove(child)
                    data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
                elif name == "[Content_Types].xml":
                    root = ET.fromstring(data)
                    for child in list(root):
                        if child.attrib.get("PartName", "").startswith("/docProps/"):
                            root.remove(child)
                    data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
                dst.writestr(info.filename, data)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def sanitise_png(path: str | Path) -> None:
    path = Path(path)
    raw = path.read_bytes()
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        return
    out = bytearray(raw[:8])
    offset = 8
    remove = {b"tEXt", b"zTXt", b"iTXt", b"tIME", b"eXIf", b"pHYs"}
    while offset + 12 <= len(raw):
        length = struct.unpack(">I", raw[offset:offset+4])[0]
        chunk_type = raw[offset+4:offset+8]
        chunk_data = raw[offset+8:offset+8+length]
        offset += 12 + length
        if chunk_type in remove:
            continue
        out.extend(struct.pack(">I", length))
        out.extend(chunk_type)
        out.extend(chunk_data)
        out.extend(struct.pack(">I", zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF))
        if chunk_type == b"IEND":
            break
    path.write_bytes(bytes(out))
