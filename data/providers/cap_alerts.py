"""CAP 1.2 ingestion building block. No alert delivery or risk invention.

Authority identity must come from an operator-reviewed feed registry. This parser
does not authenticate a feed, verify a digital signature or determine live state.
"""

from datetime import datetime
import hashlib
from xml.etree import ElementTree as ET

NS = {"cap": "urn:oasis:names:tc:emergency:cap:1.2"}


def parse_cap(raw: bytes, *, authority_id: str, source_url: str) -> dict:
    if len(raw) > 2 * 1024 * 1024 or b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("CAP payload too large or contains forbidden XML declarations")
    root = ET.fromstring(raw)
    if root.tag != f"{{{NS['cap']}}}alert":
        raise ValueError("Expected CAP 1.2 alert")
    def text(element, name, required=False):
        value = (element.findtext(f"cap:{name}", default="", namespaces=NS) or "").strip()
        if required and not value:
            raise ValueError(f"Missing CAP {name}")
        return value or None
    def stamp(value):
        if not value:
            return None
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.utcoffset() is None:
            raise ValueError("CAP timestamp must include timezone")
        return parsed.isoformat()
    result = {"authority_id": authority_id, "source_url": source_url,
              "payload_sha256": hashlib.sha256(raw).hexdigest(), "origin": "official_feed",
              "identifier": text(root, "identifier", True), "sender": text(root, "sender", True),
              "sent": stamp(text(root, "sent", True)), "status": text(root, "status", True),
              "message_type": text(root, "msgType", True), "scope": text(root, "scope", True),
              "references": [], "info": []}
    references = text(root, "references")
    for reference in references.split() if references else []:
        parts = reference.split(",")
        if len(parts) != 3:
            raise ValueError("Malformed CAP reference")
        result["references"].append({"sender": parts[0], "identifier": parts[1], "sent": stamp(parts[2])})
    if result["status"] not in {"Actual", "Exercise", "System", "Test", "Draft"}:
        raise ValueError("Invalid CAP status")
    if result["message_type"] not in {"Alert", "Update", "Cancel", "Ack", "Error"}:
        raise ValueError("Invalid CAP message type")
    if result["message_type"] in {"Update", "Cancel"} and not result["references"]:
        raise ValueError("CAP update/cancel requires references")
    for info in root.findall("cap:info", NS):
        entry = {name: text(info, name) for name in
                 ("language", "event", "urgency", "severity", "certainty", "headline",
                  "description", "instruction", "web", "senderName")}
        for name in ("effective", "onset", "expires"):
            entry[name] = stamp(text(info, name))
        entry["language"] = entry["language"] or "en-US"
        entry["areas"] = []
        for area in info.findall("cap:area", NS):
            entry["areas"].append({"description": text(area, "areaDesc"),
                "polygons_lat_lon": [p.text for p in area.findall("cap:polygon", NS)],
                "circles_lat_lon_km": [c.text for c in area.findall("cap:circle", NS)],
                "geocodes": [{"name": text(g, "valueName"), "value": text(g, "value")}
                             for g in area.findall("cap:geocode", NS)]})
        result["info"].append(entry)
    return result
