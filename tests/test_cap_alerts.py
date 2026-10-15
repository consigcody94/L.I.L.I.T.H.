import pytest

from data.providers.cap_alerts import parse_cap


def payload(message_type="Alert", status="Actual", extra=""):
    return f'''<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
    <identifier>id-2</identifier><sender>authority.example</sender><sent>2026-01-02T12:00:00Z</sent>
    <status>{status}</status><msgType>{message_type}</msgType><scope>Public</scope>{extra}
    <info><language>en-US</language><event>Heavy rain</event><severity>Severe</severity>
    <expires>2026-01-02T18:00:00Z</expires><instruction>Official instructions.</instruction>
    <area><areaDesc>District</areaDesc><polygon>1,2 1,3 2,3 1,2</polygon></area></info>
    <info><language>fr</language><event>Pluie</event></info></alert>'''.encode()


def parse(raw):
    return parse_cap(raw, authority_id="reviewed-test-authority", source_url="https://authority.example/feed")


def test_keeps_official_identity_multilingual_content_and_lat_lon_order():
    result = parse(payload())
    assert result["origin"] == "official_feed"
    assert result["info"][1]["language"] == "fr"
    assert result["info"][0]["areas"][0]["polygons_lat_lon"] == ["1,2 1,3 2,3 1,2"]
    assert result["info"][0]["expires"].endswith("+00:00")
    assert len(result["payload_sha256"]) == 64


def test_cancellation_keeps_reference_and_test_status_is_not_promoted():
    result = parse(payload("Cancel", "Test", '<references>authority.example,id-1,2026-01-02T11:00:00Z</references>'))
    assert result["status"] == "Test"
    assert result["references"][0]["identifier"] == "id-1"
    with pytest.raises(ValueError, match="references"):
        parse(payload("Cancel"))


def test_rejects_entities_and_naive_timestamps():
    with pytest.raises(ValueError):
        parse(b'<!DOCTYPE alert [<!ENTITY x "value">]>' + payload())
    with pytest.raises(ValueError, match="timezone"):
        parse(payload().replace(b"2026-01-02T12:00:00Z", b"2026-01-02T12:00:00"))
