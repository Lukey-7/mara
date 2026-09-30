def test_index_page_is_served(make_client):
    with make_client() as c:
        r = c.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "MARA" in r.text and "/research" in r.text and "EventSource" in r.text
