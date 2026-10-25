"""The web app's routes, including driving a (pretend) plotter."""

import io
import time

import cv2
import pytest

import app as webapp


@pytest.fixture
def client():
    webapp.app.config["TESTING"] = True
    yield webapp.app.test_client()
    if webapp._machine.connected:
        webapp._machine.disconnect()


LIGHT_FORM = {"use_depth": "0", "line_method": "coherent", "subject_focus": "0",
              "resolution": "300", "outline": "1"}


def _generate(client, photo, **form):
    ok, png = cv2.imencode(".png", photo)
    data = {"image": (io.BytesIO(png.tobytes()), "test.png"), **LIGHT_FORM, **form}
    return client.post("/generate", data=data, content_type="multipart/form-data").get_json()


def test_index_lists_featured_styles_first(client):
    html = client.get("/").get_data(as_text=True)
    order = [html.index(f'value="{e}"') for e in ("illustration", "waves", "hilbert", "crosshatch")]
    assert order == sorted(order)
    assert "More styles" in html and "Take a photo" in html


def test_generate_returns_preview_and_stats(client, photo):
    d = _generate(client, photo, engine="waves")
    assert "error" not in d
    assert d["preview"] and d["stats"]["lifts"] > 0
    assert client.get("/download").status_code == 200


def test_procedural_names_the_file(client):
    d = client.post("/procedural", data={"subject": "toad", "name": "Bufo testii"}).get_json()
    assert d["name"] == "Bufo testii"


def test_plotter_flow(client, photo, fake):
    d = client.post("/machine/connect", data={"port": fake.path}).get_json()
    assert d["ok"] and d["info"]["version"] == "Grbl 0.9i"

    # A portrait page goes to the bed turned anticlockwise, so its origin is
    # the drawing's top-right corner and "right" on the preview is machine +Y.
    _generate(client, photo, engine="illustration", paper="a4", portrait="1", rotate_out="1")
    st = client.get("/machine/status").get_json()
    assert st["origin_corner"] == "top-right"
    client.post("/machine/jog", data={"du": 10, "dv": 0})
    time.sleep(0.4)
    st = client.get("/machine/status").get_json()
    assert (st["x"], st["y"]) == pytest.approx((0, 10))

    client.post("/machine/origin")
    assert client.post("/machine/frame").get_json()["ok"]
    while client.get("/machine/status").get_json()["job"]["state"] == "running":
        time.sleep(0.05)

    assert client.post("/machine/start").get_json()["ok"]
    while client.get("/machine/status").get_json()["job"]["state"] == "running":
        time.sleep(0.05)
    job = client.get("/machine/status").get_json()["job"]
    assert job["state"] == "done" and not job["errors"]
    assert not fake.overflow


def test_a_drawing_bigger_than_the_bed_asks_first(client, photo, fake):
    client.post("/machine/connect", data={"port": fake.path})
    _generate(client, photo, engine="illustration", paper="a3")    # 420 x 297 > 300 x 210
    d = client.post("/machine/start").get_json()
    assert not d["ok"] and "bed" in d["confirm"]


def test_lettering_route(client):
    d = client.post("/lettering", data={"text": "# Grüezi\nEin kleiner Test.",
                                        "style": "book", "align": "center"}).get_json()
    assert "error" not in d and d["stats"]["lifts"] > 0
    assert webapp._last["name"] == "gruezi.gcode"
