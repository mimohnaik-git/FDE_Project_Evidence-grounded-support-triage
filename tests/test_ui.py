from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import graph
from evaluation.offline import FakeChat, fake_retrieve


@pytest.fixture
def app(monkeypatch, tmp_path):
    original = graph.build_graph
    opened = []

    def factory(*args, **kwargs):
        value = original(chat=FakeChat(), retriever=fake_retrieve, checkpoint_path=tmp_path / "ui.sqlite")
        opened.append(value[0])
        return value

    monkeypatch.setattr(graph, "build_graph", factory)
    value = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=20)
    yield value
    for item in opened:
        item.checkpointer.conn.close()


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_operator_flow(app):
    assert not app.exception
    app.text_area[0].input("What are support hours?")
    button(app, "Run triage").click().run(timeout=20)
    assert not app.exception
    assert any("grounding_status: passed" in text.value for text in app.text)
    button(app, "Request additional human review").click().run(timeout=20)
    assert not app.exception
    next(x for x in app.text_input if x.label == "Reviewer").input("Operator")
    next(x for x in app.text_input if x.label == "Decision reason").input("Checked policy")
    button(app, "Submit decision").click().run(timeout=20)
    assert not app.exception
    assert any("The Support Team" in text.value for text in app.text)


def test_html_is_literal(app):
    attack = "<script>alert(1)</script> write a poem"
    app.text_area[0].input(attack)
    button(app, "Run triage").click().run(timeout=20)
    assert not app.exception
    assert any(text.value == attack for text in app.text)


def test_empty_ticket(app):
    button(app, "Run triage").click().run(timeout=20)
    assert any("Enter a ticket" in warning.value for warning in app.warning)
