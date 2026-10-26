from types import SimpleNamespace

from google.genai import types

from app import ask


def test_ask_sends_and_extends_conversation_history(monkeypatch):
    assistant_content = types.Content(
        role="model",
        parts=[types.Part.from_text(text="最初の回答")],
    )
    responses = [
        SimpleNamespace(
            candidates=[SimpleNamespace(content=assistant_content)],
            text="最初の回答",
            usage_metadata=None,
        ),
        SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text="続きの回答")],
                    )
                )
            ],
            text="続きの回答",
            usage_metadata=None,
        ),
    ]
    requests = []

    class FakeModels:
        def generate_content(self, *, model, contents):
            requests.append((model, contents))
            return responses.pop(0)

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: FakeClient())
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    history = []

    assert ask.ask("最初の質問", history) == "最初の回答"
    assert ask.ask("続きの質問", history) == "続きの回答"

    assert [content.role for content in requests[0][1]] == ["user"]
    assert [content.role for content in requests[1][1]] == [
        "user",
        "model",
        "user",
    ]
    assert requests[1][1][-1].parts[0].text == "続きの質問"
    assert len(history) == 4
