"""OpenTelemetry tracing to a local Phoenix instance (Docker, default port 6006)."""
from contextlib import contextmanager

from app.config import PHOENIX_ENDPOINT, PHOENIX_PROJECT

_registered = False


def setup_tracing() -> None:
    """Idempotently register the Phoenix tracer and auto-instrument LangChain/LangGraph and OpenAI."""
    global _registered
    if _registered:
        return
    from phoenix.otel import register

    register(
        project_name=PHOENIX_PROJECT,
        endpoint=PHOENIX_ENDPOINT,
        auto_instrument=True,  # picks up installed openinference instrumentors
        batch=True,
    )
    _registered = True


@contextmanager
def turn_span(session_id: str, user_input: str, name: str = "chat_turn"):
    """Root span for ONE user turn.

    Started with an empty OTel Context so it never inherits a parent (e.g. an HTTP request span or a
    previous turn's span) and therefore always gets a fresh trace_id. The conversation is tied together
    via the OpenInference `session.id` attribute (= thread_id), which Phoenix groups under Sessions.
    LangChain/LangGraph spans created inside become children of this root span.
    """
    from openinference.instrumentation import using_session
    from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
    from opentelemetry import trace
    from opentelemetry.context import Context

    tracer = trace.get_tracer("app.agent")
    with using_session(session_id):
        with tracer.start_as_current_span(
            name,
            context=Context(),
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                SpanAttributes.SESSION_ID: session_id,
                SpanAttributes.INPUT_VALUE: user_input,
            },
        ) as span:
            yield span
