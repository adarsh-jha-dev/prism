"""The LangGraph orchestration layer.

Node names are canonical: they are the trace rows and the dashboard's waterfall,
so they match CLAUDE.md exactly.
"""

import os

# langchain-core ships a tracing client. Nothing in this repo sends query text,
# which is tenant data, to a third party.
os.environ.setdefault("LANGSMITH_TRACING", "false")
