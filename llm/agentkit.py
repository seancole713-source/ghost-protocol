"""
OpenAI AgentKit Integration - Full Implementation
Uses OpenAI Assistants API for stateful, persistent agent workflows.
"""

import json
import os
import time
from typing import Any

import requests

# Environment configuration
OPENAI_API_KEY = (os.getenv("OPENAI_AGENT_API_KEY") or os.getenv("OPENAI_API_KEY", "")).strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
AGENT_MODEL = os.getenv("AGENT_MODEL", os.getenv("AI_MODEL", "gpt-4o-mini")).strip()
AGENTKIT_ENABLED = os.getenv("AGENTKIT_ENABLED", "false").lower() in ("true", "1", "yes")
AI_TIMEOUT_S = int(os.getenv("AI_TIMEOUT_S", "30"))

# Assistant configuration
ASSISTANT_NAME = "Ghost WOLF Analyst"
ASSISTANT_INSTRUCTIONS = """You are Ghost's WOLF trading analyst.

Your role:
- Monitor WOLF price, news, and portfolio position
- Provide BUY/SELL/HOLD recommendations with confidence scores
- Track market conditions and risks
- Use available tools to fetch live data
- Maintain context across conversations

Response format (JSON):
{
  "action": "BUY|SELL|HOLD",
  "confidence": 0-100,
  "rationale": "brief explanation",
  "risks": ["risk1", "risk2"],
  "evidence": ["source1", "source2"],
  "checklist": ["item1", "item2"],
  "card": "telegram message text"
}
"""

# Tool definitions for Assistant
ASSISTANT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Get current WOLF price and previous close",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Fetch recent WOLF news articles",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of articles to fetch (default: 5)",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_position",
            "description": "Get user's current WOLF position (qty, avg cost, market value)",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dispatch_alert",
            "description": "Send alert to user via Telegram",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Alert message text"}},
                "required": ["text"],
            },
        },
    },
]


class AgentKitClient:
    """OpenAI Assistants API client for persistent agent workflows."""

    def __init__(self, api_key: str = OPENAI_API_KEY):
        if not api_key:
            raise ValueError("OPENAI_API_KEY or OPENAI_AGENT_API_KEY required for AgentKit")
        self.api_key = api_key
        self.base_url = OPENAI_BASE_URL
        self.assistant_id: str | None = None
        self.thread_id: str | None = None

    def _request(
        self, method: str, path: str, json_data: dict | None = None, params: dict | None = None
    ) -> dict[str, Any]:
        """Make authenticated request to OpenAI API with retry."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2",
        }
        url = f"{self.base_url}/{path.lstrip('/')}"

        last_err = None
        for attempt in range(1, 4):
            try:
                if method.upper() == "GET":
                    r = requests.get(url, headers=headers, params=params, timeout=AI_TIMEOUT_S)
                elif method.upper() == "POST":
                    r = requests.post(url, headers=headers, json=json_data, timeout=AI_TIMEOUT_S)
                elif method.upper() == "DELETE":
                    r = requests.delete(url, headers=headers, timeout=AI_TIMEOUT_S)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                if r.status_code in (429, 500, 502, 503, 504):
                    last_err = RuntimeError(f"HTTP {r.status_code}")
                    time.sleep(min(4.0, 0.5 * (2 ** (attempt - 1))))
                    continue

                r.raise_for_status()
                return r.json() if r.content else {}

            except Exception as e:
                last_err = e
                if attempt < 3:
                    time.sleep(min(4.0, 0.5 * (2 ** (attempt - 1))))

        raise last_err or RuntimeError(f"Request failed: {method} {path}")

    def create_assistant(self) -> str:
        """Create or retrieve existing Ghost assistant."""
        # List existing assistants to avoid duplicates
        assistants = self._request("GET", "/assistants", params={"limit": 20})
        for asst in assistants.get("data", []):
            if asst.get("name") == ASSISTANT_NAME:
                self.assistant_id = asst["id"]
                return self.assistant_id

        # Create new assistant
        payload = {
            "name": ASSISTANT_NAME,
            "instructions": ASSISTANT_INSTRUCTIONS,
            "model": AGENT_MODEL,
            "tools": ASSISTANT_TOOLS,
        }
        result = self._request("POST", "/assistants", json_data=payload)
        self.assistant_id = result["id"]
        return self.assistant_id

    def create_thread(self) -> str:
        """Create a new conversation thread."""
        result = self._request("POST", "/threads", json_data={})
        self.thread_id = result["id"]
        return self.thread_id

    def send_message(self, content: str) -> str:
        """Add user message to thread."""
        if not self.thread_id:
            raise ValueError("No active thread. Call create_thread() first.")

        payload = {"role": "user", "content": content}
        result = self._request("POST", f"/threads/{self.thread_id}/messages", json_data=payload)
        return result["id"]

    def run_assistant(self, tool_router) -> dict[str, Any]:
        """Execute assistant run with tool execution support."""
        if not self.assistant_id or not self.thread_id:
            raise ValueError("Assistant and thread must be created first")

        # Create run
        payload = {"assistant_id": self.assistant_id}
        run = self._request("POST", f"/threads/{self.thread_id}/runs", json_data=payload)
        run_id = run["id"]

        # Poll run status and handle tool calls
        max_iterations = 10
        for _ in range(max_iterations):
            time.sleep(1)  # Polling delay
            run = self._request("GET", f"/threads/{self.thread_id}/runs/{run_id}")
            status = run.get("status")

            if status == "completed":
                # Retrieve assistant's response
                messages = self._request(
                    "GET", f"/threads/{self.thread_id}/messages", params={"limit": 1}
                )
                latest = messages.get("data", [{}])[0]
                content = latest.get("content", [{}])[0]
                text = content.get("text", {}).get("value", "{}")
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"action": "HOLD", "confidence": 50, "rationale": text[:200]}

            elif status == "requires_action":
                # Execute tool calls
                tool_calls = (
                    run.get("required_action", {})
                    .get("submit_tool_outputs", {})
                    .get("tool_calls", [])
                )
                tool_outputs = []

                for call in tool_calls:
                    func_name = call.get("function", {}).get("name")
                    args_str = call.get("function", {}).get("arguments", "{}")
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        args = {}

                    # Execute tool via router
                    try:
                        result = tool_router(func_name, args)
                        output = json.dumps(result)
                    except Exception as e:
                        output = json.dumps({"error": str(e)})

                    tool_outputs.append({"tool_call_id": call["id"], "output": output})

                # Submit tool outputs
                self._request(
                    "POST",
                    f"/threads/{self.thread_id}/runs/{run_id}/submit_tool_outputs",
                    json_data={"tool_outputs": tool_outputs},
                )

            elif status in ("failed", "cancelled", "expired"):
                raise RuntimeError(
                    f"Run {status}: {run.get('last_error', {}).get('message', 'Unknown')}"
                )

        raise TimeoutError("Assistant run exceeded max iterations")

    def delete_assistant(self):
        """Clean up assistant."""
        if self.assistant_id:
            self._request("DELETE", f"/assistants/{self.assistant_id}")
            self.assistant_id = None


def run_agentkit_once(tool_router) -> dict[str, Any]:
    """
    Execute one AgentKit analysis cycle with persistent assistant.

    Args:
        tool_router: Callable that executes tools (get_price, get_news, etc.)

    Returns:
        Decision dict with action, confidence, rationale, etc.
    """
    if not AGENTKIT_ENABLED:
        return {
            "action": "HOLD",
            "confidence": 50,
            "rationale": "AgentKit disabled (set AGENTKIT_ENABLED=true)",
            "risks": [],
            "evidence": [],
            "checklist": [],
            "card": "🔵 AgentKit disabled",
        }

    if not OPENAI_API_KEY:
        return {
            "action": "HOLD",
            "confidence": 50,
            "rationale": "AgentKit requires OPENAI_API_KEY",
            "risks": [],
            "evidence": [],
            "checklist": [],
            "card": "🔴 API key missing",
        }

    try:
        client = AgentKitClient()

        # Create or retrieve assistant
        client.create_assistant()

        # Create new thread for this analysis
        client.create_thread()

        # Send analysis request
        client.send_message(
            "Analyze WOLF now. Use tools to fetch current data and provide a decision with rationale."
        )

        # Run assistant and get decision
        decision = client.run_assistant(tool_router)

        # Normalize decision
        return _normalize_decision(decision)

    except Exception as e:
        return {
            "action": "HOLD",
            "confidence": 0,
            "rationale": f"AgentKit error: {str(e)[:200]}",
            "risks": [str(e)],
            "evidence": [],
            "checklist": [],
            "card": f"⚠️ AgentKit error: {str(e)[:100]}",
        }


def _normalize_decision(obj: dict[str, Any]) -> dict[str, Any]:
    """Ensure decision has required schema and sane bounds."""
    action = str(obj.get("action", "HOLD")).upper()
    if action not in ("BUY", "SELL", "HOLD"):
        action = "HOLD"

    try:
        conf = int(obj.get("confidence", 50))
    except (ValueError, TypeError):
        conf = 50
    conf = max(0, min(100, conf))

    rationale = str(obj.get("rationale", ""))[:500]
    risks = obj.get("risks", [])
    if not isinstance(risks, list):
        risks = [str(risks)]
    evidence = obj.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = [str(evidence)]
    checklist = obj.get("checklist", [])
    if not isinstance(checklist, list):
        checklist = [str(checklist)]
    card = str(obj.get("card", ""))[:2000]

    return {
        "action": action,
        "confidence": conf,
        "rationale": rationale,
        "risks": risks[:10],
        "evidence": evidence[:10],
        "checklist": checklist[:10],
        "card": card,
    }
