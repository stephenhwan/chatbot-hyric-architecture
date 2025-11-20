import json
from typing import Any, Dict

from tools import call_tool, TOOL_SCHEMAS
from engine import BackendRouter, CURRENT_BACKEND, chat

validation_key = {
    "predict_is_canceled": {"adults", "children"},
    "predict_total_stay_nights": {"adults", "children"},
    "predict_deposit_type": {"adults", "children"},
}


def build_router_prompt(user_input: str) -> str:
    """Build system prompt for LLM router."""
    tool_desc_lines = []
    for name, schema in TOOL_SCHEMAS.items():
        tool_desc_lines.append(f"- {name}: {schema['description']}")
    tool_desc = "\n".join(tool_desc_lines)

    system_instructions = f"""
    You are a router for a hotel booking ML assistant. You have these tools:

    {tool_desc}

    Your job: Decide whether to ANSWER NORMALLY or CALL ONE TOOL.

    You MUST output a SINGLE JSON object with one of these shapes:

    1) For normal chat (no tool):
       {{
         "mode": "chat",
         "answer": "your natural language answer here"
       }}

    2) For calling a tool:
       {{
         "mode": "tool_call",
         "tool": "predict_is_canceled" | "predict_total_stay_nights" | "predict_deposit_type" | "cancellation_rate",
         "arguments": {{ "adults": 2, "children": 1, "babies": 0, ... }}
       }}

    CRITICAL RULES for tool arguments:

    - For prediction tools (predict_is_canceled, predict_total_stay_nights, predict_deposit_type):
      * ALWAYS extract booking details from the user's question
      * ALWAYS include "adults", "children", and "babies" in arguments (use 0 if not mentioned)
      * Example: "2 adults 2 children stay 5 days" → {{"adults": 2, "children": 2, "babies": 0}}
      * You can also include other fields like: adr, market_segment, distribution_channel, etc.

    - For cancellation_rate tool:
      * No arguments needed, use empty object: "arguments": {{}}

    General Rules:

    - ONLY call tools when the user is asking about HOTEL BOOKINGS or the hotel_booking.csv dataset
    - For general questions about ML theory, definitions, concepts: use "mode": "chat"
    - Do NOT output explanations, comments, or multiple JSON objects
    - Output EXACTLY ONE JSON object, and nothing else
    """

    return system_instructions + f"\n\nUser: {user_input}\nAssistant:"


def parse_llm_response(text: str) -> Dict[str, Any]:

    # Try strict JSON parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON block
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start: end + 1])
        except json.JSONDecodeError:
            pass

    # Fallback: treat as chat response
    return {
        "mode": "chat",
        "answer": text or "LLM returned empty response."
    }


def is_valid_tool_call(tool_name: str, arguments: Dict[str, Any]) -> bool:
    # Tool doesn't exist
    if tool_name not in TOOL_SCHEMAS:
        print(f"[VALIDATION] Tool '{tool_name}' not found in TOOL_SCHEMAS")
        return False

    # cancellation_rate: no args needed
    if tool_name == "cancellation_rate":
        return True

    # For ML prediction tools: need valid booking data
    required_keys = validation_key.get(tool_name)
    if not required_keys:
        print(f"[VALIDATION] No validation keys defined for '{tool_name}'")
        return False

    if not isinstance(arguments, dict):
        print(f"[VALIDATION] Arguments is not a dict: {type(arguments)}")
        return False

    # Check if at least ONE required key is present
    provided_keys = set(arguments.keys())
    has_required = bool(provided_keys & required_keys)

    if not has_required:
        print(f"[VALIDATION] Missing required keys. Expected one of {required_keys}, got {provided_keys}")
        return False

    print(f"[VALIDATION] ✓ Tool '{tool_name}' validated successfully")
    return True


def llm_route_and_call(user_input: str) -> str:
    prompt = build_router_prompt(user_input)

    # Get LLM routing decision
    try:
        llm_response = BackendRouter.route(
            msg=prompt,
            history=[],
            backend=CURRENT_BACKEND
        )
    except Exception as e:
        return f"❌ All LLM backends failed: {e}"

    # Parse LLM response
    obj = parse_llm_response(llm_response)
    mode = obj.get("mode")

    print(f"[ROUTER] LLM Response: {obj}")  # Debug output

    # Handle chat mode
    if mode == "chat":
        answer = obj.get("answer") or "(empty answer)"
        return answer

    # Handle tool_call mode
    if mode == "tool_call":
        tool_name = obj.get("tool")
        arguments = obj.get("arguments") or {}

        print(f"[ROUTER] Tool: {tool_name}, Arguments: {arguments}")

        # Validate tool call
        if not is_valid_tool_call(tool_name, arguments):
            print(f"[ROUTER] Tool call validation failed, falling back to chat")
            chat_answer, _ = chat(user_input)
            return chat_answer

        # Execute tool
        try:
            result = call_tool(tool_name, arguments)
            print(f"[ROUTER] Tool result: {result}")
        except Exception as e:
            return f"Error when calling tool {tool_name}: {e}"

        # Generate natural language explanation
        explain_prompt = (
            "You are a hotel booking data analyst.\n"
            "The user asked the following question:\n"
            f'"{user_input}"\n\n'
            f"You ran an internal analysis tool named '{tool_name}' and got this result:\n"
            "-----\n"
            f"{result}\n"
            "-----\n\n"
            "Now, answer the user directly:\n"
            "- Present key numbers clearly\n"
            "- Give 1-2 lines of overall evaluation\n"
            "- Be concise and friendly\n"
        )

        final_answer, _ = chat(explain_prompt)
        return final_answer

    # Unknown mode
    return "❌ LLM returned unknown mode. Please try again."


def main():
    print("=" * 70)
    print("🏨 Hotel ML Assistant (LLM Function-Calling + Auto-Fallback)")
    print("=" * 70)
    print("Backend: Gemini (with auto-fallback to local LLM)")
    print("\nAsk about:")
    print("  • Booking predictions (cancellation, stay nights, deposit type)")
    print("  • Dataset analysis (cancellation rates, segments)")
    print("\nType 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "bye"}:
            print("👋 Bye!")
            break

        try:
            answer = llm_route_and_call(user_input)
        except Exception as e:
            answer = f"💥 Fatal error: {e}"

        print("\n🤖 Assistant:")
        print(answer)
        print("-" * 70 + "\n")


if __name__ == "__main__":
    main()