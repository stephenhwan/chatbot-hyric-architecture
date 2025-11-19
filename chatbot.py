
import json
from typing import Any, Dict

from tools import call_tool, TOOL_SCHEMAS
from engine import BackendRouter, CURRENT_BACKEND, chat  # ← FIX: Import BackendRouter

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
         "arguments": {{ ... }}
       }}

    Rules (VERY IMPORTANT):

    - ONLY call tools when the user is asking about HOTEL BOOKINGS or the hotel_booking.csv dataset:
      * whether a booking will be canceled,
      * how many nights a booking will stay,
      * what deposit_type should be used,
      * or the overall cancellation rate in the dataset.

    - For ALL OTHER questions (for example:
      * questions about machine learning theory such as "what is regression vs classification",
      * questions about definitions, concepts, general AI, etc.),
      you MUST use:
        "mode": "chat"
      and provide a direct explanation in "answer".

    - Do NOT output explanations, comments, or multiple JSON objects.
      Output EXACTLY ONE JSON object, and nothing else.
    """

    return system_instructions + f"\n\nUser: {user_input}\nAssistant:"


def parse_llm_response(text: str) -> Dict[str, Any]:
    """
    Parse LLM response to extract JSON.
    Handles cases where model wraps JSON in explanation.
    """
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
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Fallback: treat as chat response
    return {
        "mode": "chat",
        "answer": text or "LLM returned empty response."
    }

def is_valid_tool_call(tool_name: str, arguments: Dict[str, Any]) -> bool:
    # Tool không tồn tại
    if tool_name not in TOOL_SCHEMAS:
        return False

    # cancellation_rate: không cần args, cho qua
    if tool_name == "cancellation_rate":
        return True

    # Các tool ML còn lại: cần ít nhất SOME key đặc trưng của booking
    required_keys = validation_key.get(tool_name)
    if not required_keys:
        return False

    if not isinstance(arguments, dict):
        return False

    # Nếu arguments rỗng, hoặc không có key nào trong required_keys → coi như invalid
    provided_keys = set(arguments.keys())
    if not (provided_keys & required_keys):
        return False

    return True

def llm_route_and_call(user_input: str) -> str:

    prompt = build_router_prompt(user_input)

    try:
        # ✅ FIX: Use BackendRouter instead of direct Gemini call
        llm_response = BackendRouter.route(
            msg=prompt,
            history=[],  # No conversation history for routing
            backend=CURRENT_BACKEND  # "auto" by default
        )
    except Exception as e:
        return f"❌ All LLM backends failed: {e}"

    # Parse LLM response
    obj = parse_llm_response(llm_response)
    mode = obj.get("mode")

    # Handle chat mode
    if mode == "chat":
        answer = obj.get("answer") or "(empty answer)"
        return answer

    # Handle tool_call mode
    if mode == "tool_call":
        tool_name = obj.get("tool")
        arguments = obj.get("arguments") or {}

        # 1) Validate tool-call dựa trên schema
        if not is_valid_tool_call(tool_name, arguments):
            # => LLM định gọi tool nhưng arguments quá vô nghĩa
            #    => fallback: coi như chat thường
            chat_answer, _ = chat(user_input)
            return chat_answer

        # 2) Hợp lệ -> thực thi tool
        try:
            result = call_tool(tool_name, arguments)
        except Exception as e:
            return f"Error when calling tool {tool_name}: {e}"

        explain_prompt = (
            "You are a hotel booking data analyst.\n"
            "The user asked the following question:\n"
            f"\"{user_input}\"\n\n"
            f"You ran an internal analysis tool named '{tool_name}' and got this result:\n"
            "-----\n"
            f"{result}\n"
            "-----\n\n"
            "Now, answer the user directly:\n"
            "- List key numbers in bullet points.\n"
            "- Then, give 1-2 lines of overall evaluation, focusing on key insights or trends.\n"
            "- Be concise and friendly.\n"
        )

        final_answer, _ = chat(explain_prompt)
        return final_answer



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