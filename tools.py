# tools.py
from typing import Any, Dict

from brain import (
    predict_is_canceled,
    predict_total_stay_nights,
    predict_deposit_type,
)
from data import get_cancellation_rate   # <--- import your analysis function


def tool_predict_is_canceled(args: Dict[str, Any]) -> str:
    return predict_is_canceled(args)


def tool_predict_total_stay_nights(args: Dict[str, Any]) -> str:
    return predict_total_stay_nights(args)


def tool_predict_deposit_type(args: Dict[str, Any]) -> str:
    return predict_deposit_type(args)


def tool_cancellation_rate(args: Dict[str, Any]) -> str:
    """Return overall cancellation rate from the dataset."""
    return get_cancellation_rate()  # args not needed


# tools.py

TOOLS = {
    "predict_is_canceled": tool_predict_is_canceled,
    "predict_total_stay_nights": tool_predict_total_stay_nights,
    "predict_deposit_type": tool_predict_deposit_type,
    "cancellation_rate": tool_cancellation_rate,
}

TOOL_SCHEMAS = {
    "predict_is_canceled": {
        "description": "Predict if a single booking will be canceled.",
        "args_description": "booking_data with adults, children, ... deposit_type.",
    },
    "predict_total_stay_nights": {
        "description": "Predict how many nights this booking will stay.",
        "args_description": "booking_data with adults, children, ... deposit_type.",
    },
    "predict_deposit_type": {
        "description": "Predict the recommended deposit_type for this booking.",
        "args_description": "booking_data with adults, children, ... deposit_type.",
    },
    "cancellation_rate": {
        "description": "Compute the overall cancellation rate in the dataset.",
        "args_description": "No arguments needed.",
    },
}


def call_tool(name: str, args: Dict[str, Any]) -> str:
    """
    Router gọi đúng tool theo tên.
    name phải là 1 key trong TOOLS, args là dict booking_data hoặc {}.
    """
    if name not in TOOLS:
        raise ValueError(f"Unknown tool: {name}")
    return TOOLS[name](args)
