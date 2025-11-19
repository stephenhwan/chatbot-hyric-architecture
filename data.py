import pandas as pd
from sklearn.model_selection import train_test_split

ALWAYS_DROP = ["booking_id", "reservation_status"]
VERBOSE = False  # change to True if you want to see debug logs


def _log(msg: str) -> None:
    if VERBOSE:
        print(msg)


def load_data(csv_file="hotel_bookings.csv"):
    df = pd.read_csv(csv_file)
    _log(f"[load_data] {csv_file}: {len(df)} rows, {len(df.columns)} columns")
    return df


def clean_data(df):
    df = df.copy()

    # Fill numeric NaN with 0
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(0)

    # Fill non-numeric NaN with "Unknown"
    cat_cols = df.select_dtypes(exclude="number").columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    # Remove rows with no guests
    if {"adults", "children", "babies"}.issubset(df.columns):
        before = len(df)
        df = df[(df["adults"] + df["children"] + df["babies"]) > 0]
        _log(f"[clean_data] Removed {before - len(df)} rows with no guests")

    _log(f"[clean_data] Remaining rows: {len(df)}")
    return df


def get_features(df, target=None):
    cols = list(df.columns)

    # drop target
    if target in cols:
        cols.remove(target)

    # drop always-drop columns
    for c in ALWAYS_DROP:
        if c in cols:
            cols.remove(c)

    _log(f"[get_features] {len(cols)} features: {cols}")
    return cols


def get_feature_types(df, target=None):

    features = get_features(df, target=target)

    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]

    _log(f"[get_feature_types] numeric: {numeric}")
    _log(f"[get_feature_types] categorical: {categorical}")
    return numeric, categorical


def prepare_data(df, target="is_canceled"):
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in DataFrame")

    # drop rows with NaN target (just in case)
    df = df[df[target].notna()]

    feature_cols = get_features(df, target=target)
    x = df[feature_cols]
    y = df[target]

    _log(f"[prepare_data] x shape: {x.shape}, y length: {len(y)}, target: {target}")
    return x, y


def split_data(x, y, test_size=0.2, random_state=42):

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    _log(f"[split_data] Train: {len(x_train)}, Test: {len(x_test)}")
    return x_train, x_test, y_train, y_test   # <= THIS is critical


def get_pipeline_data(csv_file="hotel_bookings.csv", target="is_canceled"):

    df = load_data(csv_file)
    df = clean_data(df)

    x, y = prepare_data(df, target=target)
    x_train, x_test, y_train, y_test = split_data(x, y)

    numeric, categorical = get_feature_types(df, target=target)

    _log("[get_pipeline_data] Done.")
    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "numeric": numeric,
        "categorical": categorical,
    }


# -------------------- ANALYZE + GIVE RESULT -------------------- #

def get_cancellation_rate(csv_file="hotel_bookings.csv"):
    df = clean_data(load_data(csv_file))

    if "is_canceled" not in df.columns:
        return "Column 'is_canceled' not found."

    total = len(df)
    canceled = int(df["is_canceled"].sum())
    rate = (canceled / total * 100) if total > 0 else 0.0

    return f"Cancellation rate: {rate:.1f}% ({canceled}/{total} bookings)"


def analyze_by_segment(csv_file="hotel_bookings.csv"):
    df = clean_data(load_data(csv_file))

    if "market_segment" not in df.columns or "is_canceled" not in df.columns:
        return "Columns 'market_segment' or 'is_canceled' not found."

    grp = df.groupby("market_segment")["is_canceled"].agg(["count", "sum", "mean"])
    grp.columns = ["total", "canceled", "rate"]
    grp["rate"] = (grp["rate"] * 100).round(1)

    lines = ["Cancellation by market_segment:"]
    for seg, row in grp.iterrows():
        lines.append(
            f"  {seg}: {row['rate']:.1f}% ({int(row['canceled'])}/{int(row['total'])})"
        )
    return "\n".join(lines)


def analyze_by_deposit_type(csv_file="hotel_bookings.csv"):
    df = clean_data(load_data(csv_file))

    needed = ["deposit_type", "is_canceled", "adr"]
    if any(col not in df.columns for col in needed):
        return "Columns 'deposit_type', 'is_canceled', or 'adr' not found."

    grp = df.groupby("deposit_type").agg(
        total=("is_canceled", "count"),
        canceled=("is_canceled", "sum"),
        rate=("is_canceled", "mean"),
        avg_adr=("adr", "mean"),
    )
    grp["rate"] = (grp["rate"] * 100).round(1)
    grp["avg_adr"] = grp["avg_adr"].round(2)

    lines = ["Cancellation & ADR by deposit_type:"]
    for dep, row in grp.iterrows():
        lines.append(
            f"  {dep}: {row['rate']:.1f}% canceled ({int(row['canceled'])}/{int(row['total'])}), "
            f"avg ADR = {row['avg_adr']:.2f}"
        )
    return "\n".join(lines)


