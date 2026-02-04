# dump_tflm_ops_from_carray.py
# Usage:
#   python dump_tflm_ops_from_carray.py path/to/model_data.c
#   python dump_tflm_ops_from_carray.py path/to/model_data.h --symbol g_magic_wand_model_data
#   python dump_tflm_ops_from_carray.py path/to/model_data.c --subgraph 0 --markdown

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---- Best-effort mapping: TFLite BuiltinOperator name -> MicroMutableOpResolver method ----
# (Unknown ops will be shown as "N/A")
RESOLVER_MAP = {
    "ADD": "AddAdd",
    "MUL": "AddMul",
    "CONV_2D": "AddConv2D",
    "DEPTHWISE_CONV_2D": "AddDepthwiseConv2D",
    "FULLY_CONNECTED": "AddFullyConnected",
    "SOFTMAX": "AddSoftmax",
    "MEAN": "AddMean",
    "RESHAPE": "AddReshape",
    "MAX_POOL_2D": "AddMaxPool2D",
    "EXPAND_DIMS": "AddExpandDims",
    "SQUEEZE": "AddSqueeze",
    "RELU": "AddRelu",
    "QUANTIZE": "AddQuantize",
    "DEQUANTIZE": "AddDequantize",
    # add more as you encounter them
}

def enum_value_to_name(enum_cls, value: int) -> str:
    """
    FlatBuffers由来の enum (BuiltinOperatorなど) で、
    value(整数) -> 名前(文字列)を取得する。
    TFのschema生成コードの差異を吸収するための互換関数。
    """
    # TFバージョンによっては Name() が存在する
    if hasattr(enum_cls, "Name"):
        try:
            return enum_cls.Name(value)
        except Exception:
            pass

    # Name() が無い場合：クラス定義の定数を逆引きする
    cache_attr = "_reverse_map_cache"
    if not hasattr(enum_cls, cache_attr):
        rev = {}
        for k, v in enum_cls.__dict__.items():
            if k.isupper() and isinstance(v, int):
                rev[v] = k
        setattr(enum_cls, cache_attr, rev)

    rev = getattr(enum_cls, cache_attr)
    return rev.get(value, f"UNKNOWN_{value}")



@dataclass
class OpCodeInfo:
    idx: int
    builtin_code: int
    builtin_name: str
    version: int
    custom_code: str


def _import_schema():
    """
    TensorFlow installs schema_py_generated here.
    If unavailable, instruct user to install tensorflow.
    """
    try:
        from tensorflow.lite.python import schema_py_generated as schema_fb
        return schema_fb
    except Exception as e:
        raise RuntimeError(
            "Failed to import TensorFlow Lite schema.\n"
            "Please install TensorFlow (pip install tensorflow) in this Python environment.\n"
            f"Original error: {e}"
        )


def extract_c_array_bytes(text: str, symbol: Optional[str] = None) -> bytes:
    """
    Extract the initializer {...} of a C array containing model bytes.
    - If symbol is provided, locate that symbol then extract the next brace block.
    - Otherwise, find the first likely '...[] = { ... }' block.
    """
    # Remove /* */ comments to avoid braces in comments confusing parsing
    text_wo_block_comments = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    start = -1
    if symbol:
        m = re.search(r"\b" + re.escape(symbol) + r"\b", text_wo_block_comments)
        if not m:
            raise ValueError(f"Symbol '{symbol}' not found.")
        # find the first '{' after symbol
        start = text_wo_block_comments.find("{", m.end())
        if start < 0:
            raise ValueError(f"Could not find '{{' after symbol '{symbol}'.")
    else:
        # Heuristic: find something like "const unsigned char xxx[] = {"
        patterns = [
            r"\b(const\s+)?(unsigned\s+char|signed\s+char|uint8_t|int8_t)\b[^;=\n]*\[\s*\]\s*=\s*\{",
            r"\b(const\s+)?(unsigned\s+char|signed\s+char|uint8_t|int8_t)\b[^;=\n]*\[\s*\w+\s*\]\s*=\s*\{",
            r"\[\s*\]\s*=\s*\{",  # fallback
        ]
        for pat in patterns:
            m = re.search(pat, text_wo_block_comments)
            if m:
                start = text_wo_block_comments.find("{", m.end() - 1)
                break
        if start < 0:
            raise ValueError("Could not find a C array initializer '{...}'. Use --symbol to specify the array name.")

    # Brace matching to extract the correct initializer block
    depth = 0
    end = -1
    for i in range(start, len(text_wo_block_comments)):
        ch = text_wo_block_comments[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        raise ValueError("Could not find matching '}' for the initializer block.")

    body = text_wo_block_comments[start + 1 : end]

    # Extract numeric literals (hex or decimal). This ignores commas/whitespace/newlines.
    nums = re.findall(r"0x[0-9A-Fa-f]+|\d+", body)
    if not nums:
        raise ValueError("No numeric literals found inside '{...}' initializer.")

    values: List[int] = []
    for s in nums:
        v = int(s, 16) if s.lower().startswith("0x") else int(s, 10)
        if not (0 <= v <= 255):
            # Model arrays should be bytes; if you hit a large number, you're probably parsing the wrong block
            raise ValueError(
                f"Found a value outside byte range (0..255): {v}. "
                "Likely parsed the wrong initializer block. Use --symbol."
            )
        values.append(v)

    return bytes(values)


def read_model_from_cfile(path: str, symbol: Optional[str]) -> bytes:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return extract_c_array_bytes(text, symbol=symbol)


def opcode_builtin(schema_fb, oc) -> int:
    # For newer schemas, ops beyond certain range use a placeholder and store actual in DeprecatedBuiltinCode.
    # Use that if present.
    try:
        placeholder = schema_fb.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES
        if oc.BuiltinCode() == placeholder:
            return oc.DeprecatedBuiltinCode()
    except Exception:
        pass
    return oc.BuiltinCode()


def parse_ops(buf: bytes, subgraph_idx: int = 0) -> Tuple[List[OpCodeInfo], List[OpCodeInfo]]:
    schema_fb = _import_schema()
    model = schema_fb.Model.GetRootAsModel(buf, 0)

    # OperatorCodes table (definitions)
    opcodes: List[OpCodeInfo] = []
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        builtin = opcode_builtin(schema_fb, oc)
        name = enum_value_to_name(schema_fb.BuiltinOperator, builtin)
        custom = oc.CustomCode().decode("utf-8") if oc.CustomCode() else ""
        ver = oc.Version()
        opcodes.append(OpCodeInfo(i, builtin, name, ver, custom))

    if subgraph_idx < 0 or subgraph_idx >= model.SubgraphsLength():
        raise ValueError(f"subgraph_idx out of range. model has {model.SubgraphsLength()} subgraphs.")

    sg = model.Subgraphs(subgraph_idx)

    used: List[OpCodeInfo] = []
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        oi = op.OpcodeIndex()
        used.append(opcodes[oi])

    return opcodes, used


def format_output(used: List[OpCodeInfo], markdown: bool) -> str:
    lines = []
    cnt = Counter((u.builtin_code, u.builtin_name, u.version, u.custom_code) for u in used)

    if markdown:
        lines.append("## Model used ops (subgraph) 解析結果")
        lines.append("")
        lines.append("| count | builtin | name | version | custom | resolver.Add... |")
        lines.append("|---:|---:|---|---:|---|---|")
        for (builtin, name, ver, custom), c in cnt.most_common():
            custom_s = custom if custom else "-"
            resolver = RESOLVER_MAP.get(name, "N/A")
            lines.append(f"| {c} | {builtin} | {name} | {ver} | {custom_s} | {resolver} |")
        lines.append("")
        lines.append("> `custom` が `-` 以外の場合は Custom OP です（追加登録が必要になる可能性があります）。")
    else:
        lines.append("[INFO] unique ops (count x builtin/name/version/custom):")
        for (builtin, name, ver, custom), c in cnt.most_common():
            custom_s = custom if custom else "-"
            resolver = RESOLVER_MAP.get(name, "N/A")
            lines.append(f"  {c:3d} x builtin={builtin:3d} {name:22s} ver={ver} custom={custom_s} resolver={resolver}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfile", help="Path to model_data.c / model_data.h (C array containing .tflite bytes)")
    ap.add_argument("--symbol", help="C array symbol name (e.g., g_magic_wand_model_data). Strongly recommended if multiple arrays exist.")
    ap.add_argument("--subgraph", type=int, default=0, help="Subgraph index to inspect (default: 0)")
    ap.add_argument("--markdown", action="store_true", help="Output as Markdown table")
    args = ap.parse_args()

    try:
        buf = read_model_from_cfile(args.cfile, args.symbol)
        _, used = parse_ops(buf, subgraph_idx=args.subgraph)
        print(format_output(used, markdown=args.markdown))
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
