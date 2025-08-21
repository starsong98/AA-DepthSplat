# len_list.py
import json
import sys

if len(sys.argv) < 2:
    print("사용법: python len_list.py <json_file>")
    raise SystemExit(1)

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, list):
    raise TypeError(f"최상위 타입이 list가 아닙니다: {type(data).__name__}")

print(len(data))