# make_model_header.py
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="g_gesture_model")
    args = ap.parse_args()

    with open(args.inp, "rb") as f:
        b = f.read()

    with open(args.out, "w", encoding="utf-8") as w:
        w.write("#pragma once\n")
        w.write("#include <cstdint>\n\n")
        w.write(f"extern const unsigned char {args.name}[];\n")
        w.write(f"extern const unsigned int {args.name}_len;\n")

    cc = args.out.replace(".h", ".cc")
    with open(cc, "w", encoding="utf-8") as w:
        w.write(f'#include "{args.out.split("/")[-1]}"\n')
        w.write("#if defined(ARDUINO)\n")
        w.write("#include <Arduino.h>\n")
        w.write("#endif\n\n")
        w.write(f"alignas(16) const unsigned char {args.name}[] = {{\n")
        for i in range(0, len(b), 12):
            chunk = b[i:i+12]
            w.write("  " + ", ".join(f"0x{x:02X}" for x in chunk) + ",\n")
        w.write("};\n")
        w.write(f"const unsigned int {args.name}_len = {len(b)};\n")

    print("[OK] wrote:", args.out, "and", cc)

if __name__ == "__main__":
    main()
