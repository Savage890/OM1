#!/usr/bin/env python3
import sys
import json5

def main():
    has_error = False
    for filename in sys.argv[1:]:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                json5.load(f)
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            has_error = True
    
    sys.exit(1 if has_error else 0)

if __name__ == "__main__":
    main()
