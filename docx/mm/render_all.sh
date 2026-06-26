#!/bin/bash
# Batch render all Mermaid diagrams safely.
# Requires: npm install -g @mermaid-js/mermaid-cli

# Get absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/out"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

count=0
failed=0

# Safely check if any .mmd files actually exist before looping
shopt -s nullglob
mmd_files=("$SCRIPT_DIR"/*.mmd)

if [ ${#mmd_files[@]} -eq 0 ]; then
    echo "No .mmd files found in $SCRIPT_DIR"
    exit 0
fi

# Process files one by one
for mmd_file in "${mmd_files[@]}"; do
    # Skip if it points to the output directory by mistake
    if [[ "$mmd_file" == "$OUTPUT_DIR"* ]]; then
        continue
    fi

    base=$(basename "$mmd_file" .mmd)
    echo "Rendering: $base ..."
    
    # Removed '2>/dev/null' so you can see the exact error if it fails
    if mmdc -i "$mmd_file" -o "$OUTPUT_DIR/${base}.png" -s 3; then
        count=$((count + 1))
    else
        echo "  FAILED: $base"
        failed=$((failed + 1))
    fi
done

echo ""
echo "Done. Rendered: $count  Failed: $failed"
echo "Output directory: $OUTPUT_DIR"
