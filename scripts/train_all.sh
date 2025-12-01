bash ./scripts/train_360_v2.sh
bash ./scripts/train_tandt.sh
bash ./scripts/train_db.sh

find ./output -maxdepth 2 -type d -print0 | \
while IFS= read -r -d '' d; do t="$d/compression/iteration_30000"; if [ -d "$t" ]; then s=$(find "$t" -type f -printf '%s\n' | awk '{s+=$1}END{print s+0}'); printf "%s\t%.2f MB\n" "$t" "$(awk -v b="$s" 'BEGIN{printf "%.2f", b/1048576}')"; fi; done