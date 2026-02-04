# Mipnerf 360
# python train.py -s data/360_v2/bicycle -m output/bicycle_hash_only \
#   --encoder_variant hash_only --iterations 30000 --save_iterations 30000 --test_iterations 30000 --quiet

python train.py -s data/360_v2/bonsai -m output/bonsai_hash_only \
  --encoder_variant hash_only --iterations 30000 --save_iterations 30000 --test_iterations 30000 --quiet

# Deep Blending
python train.py -s data/DB/playroom -m output/playroom_hash_only \
  --encoder_variant hash_only --iterations 30000 --save_iterations 30000 --test_iterations 30000 --quiet

# Tanks and Temples
python train.py -s data/TanksAndTemples/truck -m output/truck_hash_only \
  --encoder_variant hash_only --iterations 30000 --save_iterations 30000 --test_iterations 30000 --quiet