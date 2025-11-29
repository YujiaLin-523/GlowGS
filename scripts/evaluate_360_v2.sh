# 360_v2 dataset evaluation script

# bicycle
# render GlowGS
python render.py -s data/360_v2/bicycle -m output/bicycle
# compute error metrics on renderings
python metrics.py -m output/bicycle

# bonsai
# render GlowGS
python render.py -s data/360_v2/bonsai -m output/bonsai
# compute error metrics on renderings
python metrics.py -m output/bonsai

# counter
# render GlowGS
python render.py -s data/360_v2/counter -m output/counter
# compute error metrics on renderings
python metrics.py -m output/counter

# garden
# render GlowGS
python render.py -s data/360_v2/garden -m output/garden
# compute error metrics on renderings
python metrics.py -m output/garden

# kitchen
# render GlowGS
python render.py -s data/360_v2/kitchen -m output/kitchen
# compute error metrics on renderings
python metrics.py -m output/kitchen

# room
# render GlowGS
python render.py -s data/360_v2/room -m output/room
# compute error metrics on renderings
python metrics.py -m output/room

# stump
# render GlowGS
python render.py -s data/360_v2/stump -m output/stump
# compute error metrics on renderings
python metrics.py -m output/stump