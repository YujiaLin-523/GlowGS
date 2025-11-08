# 360_v2 dataset evaluation script

# bicycle
# render LocoGS
python render.py -s data/360_v2/bicycle -m output/bicycle_lrtfr
# compute error metrics on renderings
python metrics.py -m output/bicycle_lrtfr

# bonsai
# render LocoGS
python render.py -s data/360_v2/bonsai -m output/bonsai_lrtfr
# compute error metrics on renderings
python metrics.py -m output/bonsai_lrtfr

# counter
# render LocoGS
python render.py -s data/360_v2/counter -m output/counter_lrtfr
# compute error metrics on renderings
python metrics.py -m output/counter_lrtfr

# garden
# render LocoGS
python render.py -s data/360_v2/garden -m output/garden_lrtfr
# compute error metrics on renderings
python metrics.py -m output/garden_lrtfr

# kitchen
# render LocoGS
python render.py -s data/360_v2/kitchen -m output/kitchen_lrtfr
# compute error metrics on renderings
python metrics.py -m output/kitchen_lrtfr

# room
# render LocoGS
python render.py -s data/360_v2/room -m output/room_lrtfr
# compute error metrics on renderings
python metrics.py -m output/room_lrtfr

# stump
# render LocoGS
python render.py -s data/360_v2/stump -m output/stump_lrtfr
# compute error metrics on renderings
python metrics.py -m output/stump_lrtfr
