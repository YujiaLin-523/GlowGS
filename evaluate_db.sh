# db dataset evaluation script

# drjohnson
# render LocoGS
python render.py -s data/db/drjohnson -m output/drjohnson_hybrid
# compute error metrics on renderings
python metrics.py -m output/drjohnson_hybrid

# playroom
# render LocoGS
python render.py -s data/db/playroom -m output/playroom_hybrid
# compute error metrics on renderings
python metrics.py -m output/playroom_hybrid
