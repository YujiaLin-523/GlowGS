# db dataset evaluation script

# drjohnson
# render LocoGS
python render.py -s data/db/drjohnson -m output/drjohnson
# compute error metrics on renderings
python metrics.py -m output/drjohnson

# playroom
# render LocoGS
python render.py -s data/db/playroom -m output/playroom
# compute error metrics on renderings
python metrics.py -m output/playroom