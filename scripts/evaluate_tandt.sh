# tandt dataset evaluation script

# train
# render LocoGS
python render.py -s data/tandt/train -m output/train
# compute error metrics on renderings
python metrics.py -m output/train

# truck
# render LocoGS
python render.py -s data/tandt/truck -m output/truck
# compute error metrics on renderings
python metrics.py -m output/truck