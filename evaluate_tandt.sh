# tandt dataset evaluation script

# train
# render LocoGS
python render.py -s data/tandt/train -m output/train_lrtfr
# compute error metrics on renderings
python metrics.py -m output/train_lrtfr

# truck
# render LocoGS
python render.py -s data/tandt/truck -m output/truck_lrtfr
# compute error metrics on renderings
python metrics.py -m output/truck_lrtfr
