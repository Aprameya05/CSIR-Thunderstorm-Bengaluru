import pickle, joblib

for slot in range(4):
    path = f'models/nowcast_slot{slot}_xgb_v3_calibrated.pkl'
    artifact = pickle.load(open(path, 'rb'))
    model = artifact['model']
    model.get_booster().save_model(f'models/slot{slot}_booster.ubj')
    joblib.dump(artifact, path)
    print(f'Slot {slot} resaved OK')
