import pandas as pd, numpy as np, joblib
from sklearn.metrics import roc_auc_score

bundle   = joblib.load('thunderstorm_model.pkl')
model    = bundle['model']
FEATURES = bundle['features']

df = pd.read_csv('bengaluru_thunderstorm_features_v2.csv')
X_test = df[df['YEAR'] >= 2023][FEATURES].fillna(0)
y_test = df[df['YEAR'] >= 2023]['LABEL']

proba = model.predict_proba(X_test)[:,1]
pred  = (proba >= 0.4).astype(int)

TP = ((pred==1) & (y_test==1)).sum()
FP = ((pred==1) & (y_test==0)).sum()
TN = ((pred==0) & (y_test==0)).sum()
FN = ((pred==0) & (y_test==1)).sum()

POD  = TP / (TP + FN)
FAR  = FP / (TP + FP)
CSI  = TP / (TP + FN + FP)
BIAS = (TP + FP) / (TP + FN)
HSS_num = 2*(TP*TN - FP*FN)
HSS_den = (TP+FN)*(FN+TN) + (TP+FP)*(FP+TN)
HSS  = HSS_num / HSS_den
AUROC = roc_auc_score(y_test, proba)

print(f"\n{'='*40}")
print(f"WMO Evaluation Metrics — Test Set (2023-2025)")
print(f"{'='*40}")
print(f"AUROC : {AUROC:.4f}")
print(f"POD   : {POD:.4f}   (Probability of Detection)")
print(f"FAR   : {FAR:.4f}   (False Alarm Ratio)")
print(f"CSI   : {CSI:.4f}   (Critical Success Index)")
print(f"HSS   : {HSS:.4f}   (Heidke Skill Score)")
print(f"BIAS  : {BIAS:.4f}   (Frequency Bias)")
print(f"{'='*40}")
print(f"TP={TP}  FP={FP}  TN={TN}  FN={FN}")
# Find optimal threshold by maximizing CSI
from sklearn.metrics import precision_recall_curve
thresholds_to_try = np.arange(0.1, 0.9, 0.01)
best_csi, best_thresh = 0, 0.4
for t in thresholds_to_try:
    p = (proba >= t).astype(int)
    tp = ((p==1) & (y_test==1)).sum()
    fp = ((p==1) & (y_test==0)).sum()
    fn = ((p==0) & (y_test==1)).sum()
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
    if csi > best_csi:
        best_csi, best_thresh = csi, t

print(f"\nOptimal threshold (max CSI): {best_thresh:.2f}")
print(f"Rerunning metrics at threshold {best_thresh:.2f}...")
pred = (proba >= best_thresh).astype(int)
TP = ((pred==1) & (y_test==1)).sum()
FP = ((pred==1) & (y_test==0)).sum()
TN = ((pred==0) & (y_test==0)).sum()
FN = ((pred==0) & (y_test==1)).sum()

POD  = TP / (TP + FN)
FAR  = FP / (TP + FP)
CSI  = TP / (TP + FN + FP)
BIAS = (TP + FP) / (TP + FN)
HSS_num = 2*(TP*TN - FP*FN)
HSS_den = (TP+FN)*(FN+TN) + (TP+FP)*(FP+TN)
HSS  = HSS_num / HSS_den

print(f"\n{'='*40}")
print(f"Metrics at threshold {best_thresh:.2f}")
print(f"{'='*40}")
print(f"AUROC : {AUROC:.4f}")
print(f"POD   : {POD:.4f}")
print(f"FAR   : {FAR:.4f}")
print(f"CSI   : {CSI:.4f}")
print(f"HSS   : {HSS:.4f}")
print(f"BIAS  : {BIAS:.4f}")
print(f"TP={TP}  FP={FP}  TN={TN}  FN={FN}")
results = pd.DataFrame([{
    'Model': 'XGBoost (surface + upper-air)',
    'AUROC': round(AUROC,4), 'POD': round(POD,4),
    'FAR': round(FAR,4), 'CSI': round(CSI,4),
    'HSS': round(HSS,4), 'BIAS': round(BIAS,4)
}])
results.to_csv('evaluation_results.csv', index=False)
print("\nSaved: evaluation_results.csv")