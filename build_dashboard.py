"""Author: Lakshay (linkedin.com/in/melakshay | github.com/melakshayy)
Builds the interactive dashboard (dashboard.html) with lead-level data + out-of-fold lead scores."""
import json, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from analysis import df, feat_cat, feat_num, pre
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
m = Pipeline([('pre', pre), ('lr', LogisticRegression(max_iter=2000, class_weight='balanced'))])
df['score'] = cross_val_predict(m, df[feat_cat+feat_num], df['converted'], cv=StratifiedKFold(5, shuffle=True, random_state=42), method='predict_proba')[:,1]
orders = {'job': sorted(df.job.unique()), 'contact': ['cellular','telephone'], 'month': ['mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],
          'pout': ['success','failure','nonexistent'], 'age': ['<=25','26-35','36-45','46-55','56-65','65+'], 'att': ['1','2','3','4-5','6+']}
col = {'job':'job','contact':'contact','month':'month','pout':'poutcome','age':'age_band','att':'contacts_band'}
idx = {k:{v:i for i,v in enumerate(o)} for k,o in orders.items()}
rows = [[idx[k][r[col[k]]] for k in orders] + [int(r['campaign']), int(r['converted']), int(round(r['score']*1000))] for r in df.to_dict('records')]
data = json.dumps({'labels': orders, 'rows': rows}, separators=(',',':'))
html = open('dashboard_template.html').read().replace('__DATA__', data)
open('dashboard.html','w').write(html); print(len(rows), len(html)//1024, 'KB')
