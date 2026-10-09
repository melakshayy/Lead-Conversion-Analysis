"""Lead Conversion & Lead-Scoring Analysis - UCI Bank Marketing (telemarketing campaign).
SQL (SQLite) for the funnel analysis + Python (pandas, scikit-learn) for lead scoring."""
import sqlite3, json, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 1. LOAD + CLEAN
df = pd.read_csv('bank.csv', sep=';')
df = df.drop_duplicates().reset_index(drop=True)
df['converted'] = (df['y'] == 'yes').astype(int)
df['previously_contacted'] = (df['pdays'] != 999).astype(int)
df['duration_band'] = pd.cut(df['duration'], [-1,60,180,300,600,10**6], labels=['<1 min','1-3 min','3-5 min','5-10 min','10+ min']).astype(str)
df['contacts_band'] = pd.cut(df['campaign'], [0,1,2,3,5,10**3], labels=['1','2','3','4-5','6+']).astype(str)
df['age_band'] = pd.cut(df['age'], [0,25,35,45,55,65,200], labels=['<=25','26-35','36-45','46-55','56-65','65+']).astype(str)
df.columns = [c.replace('.','_') for c in df.columns]
con = sqlite3.connect(':memory:'); df.to_sql('leads', con, index=False)
q = lambda s: pd.read_sql(s, con)

# 2. SQL FUNNEL ANALYSIS (same queries are in analysis.sql)
overall = q("SELECT COUNT(*) AS leads, SUM(converted) AS conversions, ROUND(100.0*AVG(converted),2) AS conv_rate_pct FROM leads").iloc[0].to_dict()
def by(col, order=None, min_n=0):
    r = q(f"SELECT {col} AS segment, COUNT(*) AS leads, SUM(converted) AS conversions, ROUND(100.0*AVG(converted),2) AS conv_rate_pct FROM leads GROUP BY {col} HAVING COUNT(*)>={min_n} ORDER BY {order or 'conv_rate_pct DESC'}")
    return r
res = {
 'overall': overall,
 'by_poutcome': by('poutcome'),
 'by_contact': by('contact'),
 'by_month': by('month', order="CASE month WHEN 'mar' THEN 3 WHEN 'apr' THEN 4 WHEN 'may' THEN 5 WHEN 'jun' THEN 6 WHEN 'jul' THEN 7 WHEN 'aug' THEN 8 WHEN 'sep' THEN 9 WHEN 'oct' THEN 10 WHEN 'nov' THEN 11 ELSE 12 END"),
 'by_job': by('job', min_n=100),
 'by_contacts_band': by('contacts_band', order="CASE contacts_band WHEN '1' THEN 1 WHEN '2' THEN 2 WHEN '3' THEN 3 WHEN '4-5' THEN 4 ELSE 5 END"),
 'by_age_band': by('age_band', order="CASE age_band WHEN '<=25' THEN 1 WHEN '26-35' THEN 2 WHEN '36-45' THEN 3 WHEN '46-55' THEN 4 WHEN '56-65' THEN 5 ELSE 6 END"),
 'by_duration_band': by('duration_band', order="CASE duration_band WHEN '<1 min' THEN 1 WHEN '1-3 min' THEN 2 WHEN '3-5 min' THEN 3 WHEN '5-10 min' THEN 4 ELSE 5 END"),
}
# effort vs. return: how much of the calling effort is spent on 4th+ attempts and what it yields
effort = q("""SELECT CASE WHEN campaign>=4 THEN '4+ attempts' ELSE '1-3 attempts' END AS attempts_group,
              SUM(campaign) AS total_calls, SUM(converted) AS conversions,
              ROUND(1000.0*SUM(converted)/SUM(campaign),1) AS conversions_per_1000_calls
              FROM leads GROUP BY 1""")
res['effort'] = effort

# 3. LEAD SCORING MODEL (duration excluded on purpose: known only AFTER the call = data leakage)
feat_cat = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
feat_num = ['age','campaign','previous','previously_contacted','emp_var_rate','cons_price_idx','cons_conf_idx','euribor3m','nr_employed']
X, y = df[feat_cat+feat_num], df['converted']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
pre = ColumnTransformer([('c', OneHotEncoder(handle_unknown='ignore'), feat_cat), ('n', StandardScaler(), feat_num)])
model = Pipeline([('pre', pre), ('lr', LogisticRegression(max_iter=2000, class_weight='balanced'))]).fit(Xtr, ytr)
p = model.predict_proba(Xte)[:,1]
auc = roc_auc_score(yte, p)
t = pd.DataFrame({'p':p,'y':yte.values}).sort_values('p', ascending=False).reset_index(drop=True)
t['decile'] = (t.index * 10 // len(t)) + 1
dec = t.groupby('decile').agg(leads=('y','size'), conversions=('y','sum')).reset_index()
dec['conv_rate_pct'] = (100*dec.conversions/dec.leads).round(2)
dec['cum_capture_pct'] = (100*dec.conversions.cumsum()/dec.conversions.sum()).round(1)
res['model'] = {'auc': round(auc,3), 'top3_deciles_capture_pct': float(dec.loc[2,'cum_capture_pct']),
                'top_decile_conv_pct': float(dec.loc[0,'conv_rate_pct']), 'baseline_pct': round(100*yte.mean(),2)}
res['deciles'] = dec
# top drivers
names = model.named_steps['pre'].get_feature_names_out()
coef = pd.Series(model.named_steps['lr'].coef_[0], index=names).sort_values()
res['drivers_pos'] = coef.tail(6).round(2).to_dict(); res['drivers_neg'] = coef.head(6).round(2).to_dict()

# 4. EXPORTS
out = {k:(v.to_dict('records') if isinstance(v, pd.DataFrame) else v) for k,v in res.items()}
json.dump(out, open('results.json','w'), indent=1, default=float)
df.drop(columns=['y']).to_csv('leads_clean_for_powerbi.csv', index=False)
for k in ['by_poutcome','by_contact','by_month','by_job','by_contacts_band','by_age_band','by_duration_band','effort','deciles']:
    res[k].to_csv(f'out_{k}.csv', index=False)
print(json.dumps({k:out[k] for k in ['overall','model','effort','by_poutcome','by_contact','by_contacts_band','drivers_pos','drivers_neg']}, indent=1, default=float))
print(dec); print(res['by_job'].head(4)); print(res['by_job'].tail(3)); print(res['by_month'])
