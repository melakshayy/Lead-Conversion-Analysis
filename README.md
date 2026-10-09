# Lead Conversion & Lead-Scoring Analysis (SQL + Python)

**Author:** Lakshay | [LinkedIn](https://linkedin.com/in/melakshay) | [GitHub](https://github.com/melakshayy)
**Tools:** Python, SQL (SQLite), pandas, scikit-learn, Chart.js, GitHub Pages
**Live dashboard:** https://melakshayy.github.io/Lead-Conversion-Analysis/dashboard.html

**Question:** In a telemarketing sales campaign, which leads convert, where is calling effort wasted, and can a simple score prioritise the pipeline?

**Data:** UCI Bank Marketing dataset (Moro, Cortez & Rita, 2014; CC BY 4.0), 41,188 calls from a Portuguese bank's term-deposit campaign. 12 duplicate rows removed, leaving 41,176 leads. The sales funnel mirrors a CRM lead pipeline: lead, contact attempts, conversion.

## Key findings
| Finding | Evidence |
|---|---|
| Warm leads convert far better | Prior-campaign success: **65.1%** vs 8.8% for never-contacted leads |
| Channel matters | Cellular **14.7%** vs landline **5.2%** |
| Repeated attempts have steeply diminishing returns | 1-3 attempts: **74.5** conversions per 1,000 calls; 4+ attempts: **10.9** (about 7x worse), yet they take **48%** of all calls |
| A simple score prioritises well | Logistic regression (AUC **0.80**): top decile converts at **49.5%** vs 11.3% baseline; top 3 deciles capture **72.9%** of conversions |

**Business recommendation:** cap repeat attempts at about 3, route warm and cellular leads first, and rank the daily call list by model score.

## Method
1. `analysis.sql` - funnel, segment and effort-vs-return queries (SQLite).
2. `analysis.py` - cleaning, SQL execution, lead-scoring model (scikit-learn), exports.
3. **Leakage control:** call `duration` is excluded from the model because it is only known after the call ends.
4. Evaluation on a stratified 25% hold-out set; decile lift table in `out_deciles.csv`.
5. `leads_clean_for_powerbi.csv` is ready to import into Power BI or Tableau.

## Limitations
- Monthly rates are confounded by campaign targeting (low-volume months such as March, September, October and December show 44-51% conversion), so treat month as seasonality plus targeting, not a causal lever.
- Macro-economic fields (e.g. employment variation rate) are strong drivers, but they describe the period, not the customer.
- Results are from one bank campaign and may not transfer to other industries.

## Run
```
pip install pandas scikit-learn
python analysis.py
```
Data source: https://archive.ics.uci.edu/dataset/222/bank+marketing
