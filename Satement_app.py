# Bank Statement Analyzer – Streamlit app
# -------------------------------------------------
# Features:
# - Robust CSV/XLSX ingestion (auto-detect date/amount/description/account)
# - Handles separate Debit/Credit columns or signed Amount
# - Year/day/week/month filters (fixed)
# - Dashboard, Overview, Categories, Merchants, Cashflow & Budgets,
#   Recurring, Anomalies & Duplicates, Compare Years, Export
# - Advanced Filters: amount range, inflow/outflow, repeated amounts/merchants,
#   exclude transfers, ignore small txns, anomalies-only
# - Smart Advisor: Pareto savings, hot categories (MoM vs 3M baseline),
#   subscription creep, fee/duplicate flags, savings plan suggestions
# -------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from io import StringIO
import io, csv, re, warnings, os
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Bank Statement Analyzer", page_icon="💳", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
DATE_HINTS = ['date', 'posting', 'post date', 'transaction date', 'time', 'дата']
DESC_HINTS = ['description', 'details', 'memo', 'narrative', 'payee', 'merchant', 'name', 'описание']
AMT_HINTS  = ['amount', 'amt', 'сумма', 'value']
DEBIT_HINTS= ['debit', 'withdrawal', 'spent', 'charge']
CREDIT_HINTS=['credit', 'deposit', 'income', 'received', 'payment']
CAT_HINTS  = ['category', 'категория']
ACC_HINTS  = ['account', 'iban', 'card', 'mask', 'acct']
CURR_HINTS = ['currency', 'валюта']

FEE_KEYWORDS = ['fee', 'charge', 'commission', 'overdraft', 'atm fee', 'monthly fee', 'maintenance']
TRANSFER_KEYWORDS = ['transfer', 'zelle', 'venmo', 'paypal', 'self transfer', 'to savings', 'from savings']

def find_col(cols, hints):
    cl = [c for c in cols if any(h in str(c).lower() for h in hints)]
    return cl[0] if cl else None

def to_datetime_safe(s):
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

def normalize_text(s):
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9\s\-\&\.\,]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def money_round(x):
    try: return float(np.round(x, 2))
    except: return np.nan

# ---------- utilities for duplicate headers ----------
def _make_unique(names):
    seen = {}
    out = []
    for n in names:
        n0 = str(n)
        if n0 not in seen:
            seen[n0] = 0
            out.append(n0)
        else:
            seen[n0] += 1
            out.append(f"{n0}.{seen[n0]}")
    return out

def first_series(df: pd.DataFrame, name: str):
    base = name.strip().lower()
    matches = [c for c in df.columns if str(c).strip().lower() == base
               or str(c).strip().lower().startswith(base + ".")]
    if not matches:
        return None
    if len(matches) == 1:
        s = df[matches[0]]
        return s if isinstance(s, pd.Series) else s.iloc[:, 0]
    tmp = df[matches].copy()
    if base in ("amount","balance"):
        def _num_clean_col(col):
            return pd.to_numeric(
                col.astype(str)
                   .str.replace("\xa0","", regex=False)
                   .str.replace(" ", "", regex=False)
                   .str.replace(",", "", regex=False)
                   .str.replace(r"[^\d\.\-\(\)]", "", regex=True)
                   .map(lambda x: f"-{x.strip('()')}" if x.startswith("(") and x.endswith(")") else x),
                errors="coerce"
            )
        for c in matches:
            tmp[c] = _num_clean_col(tmp[c])
    return tmp.bfill(axis=1).iloc[:, 0]

# ---------- Robust CSV loader ----------
def _clean_number(s: str):
    s = str(s).strip()
    if s == "" or s.lower() in ("nan","none","null","—","-"): return np.nan
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace("\xa0","").replace(" ", "")
    if re.fullmatch(r"-?\d{1,3}(\.\d{3})*,\d{2}", s):
        s = s.replace(".","").replace(",",".")
    else:
        s = s.replace(",", "")
    s = re.sub(r"[^0-9\.\-]", "", s)
    try:
        v = float(s)
        return -v if neg else v
    except:
        return np.nan

def load_bank_csv(path_or_buf):
    if hasattr(path_or_buf, "read"):
        text = path_or_buf.read()
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        stream = io.StringIO(text)
    else:
        stream = open(path_or_buf, "r", encoding="utf-8", errors="replace", newline="")
    with stream:
        reader = csv.reader(stream, delimiter=",", quotechar='"', strict=False)
        rows = [r for r in reader if r and any(cell.strip() for cell in r)]
    hdr_idx = 0
    for i, r in enumerate(rows):
        if len(r) >= 2 and str(r[0]).strip().lower() == "date":
            hdr_idx = i; break
    header = rows[hdr_idx]
    cols = []
    for i, h in enumerate(header):
        h = (h or "").strip().lower()
        if h == "date": cols.append("Date")
        elif h in ("description","details","memo","narrative","payee","merchant","name"): cols.append("Description")
        elif h in ("amount","debit/credit","debit","credit","amt"): cols.append("Amount")
        elif h in ("balance","running balance"): cols.append("Balance")
        else: cols.append(f"col{i}")
    cols = _make_unique(cols)
    body = rows[hdr_idx+1:]
    body = [r[:len(cols)] + [""]*(len(cols)-len(r)) if len(r)<len(cols) else r[:len(cols)] for r in body]
    df = pd.DataFrame(body, columns=cols)
    if first_series(df, "Date") is not None:
        def _to_date(x):
            for fmt in ("%m/%d/%Y","%m/%d/%y","%Y-%m-%d","%d.%m.%Y","%d/%m/%Y"):
                try: return pd.to_datetime(x, format=fmt)
                except: pass
            return pd.to_datetime(x, errors="coerce")
        df["Date"] = first_series(df, "Date").map(_to_date)
    amt_ser = first_series(df, "Amount")
    if amt_ser is not None:
        df["Amount"] = amt_ser.map(_clean_number)
    bal_ser = first_series(df, "Balance")
    if bal_ser is not None:
        df["Balance"] = bal_ser.map(_clean_number)
    return df

def is_excel_file(uploaded_file) -> bool:
    name = getattr(uploaded_file, "name", "") or ""
    return name.lower().endswith((".xlsx", ".xls"))

def load_excel_first_sheet(file_obj):
    try:
        return pd.read_excel(file_obj, sheet_name=0, engine=None)
    except ImportError:
        st.error("Missing Excel engine. Please install `openpyxl` for .xlsx files.")
        raise

@st.cache_data
def load_any(file, use_robust_csv: bool = True):
    if is_excel_file(file):
        return load_excel_first_sheet(file)
    if use_robust_csv:
        return load_bank_csv(file)
    try:
        return pd.read_csv(file, encoding="utf-8")
    except Exception:
        return load_bank_csv(file)

# ---------- schema + enrich ----------
def coerce_schema(df):
    df = df.copy()
    if set(df.columns.str.lower()).issuperset({"date","description","amount"}):
        out = pd.DataFrame(index=df.index)
        out["date"] = to_datetime_safe(first_series(df, "Date"))
        out["description"] = first_series(df, "Description").astype(str)
        out["account"] = df.get("Account", "Account-1")
        out["currency"] = df.get("Currency", "")
        out["category_raw"] = df.get("Category", "")
        out["amount"] = pd.to_numeric(first_series(df, "Amount"), errors="coerce")
    else:
        date_col = find_col(df.columns, DATE_HINTS)
        desc_col = find_col(df.columns, DESC_HINTS)
        amt_col  = find_col(df.columns, AMT_HINTS)
        debit_col= find_col(df.columns, DEBIT_HINTS)
        credit_col=find_col(df.columns, CREDIT_HINTS)
        cat_col  = find_col(df.columns, CAT_HINTS)
        acc_col  = find_col(df.columns, ACC_HINTS)
        curr_col = find_col(df.columns, CURR_HINTS)
        out = pd.DataFrame(index=df.index)
        if date_col is None:
            candidate = None
            for c in df.columns:
                dt = to_datetime_safe(df[c])
                if dt.notna().sum() >= len(df)*0.4:
                    candidate = c; break
            date_col = candidate
        out['date'] = to_datetime_safe(df[date_col]) if date_col else pd.NaT
        if desc_col is None: desc_col = df.columns[0]
        out['description'] = df[desc_col].astype(str)
        out['account'] = df[acc_col] if acc_col else "Account-1"
        out['currency'] = df[curr_col] if curr_col else ""
        def parse_amt_series(s):
            return pd.to_numeric(
                s.astype(str)
                 .str.replace("\xa0","", regex=False)
                 .str.replace(" ", "", regex=False)
                 .str.replace(",", "", regex=False)
                 .str.replace(r"[^\d\.\-\(\)]", "", regex=True)
                 .map(lambda x: f"-{x.strip('()')}" if x.startswith("(") and x.endswith(")") else x),
                errors="coerce"
            )
        if amt_col is not None:
            out['amount'] = parse_amt_series(first_series(df, amt_col))
        else:
            deb = parse_amt_series(first_series(df, "debit")) if first_series(df, "debit") is not None else 0.0
            cre = parse_amt_series(first_series(df, "credit")) if first_series(df, "credit") is not None else 0.0
            out['amount'] = (pd.Series(cre).fillna(0) - pd.Series(deb).fillna(0)).astype(float)
        out['category_raw'] = df[cat_col] if cat_col else ""
    out['amount'] = out['amount'].apply(money_round)
    out = out.dropna(subset=['date','amount'])
    out['y'] = out['date'].dt.year.astype('Int64')
    out['m'] = out['date'].dt.to_period('M').astype(str)
    iso = out['date'].dt.isocalendar()
    out['d'] = out['date'].dt.date.astype(str)
    out['w_num'] = iso.week.astype(int)
    out['w_year'] = iso.year.astype(int)
    out['w'] = out['w_year'].astype(str) + "-W" + out['w_num'].astype(str).str.zfill(2)
    out['sign'] = np.where(out['amount'] >= 0, 'inflow','outflow')
    out['desc_norm'] = out['description'].map(normalize_text)
    return out

def apply_rules(df, rules_text):
    df = df.copy()
    df['category'] = df['category_raw'].astype(str)
    rules = {}
    for line in rules_text.splitlines():
        if ':' in line:
            name, kws = line.split(':',1)
            name = name.strip()
            kw_list = [k.strip().lower() for k in kws.split(',') if k.strip()]
            if name and kw_list: rules[name] = kw_list
    for cat, kwds in rules.items():
        mask = False
        for kw in kwds:
            mask = mask | df['desc_norm'].str.contains(re.escape(kw))
        df.loc[mask, 'category'] = cat
    df['category'] = df['category'].replace('', 'Uncategorized')
    return df

def recurring_candidates(df, min_occ=3, max_cv=0.25):
    X = df[df['sign']=='outflow'].copy()
    X['abs_amt'] = X['amount'].abs()
    grp = X.groupby(['desc_norm'])
    stat = grp.agg(
        txns=('amount','count'),
        months=('m', 'nunique'),
        mean_amt=('abs_amt','mean'),
        std_amt=('abs_amt','std'),
        first=('date','min'),
        last=('date','max'),
        merchant=('description','first')
    ).fillna({'std_amt':0})
    stat['cv'] = stat['std_amt'] / stat['mean_amt'].replace(0,np.nan)
    rec = stat[(stat['months']>=min_occ) & (stat['cv']<=max_cv) & (stat['txns']>=min_occ)]
    rec = rec.sort_values(['months','txns','mean_amt'], ascending=[False,False,False])
    return rec.reset_index()

def detect_duplicates(df, days_window=2, amount_tol=0.01):
    X = df.copy()
    X['abs_amount'] = X['amount'].abs().round(2)
    X = X.sort_values('date')
    dups = []
    for amt, bucket in X.groupby('abs_amount'):
        b = bucket.sort_values('date')
        for i in range(1, len(b)):
            prev = b.iloc[i-1]; cur = b.iloc[i]
            if abs((cur['date'] - prev['date']).days) <= days_window:
                if cur['desc_norm'][:24] == prev['desc_norm'][:24]:
                    if abs(abs(cur['amount']) - abs(prev['amount'])) <= amount_tol:
                        dups.append((prev.name, cur.name))
    mark = pd.Series(False, index=df.index)
    for a,b in dups:
        mark.loc[[a,b]] = True
    return df.loc[mark].copy().sort_values(['date','amount'])

def isolation_anomalies(df, contamination=0.02):
    res = []
    for sgn in ['outflow','inflow']:
        X = df[df['sign']==sgn].copy()
        if len(X) < 20: continue
        model = IsolationForest(random_state=42, contamination=contamination)
        X['anom'] = model.fit_predict(X[['amount']])
        X['score'] = model.decision_function(X[['amount']])
        res.append(X[X['anom'] == -1])
    if not res:
        return pd.DataFrame(columns=list(df.columns)+['score'])
    Z = pd.concat(res).sort_values('date')
    return Z

def download_csv(df, name="export.csv"):
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", data=csv_bytes, file_name=name, mime="text/csv")

# -----------------------------
# Sidebar – data + options
# -----------------------------
st.title("💳 Bank Statement Analyzer")
st.markdown("---")

st.sidebar.header("📂 Data")
use_robust = st.sidebar.checkbox("Use robust CSV parser (recommended)", value=True)
uploaded = st.sidebar.file_uploader(
    "Upload one or more CSV/XLSX files",
    type=["csv","xlsx","xls"],
    accept_multiple_files=True
)
sample_btn = st.sidebar.button("Load small demo sample (CSV)")
if sample_btn and not uploaded:
    demo = StringIO("""Date,Description,Debit,Credit,Account
2024-11-02,NETFLIX.COM,-15.99,,Chase
2024-11-03,Payroll,,2500.00,Chase
2024-11-05,STARBUCKS - 123,-6.75,,Chase
2024-11-12,RENT 123 MAIN ST,-1800.00,,Chase
2024-12-02,NETFLIX.COM,-15.99,,Chase
2024-12-03,Payroll,,2500.00,Chase
2024-12-05,STARBUCKS - 123,-7.15,,Chase
2024-12-12,RENT 123 MAIN ST,-1800.00,,Chase
""")
    uploaded = [demo]

raw_list = []
if uploaded:
    for f in uploaded:
        try:
            df0 = load_any(f, use_robust_csv=use_robust)
            raw_list.append(df0)
        except Exception as e:
            st.error(f"Failed to read {getattr(f,'name','file')}: {e}")

if not raw_list:
    st.info("Upload CSV/XLSX to begin. Supported: Date, Description, Amount or Debit/Credit, Account, Category.")
    st.stop()

df_raw = pd.concat(raw_list, ignore_index=True, sort=False)
df = coerce_schema(df_raw)

# -----------------------------
# Rules + Income mapping
# -----------------------------
st.sidebar.header("🏷️ Category Rules (optional)")
rules_default = """groceries: walmart, aldi, trader joe, kroger
coffee: starbucks, dunkin
subscriptions: netflix, spotify, youtube, apple media, icloud
salary: payroll, stripe, upwork
rent: rent, landlord
utilities: comed, at&t, verizon, tmobile, spectrum
transport: uber, lyft, shell, exxon, chevron, bp
fees: fee, overdraft, maintenance, commission
"""
rules_text = st.sidebar.text_area("Edit rules (cat: kw1, kw2, ...)", value=rules_default, height=170)
df = apply_rules(df, rules_text)

force_income = st.sidebar.checkbox("Treat every inflow (amount > 0) as 'income'", value=True)
if force_income:
    df.loc[df['amount'] > 0, 'category'] = 'income'

# -----------------------------
# Advanced Filters (ALL FIXED)
# -----------------------------
st.sidebar.header("🔎 Filters")

years_avail = sorted(df['y'].dropna().astype(int).unique().tolist())
year_single = st.sidebar.selectbox("Single-year view", ["All"] + years_avail)

accounts = ["All"] + sorted(df['account'].astype(str).unique().tolist())
acc_pick = st.sidebar.selectbox("Account", accounts)

cats = ["All"] + sorted(df['category'].astype(str).unique().tolist())
cat_pick = st.sidebar.selectbox("Category", cats)

st.sidebar.subheader("🗓️ Date Filters")
months_avail = sorted(df['m'].unique().tolist())
weeks_avail  = sorted(df['w'].unique().tolist())
days_avail   = sorted(pd.to_datetime(df['d']).dt.date.astype(str).unique().tolist())

months_selected = st.sidebar.multiselect("Month(s) (YYYY-MM)", options=months_avail, default=[])
weeks_selected  = st.sidebar.multiselect("Week(s) (ISO, e.g., 2025-W45)", options=weeks_avail, default=[])
days_selected   = st.sidebar.multiselect("Day(s) (YYYY-MM-DD)", options=days_avail, default=[])

date_range = st.sidebar.date_input("Date range (optional)", value=None)

st.sidebar.subheader("💵 Amount & Type")
min_amt = float(np.nanmin(df['amount'])) if len(df) else -1000.0
max_amt = float(np.nanmax(df['amount'])) if len(df) else 1000.0
amt_lo, amt_hi = st.sidebar.slider("Amount range", min_value=float(min_amt), max_value=float(max_amt),
                                   value=(float(min_amt), float(max_amt)), step=1.0)
only_high = st.sidebar.checkbox("High amount only (>|$1,000|)", value=False)
io_type = st.sidebar.selectbox("Flow type", ["All", "Expenses (outflow)", "Income (inflow)"])

st.sidebar.subheader("🔁 Repeats & Noise")
min_occ_amt = st.sidebar.number_input("Repeated amount threshold (N occurrences)", 2, 50, 3)
filter_repeated_amounts = st.sidebar.checkbox("Show only repeated amounts", value=False)

min_occ_merch = st.sidebar.number_input("Repeated merchant threshold (N txns)", 2, 100, 3)
filter_repeated_merch = st.sidebar.checkbox("Show only repeated merchants", value=False)

ignore_small = st.sidebar.checkbox("Ignore small transactions (|amount| < $5)", value=False)
exclude_transfers = st.sidebar.text_input("Exclude keywords (comma-separated)", value=", ".join(TRANSFER_KEYWORDS))

st.sidebar.subheader("🚨 Anomalies")
anom_only = st.sidebar.checkbox("Show anomalies only (IsolationForest)", value=False)
anom_contam = st.sidebar.slider("Anomaly sensitivity", 0.005, 0.15, 0.02, step=0.005)

# ---- Apply filters
df_f = df.copy()
if acc_pick != "All": df_f = df_f[df_f['account']==acc_pick]
if cat_pick != "All": df_f = df_f[df_f['category']==cat_pick]
if year_single != "All": df_f = df_f[df_f['y']==int(year_single)]

if months_selected: df_f = df_f[df_f['m'].isin(set(months_selected))]
if weeks_selected:  df_f = df_f[df_f['w'].isin(set(weeks_selected))]
if days_selected:   df_f = df_f[df_f['d'].isin(set(days_selected))]

if isinstance(date_range, (list, tuple)) and len(date_range) == 2 and all(pd.notna(date_range)):
    start, end = date_range
    start = pd.to_datetime(start).normalize()
    end   = pd.to_datetime(end).normalize()
    df_f = df_f[(df_f['date'] >= start) & (df_f['date'] <= end)]

# amount & type
if only_high:
    df_f = df_f[df_f['amount'].abs() > 1000]
df_f = df_f[(df_f['amount'] >= amt_lo) & (df_f['amount'] <= amt_hi)]
if io_type == "Expenses (outflow)": df_f = df_f[df_f['sign']=='outflow']
elif io_type == "Income (inflow)":   df_f = df_f[df_f['sign']=='inflow']
if ignore_small: df_f = df_f[df_f['amount'].abs() >= 5]

# exclude keywords
if exclude_transfers.strip():
    kws = [k.strip().lower() for k in exclude_transfers.split(",") if k.strip()]
    if kws:
        mask = np.zeros(len(df_f), dtype=bool)
        for kw in kws:
            mask |= df_f['desc_norm'].str.contains(re.escape(kw))
        df_f = df_f[~mask]

# repeats
if filter_repeated_amounts:
    counts_amt = df_f['amount'].abs().round(2).value_counts()
    keep_amts = counts_amt[counts_amt >= int(min_occ_amt)].index
    df_f = df_f[df_f['amount'].abs().round(2).isin(keep_amts)]

if filter_repeated_merch:
    counts_m = df_f['desc_norm'].value_counts()
    keep_m = counts_m[counts_m >= int(min_occ_merch)].index
    df_f = df_f[df_f['desc_norm'].isin(keep_m)]

# anomalies
if anom_only:
    an = isolation_anomalies(df_f, contamination=anom_contam)
    df_f = df_f.merge(an[['date','description','amount','score']], on=['date','description','amount'], how='inner')

# -----------------------------
# Budget (optional)
# -----------------------------
st.sidebar.header("📊 Monthly Budgets (optional)")
budget_input = st.sidebar.text_area(
    "Format: category = amount (per month)",
    value="groceries = 500\nsubscriptions = 60\ncoffee = 80\nrent = 1800\nutilities = 250\ntransport = 200\nincome = 0",
    height=140
)
budgets = {}
for line in budget_input.splitlines():
    if '=' in line:
        k,v = line.split('=',1)
        k = k.strip().lower()
        try: budgets[k] = float(v.strip())
        except: pass

# -----------------------------
# Routing
# -----------------------------
st.sidebar.header("📑 Sections")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Overview", "Categories", "Merchants", "Cashflow & Budgets",
     "Recurring & Subscriptions", "Anomalies & Duplicates", "Compare Years", "Smart Advisor", "Export"]
)

# -----------------------------
# Dashboard
# -----------------------------
if page == "Dashboard":
    st.subheader("📊 Dashboard")
    total_in = df_f.loc[df_f['amount']>0, 'amount'].sum()
    total_out = df_f.loc[df_f['amount']<0, 'amount'].sum()
    net = total_in + total_out

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Total Inflow", f"${total_in:,.2f}")
    with c2: st.metric("Total Outflow", f"${total_out:,.2f}")
    with c3: st.metric("Net", f"${net:,.2f}")
    with c4: st.metric("Transactions", f"{len(df_f):,}")

    monthly = df_f.groupby('m')['amount'].sum().reset_index()
    monthly['Cashflow'] = monthly['amount']
    st.plotly_chart(px.bar(monthly, x='m', y='Cashflow', title="Monthly Net Cashflow", labels={'m':'Month'}), use_container_width=True)

    spend_by_cat = (df_f[df_f['amount']<0].groupby('category')['amount'].sum().abs().sort_values(ascending=False))
    st.subheader("Top Spend by Category")
    st.plotly_chart(px.bar(top_n_series(spend_by_cat).sort_values(), orientation='h',
                           title="Top Categories by Spend", labels={'value':'Amount','index':'Category'}),
                    use_container_width=True)

# -----------------------------
elif page == "Overview":
    st.subheader("🧾 Transactions")
    st.dataframe(df_f.sort_values('date', ascending=False), use_container_width=True, height=420)
    dly = df_f.groupby('date')['amount'].sum().reset_index()
    st.plotly_chart(px.line(dly, x='date', y='amount', title="Daily Net Flow"), use_container_width=True)
    mon = df_f.groupby('m')['amount'].sum().reset_index()
    st.plotly_chart(px.bar(mon, x='m', y='amount', title="Monthly Net Flow", labels={'m':'Month'}), use_container_width=True)

# -----------------------------
elif page == "Categories":
    st.subheader("🏷️ Category Analysis")
    cat_agg = df_f.groupby(['category','sign'])['amount'].sum().unstack(fill_value=0)
    cat_agg['Net'] = cat_agg.sum(axis=1)
    st.dataframe(cat_agg.sort_values('Net'), use_container_width=True, height=420)

    spend = (df_f[df_f['amount']<0].groupby('category')['amount'].sum().abs().sort_values(ascending=False))
    st.plotly_chart(px.bar(top_n_series(spend).sort_values(), orientation='h',
                 title="Top Spending Categories", labels={'value':'Amount','index':'Category'}), use_container_width=True)

    cm = (df_f[df_f['amount']<0].groupby(['m','category'])['amount'].sum().abs().reset_index())
    if len(cm)>0:
        st.plotly_chart(px.line(cm, x='m', y='amount', color='category',
                                title="Monthly Spend by Category", labels={'m':'Month','amount':'Amount'}),
                        use_container_width=True)

# -----------------------------
elif page == "Merchants":
    st.subheader("🏪 Merchants / Payees")
    merch = (df_f.groupby('desc_norm')['amount'].sum().sort_values())
    tbl = pd.DataFrame({
        'merchant': merch.index,
        'total': merch.values,
        'transactions': df_f.groupby('desc_norm')['amount'].size().reindex(merch.index).values
    })
    st.dataframe(tbl.sort_values('total'), use_container_width=True, height=420)

    top_spend_merch = (df_f[df_f['amount']<0].groupby('desc_norm')['amount'].sum().abs().sort_values(ascending=False))
    st.plotly_chart(px.bar(top_n_series(top_spend_merch).sort_values(), orientation='h',
                           title="Top Spending Merchants", labels={'value':'Amount','index':'Merchant'}),
                    use_container_width=True)

# -----------------------------
elif page == "Cashflow & Budgets":
    st.subheader("💵 Cashflow")
    flow = df_f.groupby('m')['amount'].sum().reset_index()
    st.plotly_chart(px.bar(flow, x='m', y='amount', title="Monthly Net Cashflow", labels={'m':'Month'}), use_container_width=True)

    st.subheader("📅 Monthly Spend vs Budget")
    spent = (df_f[df_f['amount']<0].assign(amount=lambda d: d['amount'].abs())
             .groupby(['m','category'])['amount'].sum().reset_index())
    if budgets:
        spent['budget'] = spent['category'].str.lower().map(budgets).fillna(0.0)
        spent['delta'] = spent['amount'] - spent['budget']
        st.dataframe(spent.sort_values(['m','category']), use_container_width=True, height=420)
        st.plotly_chart(px.bar(spent, x='m', y='delta', color='category',
                               title="Over/Under Budget by Month",
                               labels={'m':'Month','delta':'Over (+) / Under (-)'}),
                        use_container_width=True)
    else:
        st.info("Add budgets in the sidebar to see over/under charts.")

# -----------------------------
elif page == "Recurring & Subscriptions":
    st.subheader("🔁 Recurring Charges / Subscriptions")
    rec = recurring_candidates(df_f)
    if len(rec)==0:
        st.info("No recurring patterns found. Add more data or relax thresholds.")
    else:
        rec_display = rec[['merchant','months','txns','mean_amt','std_amt','first','last']]
        rec_display['mean_amt'] = rec_display['mean_amt'].apply(lambda x: f"${x:,.2f}")
        rec_display['std_amt'] = rec_display['std_amt'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(rec_display, use_container_width=True, height=420)
        topm = rec.iloc[0]['desc_norm']
        trend = (df_f[df_f['desc_norm']==topm].groupby('m')['amount'].sum().abs().reset_index())
        st.plotly_chart(px.line(trend, x='m', y='amount', title=f"Monthly spend: {rec.iloc[0]['merchant']}"),
                        use_container_width=True)

# -----------------------------
elif page == "Anomalies & Duplicates":
    st.subheader("⚠️ Potential Duplicates (near-same desc + date proximity)")
    dups = detect_duplicates(df_f)
    if len(dups)==0: st.success("No obvious duplicates detected.")
    else: st.dataframe(dups[['date','description','amount','account','category']], use_container_width=True, height=300)

    st.subheader("🚨 Amount Anomalies (IsolationForest)")
    anomalies = isolation_anomalies(df_f, contamination=anom_contam)
    if len(anomalies)==0: st.success("No amount anomalies detected (with current data).")
    else:
        show = anomalies[['date','description','amount','score']].sort_values('score')
        st.dataframe(show, use_container_width=True, height=300)

# -----------------------------
elif page == "Compare Years":
    st.subheader("📆 Year-to-Year Comparison")
    years_multi = st.sidebar.multiselect("Compare years (up to 4)", years_avail, max_selections=4)
    if not years_multi:
        st.info("Pick up to 4 years in the sidebar (top).")
    else:
        cdf = df[df['y'].isin(years_multi)].copy()
        totals = cdf.groupby('y')['amount'].sum().reset_index(name='Net')
        st.plotly_chart(px.bar(totals, x='y', y='Net', title="Net cashflow by year", labels={'y':'Year'}), use_container_width=True)
        monthly = (cdf.groupby(['m','y'])['amount'].sum().reset_index())
        monthly['_order'] = pd.to_datetime(monthly['m']+'-01')
        monthly = monthly.sort_values('_order')
        st.plotly_chart(px.line(monthly, x='m', y='amount', color='y', title="Monthly net by year", labels={'m':'Month'}), use_container_width=True)
        spend_by_cat = (cdf[cdf['amount']<0].groupby(['y','category'])['amount'].sum().abs().reset_index())
        st.plotly_chart(px.bar(spend_by_cat, x='category', y='amount', color='y', barmode='group',
                               title="Yearly spend by category"), use_container_width=True)

# -----------------------------
elif page == "Smart Advisor":
    st.subheader("🧠 Smart Advisor – Savings Opportunities")

    # 1) Pareto: 80% of spend comes from top categories
    spend_cat = df[df['amount']<0].groupby('category')['amount'].sum().abs().sort_values(ascending=False)
    pareto = spend_cat.cumsum()/spend_cat.sum()
    pareto_tbl = pd.DataFrame({"category":spend_cat.index, "spend":spend_cat.values, "cum_share":pareto.values})
    st.markdown("**Pareto categories (cover ≈80% of expenses):**")
    st.dataframe(pareto_tbl[pareto_tbl["cum_share"]<=0.8], use_container_width=True, height=260)

    # 2) Hot categories: latest month vs 3-month baseline (outflows only)
    if len(df) > 0:
        last_month = df['m'].max()
        last_dt = pd.to_datetime(last_month + "-01")
        prev3 = pd.date_range(end=last_dt, periods=4, freq='MS')[:-1].strftime("%Y-%m")
        base = df[(df['amount']<0) & (df['m'].isin(prev3))].groupby('category')['amount'].sum().abs()
        latest = df[(df['amount']<0) & (df['m']==last_month)].groupby('category')['amount'].sum().abs()
        hot = (pd.DataFrame({"baseline_3m_avg":base/3}).join(latest.rename("latest"), how="outer")
               .fillna(0.0))
        hot["delta"] = hot["latest"] - hot["baseline_3m_avg"]
        hot = hot.sort_values("delta", ascending=False).head(10)
        st.markdown(f"**Hot categories (spike in {last_month} vs 3-month avg):**")
        st.dataframe(hot, use_container_width=True, height=280)

    # 3) Subscription creep: recurring with rising trend
    rec_all = recurring_candidates(df)
    if len(rec_all):
        rec_all['trend'] = rec_all.apply(lambda r: df[(df['desc_norm']==r['desc_norm'])].groupby('m')['amount']
                                         .sum().abs().reset_index()['amount'].tail(3).diff().mean(), axis=1)
        creep = rec_all.sort_values('trend', ascending=False).head(10)
        st.markdown("**Subscriptions trending up (last 3 months):**")
        st.dataframe(creep[['merchant','months','txns','mean_amt','std_amt','first','last','trend']], use_container_width=True, height=260)
    else:
        st.info("No recurring patterns yet for subscription analysis.")

    # 4) Fees/charges
    fee_mask = np.zeros(len(df), dtype=bool)
    for kw in FEE_KEYWORDS:
        fee_mask |= df['desc_norm'].str.contains(re.escape(kw))
    fees = df[(df['amount']<0) & fee_mask]
    st.markdown("**Potential fees/charges to reduce:**")
    if len(fees): st.dataframe(fees[['date','description','amount','account','category']].sort_values('date', ascending=False), use_container_width=True, height=240)
    else: st.success("No obvious fee keywords found.")

    # 5) Suggested Savings Plan
    target_pct = st.slider("Target savings % on top Pareto categories", 5, 30, 10, step=1)
    suggest = pareto_tbl[pareto_tbl["cum_share"]<=0.8].copy()
    suggest["suggested_cut"] = (suggest["spend"] * target_pct/100.0).round(2)
    st.markdown("**Suggested savings plan (apply % cut to top categories):**")
    st.dataframe(suggest.rename(columns={"spend":"monthly_spend"}), use_container_width=True, height=260)

# -----------------------------
elif page == "Export":
    st.subheader("📤 Export current filtered data")
    download_csv(df_f.sort_values('date'), name="bank_transactions_filtered.csv")
    st.dataframe(df_f.sort_values('date'), use_container_width=True, height=420)

# Footer
st.markdown("---")
st.caption("Tips: refine Category Rules • Use Advanced Filters for noisy data • Smart Advisor highlights where to save • Upload multiple CSVs/XLSX to merge banks/cards")
