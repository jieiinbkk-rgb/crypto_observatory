"""
strategy/classifier.py  v2
"""
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats
from config.settings import MARKET_STATES, VOL_REGIMES

@st.cache_resource
def get_classifier_store():  # v2
    return {"gmm_model":None,"gmm_scaler":None,"gmm_label_map":{},"state_history":[],"transition_cache":None}

GMM_FEATURES = ["BTC_Z","ETH_Z","Ratio_Z","BTC_Mom5","ETH_Mom5","IV_Div_Z","Funding_Z","FG_Z"]

def fit_gmm(store, df):
    available = [f for f in GMM_FEATURES if f in df.columns]
    sub = df[available].dropna()
    if len(sub) < 30: return
    scaler = StandardScaler()
    Xs = scaler.fit_transform(sub.values)
    gmm = GaussianMixture(n_components=4,covariance_type="full",random_state=42,max_iter=300,n_init=3)
    gmm.fit(Xs)
    means = scaler.inverse_transform(gmm.means_)
    label_map = {}
    for i,m in enumerate(means):
        btc_z=m[0] if len(m)>0 else 0
        eth_z=m[1] if len(m)>1 else 0
        ratio_z=m[2] if len(m)>2 else 0
        btc_mom=m[3] if len(m)>3 else 0
        eth_mom=m[4] if len(m)>4 else 0
        iv_div=m[5] if len(m)>5 else 0
        avg_z=(abs(btc_z)+abs(eth_z))/2
        avg_mom=(abs(btc_mom)+abs(eth_mom))/2
        if avg_z>1.8 or avg_mom>0.03: label_map[i]="panic"
        elif btc_z<-0.5 and eth_z<-0.5 and avg_mom<0.01: label_map[i]="squeeze"
        elif abs(ratio_z)>0.8 or abs(iv_div)>0.5: label_map[i]="hedging"
        else: label_map[i]="risk_on"
    store["gmm_model"]=gmm; store["gmm_scaler"]=scaler
    store["gmm_label_map"]=label_map; store["transition_cache"]=None

def classify_state(store, df):
    available=[f for f in GMM_FEATURES if f in df.columns]
    latest_row=df[available].dropna().tail(1)
    if latest_row.empty: return "unknown",0.0,"no_data"
    row=latest_row.iloc[0]
    values=[float(row.get(f,0) or 0) for f in available]
    btc_z=float(row.get("BTC_Z",0) or 0)
    eth_z=float(row.get("ETH_Z",0) or 0)
    ratio_z=float(row.get("Ratio_Z",0) or 0)
    btc_mom=float(row.get("BTC_Mom5",0) or 0)
    if store.get("gmm_model") and len(values)>=3:
        try:
            scaler=store["gmm_scaler"]; gmm=store["gmm_model"]; lmap=store["gmm_label_map"]
            n=gmm.means_.shape[1]
            padded=values[:n]+[0]*max(0,n-len(values))
            Xs=scaler.transform([padded]); proba=gmm.predict_proba(Xs)[0]
            cluster=int(np.argmax(proba)); confidence=float(proba[cluster])
            return lmap.get(cluster,"unknown"),confidence,"GMM"
        except: pass
    avg_z=(abs(btc_z)+abs(eth_z))/2
    if avg_z>2.0 or abs(btc_mom)>0.05: return "panic",0.80,"rule"
    if btc_z<-0.5 and eth_z<-0.5: return "risk_on",0.70,"rule"
    if abs(ratio_z)>0.8: return "hedging",0.65,"rule"
    if avg_z<0.3: return "squeeze",0.60,"rule"
    return "risk_on",0.55,"rule"

def record_state(store, state_key, confidence, method):
    store["state_history"].append({"Timestamp":datetime.utcnow().strftime("%Y-%m-%d %H:%M"),"State":state_key,"Confidence":round(confidence,4),"Method":method})
    if len(store["state_history"])>1000: store["state_history"]=store["state_history"][-1000:]
    store["transition_cache"]=None

def get_state_history_df(store):
    if not store["state_history"]: return pd.DataFrame()
    df=pd.DataFrame(store["state_history"]); df["Timestamp"]=pd.to_datetime(df["Timestamp"]); return df

def classify_vol_regime(df):
    iv_series=df["BTC_IV"].dropna()
    if len(iv_series)<10: return "normal",VOL_REGIMES["normal"]["label"],50.0
    current=float(iv_series.iloc[-1]); pct=float(stats.percentileofscore(iv_series.tail(500),current))
    for key,info in VOL_REGIMES.items():
        lo,hi=info["pct_range"]
        if lo<=pct<hi: return key,info["label"],pct
    return "very_high",VOL_REGIMES["very_high"]["label"],pct

def compute_transition_matrix(store):
    if store.get("transition_cache") is not None: return store["transition_cache"]
    hist=store["state_history"]
    if len(hist)<10: return None
    states=[h["State"] for h in hist]
    labels=[s for s in MARKET_STATES if s!="unknown"]
    matrix=pd.DataFrame(0,index=labels,columns=labels)
    for i in range(len(states)-1):
        s_from,s_to=states[i],states[i+1]
        if s_from in matrix.index and s_to in matrix.columns: matrix.loc[s_from,s_to]+=1
    row_sums=matrix.sum(axis=1).replace(0,float("nan"))
    result=(matrix.div(row_sums,axis=0)*100).round(1).fillna(0)
    store["transition_cache"]=result; return result

def get_next_state_probability(store, current_state):
    mx=compute_transition_matrix(store)
    if mx is None or current_state not in mx.index: return {}
    return mx.loc[current_state].to_dict()

def compute_opportunity_score(df, state_key, confidence):
    score=0; reasons=[]
    if df.empty or len(df)<5: return 0,["データ不足"]
    latest=df.iloc[-1]
    conf_pts=int(confidence*25); score+=conf_pts; reasons.append(f"GMM信頼度 {confidence*100:.0f}% → +{conf_pts}pt")
    state_bonus={"panic":20,"squeeze":18,"hedging":12,"risk_on":8,"unknown":0}
    sb=state_bonus.get(state_key,0); score+=sb; reasons.append(f"状態[{state_key}] → +{sb}pt")
    sp20=float(latest.get("BTC_Spread20") or 0); sp_pts=min(20,int(abs(sp20)/3)); score+=sp_pts
    if sp_pts>5: reasons.append(f"IV-RVスプレッド {sp20:+.1f} → +{sp_pts}pt")
    btc_z=abs(float(latest.get("BTC_Z") or 0)); z_pts=min(15,int(btc_z*4)); score+=z_pts
    if z_pts>4: reasons.append(f"|BTC_Z| {btc_z:.2f} → +{z_pts}pt")
    mom=abs(float(latest.get("BTC_Mom5") or 0)); m_pts=min(10,int(mom*200)); score+=m_pts
    if m_pts>3: reasons.append(f"IV Mom {mom*100:.1f}% → +{m_pts}pt")
    vc=float(latest.get("Vol_Compression") or 1.0)
    if vc<0.3:
        vc_pts=min(10,int((0.3-vc)*33)); score+=vc_pts; reasons.append(f"Vol圧縮 {vc:.2f} → +{vc_pts}pt")
    return min(100,max(0,score)),reasons
