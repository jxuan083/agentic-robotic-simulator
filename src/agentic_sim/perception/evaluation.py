"""Engine-independent metrics for image-only cube perception evaluations."""
from __future__ import annotations
import math
import numpy as np

def _summary(xs):
    if not xs: return {"n":0,"mean":None,"median":None,"p95":None,"max":None}
    a=np.asarray(xs,float)
    return {"n":len(a),"mean":float(a.mean()),"median":float(np.median(a)),"p95":float(np.percentile(a,95)),"max":float(a.max())}

def _yaw_error(a,b):
    p=math.pi/2
    return math.degrees(abs((float(a)-float(b)+p/2)%p-p/2))

def _assign(pred,truth,gate=.10):
    ds=[[float(np.linalg.norm(np.asarray(p["center_xyz_m"])-np.asarray(t["center_xyz_m"]))) for t in truth] for p in pred]
    best=[]
    def visit(i,used,pairs,total):
        nonlocal best
        if i==len(pred):
            if len(pairs)>len(best) or (len(pairs)==len(best) and total<sum(x[2] for x in best)): best=list(pairs)
            return
        visit(i+1,used,pairs,total)
        for j,d in enumerate(ds[i]):
            if j not in used and d<=gate: visit(i+1,used|{j},pairs+[(i,j,d)],total+d)
    visit(0,set(),[],0.)
    return best

def camera_metrics(records,rejected,truth,colors):
    pred=[r["geometry"] for r in records if r["geometry"]["status"]=="estimated"]
    pairs=_assign(pred,truth); usedp={i for i,j,d in pairs}; usedt={j for i,j,d in pairs}; rows=[]
    for i,j,d in pairs:
        p,t=pred[i],truth[j]; a=np.asarray([v["xyz_m"] for v in p["vertices"]],float); b=np.asarray(t["corners_xyz_m"],float)
        corner=min(float(np.sqrt(np.mean(np.sum((a-b[[(k+n)%4 for n in range(4)]+[4+(k+n)%4 for n in range(4)]])**2,axis=1)))) for k in range(4))*1000
        rec=next(r for r in records if r["geometry"] is p)
        rows.append({"gt_index":j,"detection_id":rec["detection_id"],"status":"matched","center_error_mm":d*1000,"yaw_error_deg_mod90":_yaw_error(p["yaw_rad_mod_pi_over_2"],t["yaw_rad"]),"corner_rmse_mm":corner,"predicted_color":rec["color"],"ground_truth_color":colors[j],"color_error":rec["color"]!=colors[j]})
    for j in range(len(truth)):
        if j not in usedt: rows.append({"gt_index":j,"detection_id":None,"status":"missed","center_error_mm":None,"yaw_error_deg_mod90":None,"corner_rmse_mm":None,"predicted_color":None,"ground_truth_color":colors[j],"color_error":None})
    for i,p in enumerate(pred):
        if i not in usedp:
            r=next(r for r in records if r["geometry"] is p); rows.append({"gt_index":None,"detection_id":r["detection_id"],"status":"false_positive","center_error_mm":None,"yaw_error_deg_mod90":None,"corner_rmse_mm":None,"predicted_color":r["color"],"ground_truth_color":None,"color_error":None})
    for r in rejected: rows.append({"gt_index":None,"detection_id":r["detection_id"],"status":"rejected","center_error_mm":None,"yaw_error_deg_mod90":None,"corner_rmse_mm":None,"predicted_color":r["color"],"ground_truth_color":None,"color_error":None})
    matched=[r for r in rows if r["status"]=="matched"]
    return {"ground_truth_n":len(truth),"estimated_n":len(pred),"matched_n":len(pairs),"missed_n":len(truth)-len(usedt),"false_positive_n":len(pred)-len(usedp),"rejected_n":len(rejected),"center_error_mm_matched":_summary([r["center_error_mm"] for r in matched]),"yaw_error_deg_mod90_matched":_summary([r["yaw_error_deg_mod90"] for r in matched]),"corner_rmse_mm_matched":_summary([r["corner_rmse_mm"] for r in matched]),"color_error_n_matched":sum(r["color_error"] for r in matched),"per_ground_truth":rows}

def cross_camera(g,w):
    g=[r for r in g if r["geometry"]["status"]=="estimated"]; w=[r for r in w if r["geometry"]["status"]=="estimated"]
    cs=sorted((float(np.linalg.norm(np.asarray(a["geometry"]["center_xyz_m"])-np.asarray(b["geometry"]["center_xyz_m"]))),i,j) for i,a in enumerate(g) for j,b in enumerate(w) if np.linalg.norm(np.asarray(a["geometry"]["center_xyz_m"])-np.asarray(b["geometry"]["center_xyz_m"]))<=.015)
    ug,uw,p=set(),set(),[]
    for d,i,j in cs:
        if i in ug or j in uw: continue
        ug.add(i);uw.add(j);p.append({"global_detection_id":g[i]["detection_id"],"wrist_detection_id":w[j]["detection_id"],"center_delta_mm":d*1000,"colors_agree":g[i]["color"]==w[j]["color"],"yaw_delta_deg_mod90":_yaw_error(g[i]["geometry"]["yaw_rad_mod_pi_over_2"],w[j]["geometry"]["yaw_rad_mod_pi_over_2"])})
    return {"association":"predicted_3d_nearest_one_to_one_gate_15mm_no_GT","matched_n":len(p),"global_unpaired_n":len(g)-len(ug),"wrist_unpaired_n":len(w)-len(uw),"color_disagreement_n":sum(not x["colors_agree"] for x in p),"center_delta_mm":_summary([x["center_delta_mm"] for x in p]),"yaw_delta_deg_mod90":_summary([x["yaw_delta_deg_mod90"] for x in p]),"pairs":p}

def summarize(episodes,seed):
    out={"seed":seed,"episodes_n":len(episodes),"ground_truth_n_per_camera":sum(len(e["settled_ground_truth"]) for e in episodes),"cameras":{},"cross_camera":{}}
    for k in ("global","wrist"):
        ms=[e["cameras"][k]["metrics"] for e in episodes]; rs=[r for e in episodes for r in e["cameras"][k]["metrics"]["per_ground_truth"]]; m=[r for r in rs if r["status"]=="matched"]; gt=sum(x["ground_truth_n"] for x in ms)
        out["cameras"][k]={"ground_truth_n":gt,"matched_n":sum(x["matched_n"] for x in ms),"missed_n":sum(x["missed_n"] for x in ms),"false_positive_n":sum(x["false_positive_n"] for x in ms),"rejected_n":sum(x["rejected_n"] for x in ms),"matched_center_error_mm":_summary([r["center_error_mm"] for r in m]),"matched_yaw_error_deg_mod90":_summary([r["yaw_error_deg_mod90"] for r in m]),"matched_corner_rmse_mm":_summary([r["corner_rmse_mm"] for r in m]),"center_le_5mm_over_all_gt":sum(r["center_error_mm"]<=5 for r in m)/max(1,gt),"yaw_le_5deg_over_all_gt":sum(r["yaw_error_deg_mod90"]<=5 for r in m)/max(1,gt),"corner_le_5mm_over_all_gt":sum(r["corner_rmse_mm"]<=5 for r in m)/max(1,gt),"color_error_n_matched":sum(x["color_error_n_matched"] for x in ms)}
    xs=[e["cross_camera"] for e in episodes]; ps=[p for x in xs for p in x["pairs"]]
    out["cross_camera"]={"matched_n":len(ps),"global_unpaired_n":sum(x["global_unpaired_n"] for x in xs),"wrist_unpaired_n":sum(x["wrist_unpaired_n"] for x in xs),"color_disagreement_n":sum(not p["colors_agree"] for p in ps),"center_disagreement_ge_5mm_n":sum(p["center_delta_mm"]>=5 for p in ps),"pair_center_delta_mm":_summary([p["center_delta_mm"] for p in ps]),"pair_yaw_delta_deg_mod90":_summary([p["yaw_delta_deg_mod90"] for p in ps])}
    return out
