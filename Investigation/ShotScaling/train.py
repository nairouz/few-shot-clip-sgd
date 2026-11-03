# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

from __future__ import annotations
import csv, gzip, hashlib, json, math, random, subprocess, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import autocast, GradScaler
import clip
from datasets.utils import build_data_loader
from p2.environment import snapshot as environment_snapshot
from p2.logging_utils import canonical_json_bytes, file_sha256
from p2.moment_estimation import denominator_direction, exact_topk_overlap, mismatch, extract_adam_ema
from p2.optimizer_variants import build_optimizer, configure_m0_first_step
from p2.progress import targets
from p2.scopes import build_scope_map
from p2.support_manifest import image_id
from p2.update_budget import gradient_norm, measured_update, snapshot_parameters

PLAIN_SOURCE_HASHES = {
    "CLIP/FT/Plain/main.py": "a4fc0051d6051ee213e608b2a63d6c64c132352f0213fc8beefd136889c1bba9",
    "CLIP/FT/Plain/run_utils.py": "e676797ab7e23de241edc18c0d9ed0de549d5b258ff4d8fb2e6c80ad13cb7ddd",
    "CLIP/FT/Plain/train.py": "8e00e012e91bbd2d57b7be4ece0dc93010978065ebb9a546fb24e15289ba9c88",
    "CLIP/FT/Plain/utils.py": "314a499550086a423b3db4f74618c64b9968c633f2bd5149d159655524f77b33",
    "CLIP/FT/Plain/datasets/__init__.py": "9c1b9039c4e9da3af4da2ac5086a81903a94f9ae1d87860c78c03c20c44cb590",
    "CLIP/FT/Plain/datasets/utils.py": "9d2af5161112e35202824eb9236de6df57e6d155b339d4fd1d0e04f8d2ff7ea2",
}

def mark_as_trainable(model):
    for parameter in model.parameters(): parameter.requires_grad = False
    for block in model.transformer.resblocks:
        for parameter in block.parameters(): parameter.requires_grad = True
    for block in model.visual.transformer.resblocks:
        for parameter in block.parameters(): parameter.requires_grad = True

def _rng():
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

def _restore(state):
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
    if state[3] is not None: torch.cuda.set_rng_state_all(state[3])

def _loss(model, images, targets, tokens):
    image = model.encode_image(images); image = image / image.norm(dim=-1, keepdim=True)
    text = model.encode_text(tokens); text = text / text.norm(dim=-1, keepdim=True)
    return F.cross_entropy(100 * image @ text.t(), targets)

def _evaluate(model, data, transform, dataset, device, workers):
    loader = build_data_loader(data, batch_size=256, tfm=transform, is_train=False, shuffle=False, num_workers=workers)
    tokens = clip.tokenize([dataset.template[0].format(x.replace("_"," ")) for x in dataset.classnames]).to(device)
    was_training = model.training; model.eval()
    loss_sum = correct = count = 0
    with torch.no_grad():
        text = model.encode_text(tokens); text = text / text.norm(dim=-1, keepdim=True)
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            image = model.encode_image(images); image = image / image.norm(dim=-1, keepdim=True)
            logits = 100 * image @ text.t()
            loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
            correct += int((logits.argmax(1) == labels).sum()); count += labels.numel()
    model.train(was_training)
    return {"loss": loss_sum/count, "accuracy": 100*correct/count, "count": count}

def _scope_parameters(parameters, scope_map):
    scopes = {"global": list(parameters), "vision": [], "text": []}
    for parameter in parameters:
        info = scope_map[parameter]
        scopes[info["encoder"]].append(parameter)
        scopes.setdefault(info["block"], []).append(parameter)
    return scopes

def _scope_budget(before_by_parameter, scope_parameters):
    before = [before_by_parameter[p] for p in scope_parameters]
    return measured_update(before, scope_parameters)

def _materialize(items, transform, device):
    images = [transform(Image.open(x.impath).convert("RGB")) for x in items]
    return torch.stack(images).to(device), torch.tensor([x.label for x in items], device=device)

def _diagnose(args, model, preprocess, dataset, full_dataset, parameters, scopes, optimizer, q_record, size):
    state, mode = _rng(), model.training
    model.eval()
    try:
        diag = args.diagnostic_manifests[size][0]
        support_map = {image_id(x): x for x in dataset.train_x}
        full_map = {image_id(x): x for x in full_dataset.train_x}
        sums = {"support": {p: torch.zeros_like(p, dtype=torch.float32) for p in parameters},
                "full_reference": {p: torch.zeros_like(p, dtype=torch.float32) for p in parameters}}
        squares = {"support": {p: torch.zeros_like(p, dtype=torch.float32) for p in parameters},
                   "full_reference": {p: torch.zeros_like(p, dtype=torch.float32) for p in parameters}}
        tokens = clip.tokenize([dataset.template[0].format(x.replace("_"," ")) for x in dataset.classnames]).to(args.device)
        for batch in diag["batches"]:
            for source, key, lookup in (("support","support_image_ids",support_map),("full_reference","full_reference_image_ids",full_map)):
                images, labels = _materialize([lookup[x] for x in batch[key]], preprocess, args.device)
                loss = _loss(model, images, labels, tokens)
                grads = torch.autograd.grad(loss, parameters, allow_unused=False)
                for parameter, grad in zip(parameters, grads):
                    value = grad.detach().float()
                    sums[source][parameter].add_(value); squares[source][parameter].add_(value.square())
        ema = extract_adam_ema(optimizer, parameters) if args.optimizer_variant != "sgd" else {}
        rows = []
        histogram = {}
        for scope_name, scoped in scopes.items():
            mf = torch.cat([(sums["support"][p]/64).flatten() for p in scoped])
            mr = torch.cat([(sums["full_reference"][p]/64).flatten() for p in scoped])
            vf = torch.cat([(squares["support"][p]/64).flatten() for p in scoped])
            vr = torch.cat([(squares["full_reference"][p]/64).flatten() for p in scoped])
            vi = mf.square()
            base = {"requested_q": q_record["requested_q"], "target_step": q_record["target_step"],
                    "actual_progress": q_record["actual_progress"], "diag_batch_size": size, "scope": scope_name,
                    "m_few_norm": float(mf.norm()), "m_full_norm": float(mr.norm()),
                    "r_m": float(mf.norm()/(mr.norm()+1e-30)),
                    "gradient_cosine": float(F.cosine_similarity(mf, mr, dim=0)),
                    "gradient_sign_agreement": float((mf.sign()==mr.sign()).float().mean()),
                    "r_v_batch": float(vf.norm()/(vr.norm()+1e-30)),
                    "r_v_instant": float(vi.norm()/(vr.norm()+1e-30))}
            for source_name, value in (("batch",vf),("instant",vi)):
                base.update({f"{source_name}_{k}": v for k,v in mismatch(value,vr).items()})
            for definition in ("adam_faithful","paper_proxy"):
                uff=denominator_direction(mf,vf,1e-8,definition); ufr=denominator_direction(mf,vr,1e-8,definition); urr=denominator_direction(mr,vr,1e-8,definition)
                prefix=definition
                base.update({f"{prefix}_cos_uff_ufr":float(F.cosine_similarity(uff,ufr,dim=0)),
                    f"{prefix}_cos_ufr_urr":float(F.cosine_similarity(ufr,urr,dim=0)),
                    f"{prefix}_uff_norm":float(uff.norm()),f"{prefix}_ufr_norm":float(ufr.norm()),
                    f"{prefix}_urr_norm":float(urr.norm()),f"{prefix}_r_u":float(uff.norm()/(urr.norm()+1e-30))})
            for density in (.001,.01,.1):
                k, overlap=exact_topk_overlap(vf,vr,density); tag=str(density).replace(".","p")
                base[f"topk_k_{tag}"]=k; base[f"topk_overlap_{tag}"]=overlap
            available = bool(ema) and all(ema[p] is not None for p in scoped)
            base["ema_state_available"] = available
            if available:
                ve=torch.cat([ema[p].flatten() for p in scoped]); base["r_v_ema"]=float(ve.norm()/(vr.norm()+1e-30))
                base.update({f"ema_{k}":v for k,v in mismatch(ve,vr).items()})
                for definition in ("adam_faithful","paper_proxy"):
                    uema=denominator_direction(mf,ve,1e-8,definition)
                    urr=denominator_direction(mr,vr,1e-8,definition)
                    base[f"{definition}_ema_direction_norm"]=float(uema.norm())
                    base[f"{definition}_ema_vs_urr_cos"]=float(F.cosine_similarity(uema,urr,dim=0))
                    base[f"{definition}_ema_vs_urr_norm_ratio"]=float(uema.norm()/(urr.norm()+1e-30))
            else: base["r_v_ema"]=float("nan")
            z=torch.log10((vf+1e-30)/(vr+1e-30)).detach()
            histogram[f"{scope_name}_b{size}_q{q_record['target_step']}"]=torch.histc(
                z, bins=120, min=-6, max=6).to(torch.int64).cpu().numpy()
            rows.append(base)
        return rows, histogram
    finally:
        model.train(mode); _restore(state)

def _write_csv(path, rows):
    if not rows: path.write_text(""); return
    keys=[] 
    for row in rows:
        for key in row:
            if key not in keys: keys.append(key)
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys); writer.writeheader(); writer.writerows(rows)

def run_p2(args, model, preprocess, dataset, full_dataset, train_loader, generator):
    run_dir=Path(args.run_dir); environment=environment_snapshot(args.pool,args.physical_gpu_id)
    (run_dir/"environment.json").write_bytes(canonical_json_bytes(environment))
    try:
        mark_as_trainable(model); model=model.to(args.device)
        parameters=[p for p in model.parameters() if p.requires_grad]
        scope_map=build_scope_map(model); scopes=_scope_parameters(parameters,scope_map)
        if len(scopes)!=27: raise RuntimeError(f"expected global+2 encoders+24 blocks, got {len(scopes)}")
        optimizer=build_optimizer(args.optimizer_variant,parameters)
        # Plain's default 65536 scale overflows the very first full-CLIP
        # backward on both A100 and A6000 for EuroSAT K=1. P2 must measure a
        # real first update (especially M0), so use one common conservative
        # initial scale. Unscaled finite gradients/updates are unchanged.
        amp_initial_scale=1024.0
        scaler=GradScaler("cuda", init_scale=amp_initial_scale)
        total_updates=args.epochs*math.ceil(len(dataset.train_x)/64)
        if args.smoke and total_updates < 5:
            smoke_steps=sorted({0,1,total_updates})
            q_records=[{"requested_q":step/total_updates,"target_step":step,"actual_progress":step/total_updates} for step in smoke_steps]
        else:
            q_records=targets(total_updates)
        q_by_step={x["target_step"]:x for x in q_records}
        try:
            git_sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
        except subprocess.CalledProcessError:
            git_sha="UNCOMMITTED_SMOKE"
        serializable_args={k:v for k,v in vars(args).items() if k!="diagnostic_manifests"}
        source_root=Path(__file__).resolve().parent
        plain_names=("main.py","run_utils.py","train.py","utils.py","datasets/__init__.py","datasets/utils.py")
        config={**serializable_args,"run_dir":str(run_dir.resolve()),"support_set_size":len(dataset.train_x),
            "nominal_batch_size":64,"drop_last":False,"batches_per_epoch":math.ceil(len(dataset.train_x)/64),
            "total_optimizer_updates":total_updates,"q_to_step_mapping":q_records,"git_sha":git_sha,
            "augmentation":args.protocol=="augmented","optimizer_betas":[.9,.999] if args.optimizer_variant!="sgd" else None,
            "optimizer_epsilon":1e-8 if args.optimizer_variant!="sgd" else None,
            "amp_initial_scale":amp_initial_scale,
            "coordinate_percentile_method":"65536-bin bounded-memory histogram",
            "coordinate_percentile_max_error":"observed max(abs(log-ratio))/65536"}
        config["plain_source_hashes"]=PLAIN_SOURCE_HASHES
        config["local_training_source_hashes"]={name:file_sha256(source_root/name) for name in plain_names}
        config["p2_source_hashes"]={str(path.relative_to(source_root)):file_sha256(path)
            for path in sorted((source_root/"p2").glob("*.py"))}
        config["diagnostic_manifests"]={str(k):v[1] for k,v in args.diagnostic_manifests.items()}
        (run_dir/"config.json").write_bytes(canonical_json_bytes(config))
        tokens=clip.tokenize([dataset.template[0].format(x.replace("_"," ")) for x in dataset.classnames]).to(args.device)
        initial={p:p.detach().clone() for p in parameters}
        update_rows=[]; diagnostic_rows=[]; epoch_rows=[]; histograms={}; order_hashes=[]; m0=None
        first_batch_hashes=[]
        def checkpoint(q_record):
            new_rows=[]
            for size in args.diagnostic_manifests:
                rows,hist=_diagnose(args,model,preprocess,dataset,full_dataset,parameters,scopes,optimizer,q_record,size)
                new_rows+=rows; histograms.update(hist)
            evaluations={name:_evaluate(model,data,preprocess,dataset,args.device,args.num_workers) for name,data in
                (("support",dataset.train_x),("full_reference",full_dataset.train_x),
                 ("validation",dataset.val),("test",dataset.test))}
            for row in new_rows:
                scoped=scopes[row["scope"]]
                now=torch.cat([p.detach().float().flatten() for p in scoped])
                start=torch.cat([initial[p].float().flatten() for p in scoped])
                row["parameter_drift"]=float((now-start).norm()/(start.norm()+1e-30))
                for split,values in evaluations.items():
                    row[f"{split}_loss"]=values["loss"];row[f"{split}_accuracy"]=values["accuracy"]
                row["full_minus_support_loss"]=evaluations["full_reference"]["loss"]-evaluations["support"]["loss"]
                row["test_minus_support_loss"]=evaluations["test"]["loss"]-evaluations["support"]["loss"]
            diagnostic_rows.extend(new_rows)
        # q=0 is strictly before the first update.
        checkpoint(q_by_step[0])
        global_step=0
        for epoch in range(1,args.epochs+1):
            model.train(); order=hashlib.sha256(); batch_sizes=[]; loss_sum=correct=count=0
            for batch_index,(images,labels,image_ids) in enumerate(train_loader):
                batch_sizes.append(labels.numel())
                for identifier in image_ids: order.update(identifier.encode())
                images,labels=images.to(args.device),labels.to(args.device)
                if epoch==1 and batch_index<2: first_batch_hashes.append(hashlib.sha256(images.detach().cpu().numpy().tobytes()).hexdigest())
                optimizer.zero_grad(set_to_none=True)
                with autocast("cuda"):
                    loss=_loss(model,images,labels,tokens)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                if args.optimizer_variant=="adam_m0" and global_step==0: m0=configure_m0_first_step(optimizer,parameters)
                before={p:p.detach().clone() for p in parameters}; grad=gradient_norm(parameters); scale=float(scaler.get_scale())
                finite=all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
                scaler.step(optimizer); scaler.update(); skipped=float(scaler.get_scale())<scale
                global_step+=1
                for scope_name,scoped in scopes.items():
                    row={"global_step":global_step,"epoch":epoch,"batch_index":batch_index,
                        "actual_batch_size":labels.numel(),"scope":scope_name,"current_lr":optimizer.param_groups[0]["lr"],
                        "gradient_norm":gradient_norm(scoped),"amp_scale":scale,
                        "optimizer_step_skipped":skipped,"non_finite_observed":not finite}
                    row.update(_scope_budget(before,scoped)); update_rows.append(row)
                with torch.no_grad():
                    logits_loss=loss.item(); loss_sum+=logits_loss*labels.numel(); count+=labels.numel()
                if global_step in q_by_step:
                    checkpoint(q_by_step[global_step])
            order_hashes.append({"epoch":epoch,"ordered_support_image_id_hash":order.hexdigest(),"actual_batch_sizes":batch_sizes})
            epoch_rows.append({"epoch":epoch,"final_step":global_step,"train_loss":loss_sum/count})
        evaluations={name:_evaluate(model,data,preprocess,dataset,args.device,args.num_workers) for name,data in
            (("support",dataset.train_x),("full_reference",full_dataset.train_x),("validation",dataset.val),("test",dataset.test))}
        _write_csv(run_dir/"epoch_metrics.csv",epoch_rows)
        with gzip.open(run_dir/"update_metrics.csv.gz","wt",newline="") as stream:
            keys=list(update_rows[0]); writer=csv.DictWriter(stream,fieldnames=keys); writer.writeheader(); writer.writerows(update_rows)
        _write_csv(run_dir/"diagnostic_global.csv",[x for x in diagnostic_rows if x["scope"]=="global"])
        _write_csv(run_dir/"diagnostic_encoder.csv",[x for x in diagnostic_rows if x["scope"] in {"vision","text"}])
        _write_csv(run_dir/"diagnostic_block.csv",[x for x in diagnostic_rows if x["scope"] not in {"global","vision","text"}])
        np.savez_compressed(run_dir/"diagnostic_histograms.npz",**histograms)
        (run_dir/"training_order_hashes.json").write_bytes(canonical_json_bytes({"epochs":order_hashes,"first_two_augmented_batch_hashes":first_batch_hashes}))
        summary={"run_id":args.run_id,"final_epoch":args.epochs,"final_optimizer_step":global_step,
            "evaluations":evaluations,"adam_m0":m0,"non_finite":any(x["non_finite_observed"] for x in update_rows)}
        if m0:
            applied=next(x["relative_update_budget"] for x in update_rows if x["global_step"]==1 and x["scope"]=="global")
            m0["applied_m0_first_step_budget"]=applied
            m0["relative_matching_error"]=abs(applied-m0["reference_sgd_first_step_budget"])/(m0["reference_sgd_first_step_budget"]+1e-30)
        collapse={}
        for scope_name in scopes:
            primary=[x for x in diagnostic_rows if x["scope"]==scope_name and x["diag_batch_size"]==16]
            by_q={x["requested_q"]:x for x in primary}
            if 0.0 in by_q and 1.0 in by_q:
                start,end=by_q[0.0],by_q[1.0]
                collapse[scope_name]={"c_m":start["r_m"]/(end["r_m"]+1e-30),
                    "c_v":start["r_v_batch"]/(end["r_v_batch"]+1e-30)}
                collapse[scope_name]["collapse_ratio"]=collapse[scope_name]["c_v"]/(collapse[scope_name]["c_m"]+1e-30)
        summary["collapse"]=collapse
        (run_dir/"summary.json").write_bytes(canonical_json_bytes(summary))
        environment["end_timestamp"]=datetime.now(timezone.utc).isoformat()
        (run_dir/"environment.json").write_bytes(canonical_json_bytes(environment))
        outputs={p.name:file_sha256(p) for p in run_dir.iterdir() if p.is_file() and p.name not in {"DONE.json","stdout.log"}}
        done={"run_id":args.run_id,"command":" ".join(__import__("sys").argv),"git_sha":git_sha,"output_checksums":outputs,
            "final_epoch":args.epochs,"final_optimizer_step":global_step,
            "expected_diagnostic_checkpoints":[x["target_step"] for x in q_records],
            "observed_diagnostic_checkpoints":sorted({x["target_step"] for x in diagnostic_rows}),
            "success_timestamp":datetime.now(timezone.utc).isoformat()}
        (run_dir/"stdout.log").touch(); (run_dir/"DONE.json").write_bytes(canonical_json_bytes(done))
    except Exception as exc:
        (run_dir/"FAILED.json").write_bytes(canonical_json_bytes({"error":repr(exc),"traceback":traceback.format_exc(),
            "timestamp":datetime.now(timezone.utc).isoformat()}))
        raise
