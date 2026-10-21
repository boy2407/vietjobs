import json
import time
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from vietjobs import config as C
from vietjobs import dataset as D
from vietjobs import evaluate as E
from vietjobs import external as X
from vietjobs import vitext as V
from vietjobs.dl import encode as ENC
from vietjobs.dl import text as T
from vietjobs.dl.heads import MultiTaskHead, DenseHead

ENCODER_SPECS = {
    'phobert': {'max_len': 256, 'batch': 64, 'segmented': True, 'prefix': ''},
    'cafebert': {'max_len': 512, 'batch': 32, 'segmented': False, 'prefix': 'cafebert-'},
    'visobert': {'max_len': 512, 'batch': 32, 'segmented': False, 'prefix': 'visobert-'},
    'xlmr': {'max_len': 512, 'batch': 32, 'segmented': False, 'prefix': 'xlmr-'},
    'bamibert': {'max_len': 1024, 'batch': 16, 'segmented': False, 'prefix': 'bamibert-'}
}

MTL_MODELS = [
    ('dl-mtl-pho-dense-last4-cw-s42', 'phobert'),
    ('dl-mtl-caf-dense-last4-cw-s42', 'cafebert'),
    ('dl-mtl-vis-dense-last4-cw-s42', 'visobert'),
    ('dl-mtl-xlm-dense-last4-cw-s42', 'xlmr'),
    ('dl-mtl-bam-dense-last4-cw-s42', 'bamibert'),
]

SALARY_MODELS = [
    ('pho', 'phobert'),
    ('cafe', 'cafebert'),
    ('visobert', 'visobert'),
    ('xlmr', 'xlmr'),
    ('bamibert', 'bamibert'),
]

def load_data():
    print('Loading Test and Ext37K data...')
    df_test = D.load_split('test')
    print(f'Test split: {len(df_test):,} rows')
    
    sal_file = C.ROOT / 'data' / 'external' / 'vietjobs37k' / 'ext37k-sal.csv'
    full = pd.read_csv(sal_file, keep_default_na=False, na_values=[''])
    overlap, dedup = X.overlap_mask(full, X.OurIndex())
    keep = ~np.asarray(overlap)
    df_ext = full[keep].reset_index(drop=True)
    print(f'Ext37K: {len(df_ext):,} rows after dedup overlap')
    
    # Pre-segment Ext37K text for PhoBERT if needed
    for col in ('job_title_masked', 'description_masked', 'requirements_masked'):
        dst = f'{col}_seg'
        if dst not in df_ext.columns:
            print(f'Segmenting {col} for Ext37K...')
            df_ext[dst] = V.segment_many(df_ext[col])
            
    return df_test, df_ext

def get_or_create_embeddings(df_test, df_ext, encoder, spec):
    max_len = spec['max_len']
    batch = spec['batch']
    prefix = spec['prefix']
    cache_dir = C.ARTIFACT_DIR / 'embeddings'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    test_path = cache_dir / f'{prefix}test-masked-len{max_len}-last4.npy'
    ext_path = cache_dir / f'{prefix}ext37k-sal-masked-len{max_len}-last4.npy'
    
    if test_path.exists():
        print(f'  [Cache Hit] {test_path.name}')
        emb_test = np.load(test_path)
    else:
        print(f'  [Encoding Test] {encoder} @ len {max_len}...')
        texts_test = T.build_text(df_test, task=C.TASK_SALARY, segmented=spec['segmented'])
        t0 = time.time()
        emb_test = ENC.encode_texts(texts_test, max_len=max_len, batch=batch,
                                    device='cuda', encoder=encoder, last4=True)
        np.save(test_path, emb_test)
        print(f'  Encoded test in {time.time() - t0:.1f}s -> saved {test_path.name}')
        
    if ext_path.exists():
        print(f'  [Cache Hit] {ext_path.name}')
        emb_ext = np.load(ext_path)
    else:
        print(f'  [Encoding Ext37K] {encoder} @ len {max_len}...')
        texts_ext = T.build_text(df_ext, task=C.TASK_SALARY, segmented=spec['segmented'])
        t0 = time.time()
        emb_ext = ENC.encode_texts(texts_ext, max_len=max_len, batch=batch,
                                   device='cuda', encoder=encoder, last4=True)
        np.save(ext_path, emb_ext)
        print(f'  Encoded Ext37K in {time.time() - t0:.1f}s -> saved {ext_path.name}')
        
    return emb_test, emb_ext

def eval_mtl(df_test, df_ext, embeddings_dict, labels):
    print('\n==================================================')
    print('=== ĐÁNH GIÁ 5 MÔ HÌNH MTL DENSE TRÊN TEST & EXT37K ===')
    print('==================================================')
    n_classes = len(labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ground truth test
    y_true_cat_test = df_test['category'].to_numpy()
    keep_sal_test = ((df_test['salary_disclosed'] == 1) & df_test['salary_mid'].notna()).to_numpy()
    y_true_sal_test = df_test.loc[keep_sal_test, 'salary_mid'].to_numpy(dtype=float)
    y_true_log_test = np.log1p(y_true_sal_test)
    
    # Ground truth ext37k
    keep_cat_ext = df_ext['strict_target'].notna().to_numpy()
    y_true_cat_ext = df_ext.loc[keep_cat_ext, 'strict_target'].to_numpy()
    y_true_sal_ext = df_ext['salary_mid'].to_numpy(dtype=float)
    y_true_log_ext = np.log1p(y_true_sal_ext)
    
    mtl_results = []
    
    for run_id, encoder in MTL_MODELS:
        run_dir = C.ARTIFACT_DIR / run_id
        cfg = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
        sc = np.load(run_dir / 'scaler.npz')
        mu, sigma = sc['mean'], sc['std']
        
        emb_test, emb_ext = embeddings_dict[encoder]
        
        # Load models
        m_cat = MultiTaskHead(ENC.hidden_of(encoder), cfg['hidden'], n_classes,
                              cfg['dropout'], pre_hidden=cfg.get('pre_hidden', 0)).to(device)
        m_cat.load_state_dict(torch.load(run_dir / 'best.pt', map_location=device))
        m_cat.eval()
        
        m_sal = MultiTaskHead(ENC.hidden_of(encoder), cfg['hidden'], n_classes,
                              cfg['dropout'], pre_hidden=cfg.get('pre_hidden', 0)).to(device)
        m_sal.load_state_dict(torch.load(run_dir / 'best_sal.pt', map_location=device))
        m_sal.eval()
        
        # 1. Test Inference
        X_test_std = torch.from_numpy(((emb_test - mu) / sigma).astype(np.float32)).to(device)
        with torch.no_grad():
            probas_test = torch.softmax(m_cat(X_test_std)[0], dim=1).cpu().numpy()
            sals_log_test = m_sal(X_test_std)[1].squeeze(-1).cpu().numpy()
            
        pred_cats_test = np.array(labels)[probas_test.argmax(1)]
        pred_sals_test = np.expm1(sals_log_test)
        
        # Metrics Test
        res_cat_test = E.classification_metrics(y_true_cat_test, pred_cats_test, labels)
        
        y_pred_sal_eval = pred_sals_test[keep_sal_test]
        y_pred_log_eval = sals_log_test[keep_sal_test]
        res_sal_test = E.regression_metrics(y_true_log_test, y_pred_log_eval)
        res_sal_test['rmse_trieu'] = float(np.sqrt(np.mean((y_pred_sal_eval - y_true_sal_test) ** 2)))
        res_sal_test['mape'] = float(np.mean(np.abs((y_pred_sal_eval - y_true_sal_test) / np.clip(y_true_sal_test, 1e-3, None))))
        res_sal_test['within_3_trieu'] = float(np.mean(np.abs(y_pred_sal_eval - y_true_sal_test) <= 3.0))
        res_sal_test['within_20pct'] = float(np.mean(np.abs(y_pred_sal_eval - y_true_sal_test) <= 0.20 * y_true_sal_test))
        
        # 2. Ext37K Inference
        X_ext_std = torch.from_numpy(((emb_ext - mu) / sigma).astype(np.float32)).to(device)
        with torch.no_grad():
            probas_ext = torch.softmax(m_cat(X_ext_std)[0], dim=1).cpu().numpy()
            sals_log_ext = m_sal(X_ext_std)[1].squeeze(-1).cpu().numpy()
            
        pred_cats_ext = np.array(labels)[probas_ext.argmax(1)]
        pred_sals_ext = np.expm1(sals_log_ext)
        
        # Metrics Ext37K
        res_cat_ext = E.classification_metrics(y_true_cat_ext, pred_cats_ext[keep_cat_ext], labels)
        
        res_sal_ext = E.regression_metrics(y_true_log_ext, sals_log_ext)
        res_sal_ext['rmse_trieu'] = float(np.sqrt(np.mean((pred_sals_ext - y_true_sal_ext) ** 2)))
        res_sal_ext['spearman'] = float(spearmanr(y_true_sal_ext, pred_sals_ext).statistic)
        res_sal_ext['mape'] = float(np.mean(np.abs((pred_sals_ext - y_true_sal_ext) / np.clip(y_true_sal_ext, 1e-3, None))))
        res_sal_ext['within_3_trieu'] = float(np.mean(np.abs(pred_sals_ext - y_true_sal_ext) <= 3.0))
        res_sal_ext['within_20pct'] = float(np.mean(np.abs(pred_sals_ext - y_true_sal_ext) <= 0.20 * y_true_sal_ext))
        
        # Save metrics_test.json
        (run_dir / 'metrics_test.json').write_text(json.dumps({
            'run_id': run_id, 'eval': 'test', 'metrics_cat': res_cat_test, 'metrics_sal': res_sal_test
        }, indent=2, ensure_ascii=False), encoding='utf-8')
        
        # Save external_37k metrics.json
        ext_out_dir = run_dir / 'external_37k'
        ext_out_dir.mkdir(parents=True, exist_ok=True)
        (ext_out_dir / 'metrics.json').write_text(json.dumps({
            'run_id': run_id, 'metrics_cat_strict': res_cat_ext, 'metrics_sal': res_sal_ext
        }, indent=2, ensure_ascii=False), encoding='utf-8')
        
        row = {
            'run_id': run_id,
            'encoder': encoder,
            'test_f1': res_cat_test['f1_macro'],
            'test_acc': res_cat_test['accuracy'],
            'test_mae': res_sal_test['mae_trieu'],
            'test_rmse': res_sal_test['rmse_trieu'],
            'test_r2_log': res_sal_test['r2_log'],
            'ext_f1': res_cat_ext['f1_macro'],
            'ext_acc': res_cat_ext['accuracy'],
            'ext_mae': res_sal_ext['mae_trieu'],
            'ext_rmse': res_sal_ext['rmse_trieu'],
            'ext_r2_log': res_sal_ext['r2_log'],
            'ext_spearman': res_sal_ext['spearman']
        }
        mtl_results.append(row)
        print(f"[{run_id}] Test: F1={row['test_f1']:.4f}, Acc={row['test_acc']:.4f}, MAE={row['test_mae']:.3f}tr, RMSE={row['test_rmse']:.3f}tr, R2log={row['test_r2_log']:.3f} | Ext37k: F1={row['ext_f1']:.4f}, MAE={row['ext_mae']:.3f}tr, Spearman={row['ext_spearman']:.4f}")
        
    return mtl_results

def eval_salary_dense(df_test, df_ext, embeddings_dict):
    print('\n==================================================')
    print('=== ĐÁNH GIÁ 5 MÔ HÌNH SALARY DENSE (5 SEEDS) TRÊN TEST & EXT37K ===')
    print('==================================================')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    keep_sal_test = ((df_test['salary_disclosed'] == 1) & df_test['salary_mid'].notna()).to_numpy()
    y_true_sal_test = df_test.loc[keep_sal_test, 'salary_mid'].to_numpy(dtype=float)
    y_true_log_test = np.log1p(y_true_sal_test)
    
    y_true_sal_ext = df_ext['salary_mid'].to_numpy(dtype=float)
    y_true_log_ext = np.log1p(y_true_sal_ext)
    
    sal_results = []
    
    for short_name, encoder in SALARY_MODELS:
        emb_test, emb_ext = embeddings_dict[encoder]
        
        seeds_test_mae, seeds_test_rmse, seeds_test_r2 = [], [], []
        seeds_ext_mae, seeds_ext_rmse, seeds_ext_r2, seeds_ext_spearman = [], [], [], []
        
        for seed in (42, 43, 44, 45, 46):
            run_id = f'dl-sal-{short_name}-dense-last4-b01-cw-s{seed}'
            run_dir = C.ARTIFACT_DIR / run_id
            if not (run_dir / 'best.pt').exists():
                continue
            cfg = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
            sc = np.load(run_dir / 'scaler.npz')
            mu, sigma = sc['mean'], sc['std']
            
            m = DenseHead(ENC.hidden_of(encoder), cfg['hidden'], 1,
                          cfg['dropout'], pre_hidden=cfg.get('pre_hidden', 0)).to(device)
            m.load_state_dict(torch.load(run_dir / 'best.pt', map_location=device))
            m.eval()
            
            # Test inference
            X_test_std = torch.from_numpy(((emb_test - mu) / sigma).astype(np.float32)).to(device)
            with torch.no_grad():
                sals_log_test = m(X_test_std).squeeze(-1).cpu().numpy()
            pred_sals_test = np.expm1(sals_log_test)
            
            y_pred_sal_eval = pred_sals_test[keep_sal_test]
            y_pred_log_eval = sals_log_test[keep_sal_test]
            res_test = E.regression_metrics(y_true_log_test, y_pred_log_eval)
            res_test['rmse_trieu'] = float(np.sqrt(np.mean((y_pred_sal_eval - y_true_sal_test) ** 2)))
            
            seeds_test_mae.append(res_test['mae_trieu'])
            seeds_test_rmse.append(res_test['rmse_trieu'])
            seeds_test_r2.append(res_test['r2_log'])
            
            # Ext37k inference
            X_ext_std = torch.from_numpy(((emb_ext - mu) / sigma).astype(np.float32)).to(device)
            with torch.no_grad():
                sals_log_ext = m(X_ext_std).squeeze(-1).cpu().numpy()
            pred_sals_ext = np.expm1(sals_log_ext)
            
            res_ext = E.regression_metrics(y_true_log_ext, sals_log_ext)
            res_ext['rmse_trieu'] = float(np.sqrt(np.mean((pred_sals_ext - y_true_sal_ext) ** 2)))
            res_ext['spearman'] = float(spearmanr(y_true_sal_ext, pred_sals_ext).statistic)
            
            seeds_ext_mae.append(res_ext['mae_trieu'])
            seeds_ext_rmse.append(res_ext['rmse_trieu'])
            seeds_ext_r2.append(res_ext['r2_log'])
            seeds_ext_spearman.append(res_ext['spearman'])
            
            # Save single seed test metrics
            (run_dir / 'metrics_test.json').write_text(json.dumps({
                'run_id': run_id, 'eval': 'test', 'metrics_sal': res_test
            }, indent=2, ensure_ascii=False), encoding='utf-8')
            ext_dir = run_dir / 'external_37k'
            ext_dir.mkdir(parents=True, exist_ok=True)
            (ext_dir / 'metrics.json').write_text(json.dumps({
                'run_id': run_id, 'metrics_sal': res_ext
            }, indent=2, ensure_ascii=False), encoding='utf-8')
            
        row = {
            'short_name': short_name,
            'encoder': encoder,
            'test_mae_mean': float(np.mean(seeds_test_mae)), 'test_mae_std': float(np.std(seeds_test_mae)),
            'test_rmse_mean': float(np.mean(seeds_test_rmse)), 'test_rmse_std': float(np.std(seeds_test_rmse)),
            'test_r2_mean': float(np.mean(seeds_test_r2)), 'test_r2_std': float(np.std(seeds_test_r2)),
            'ext_mae_mean': float(np.mean(seeds_ext_mae)), 'ext_mae_std': float(np.std(seeds_ext_mae)),
            'ext_rmse_mean': float(np.mean(seeds_ext_rmse)), 'ext_rmse_std': float(np.std(seeds_ext_rmse)),
            'ext_r2_mean': float(np.mean(seeds_ext_r2)), 'ext_r2_std': float(np.std(seeds_ext_r2)),
            'ext_spearman_mean': float(np.mean(seeds_ext_spearman)), 'ext_spearman_std': float(np.std(seeds_ext_spearman)),
        }
        sal_results.append(row)
        print(f"[{short_name} Dense 5-seed] Test: MAE={row['test_mae_mean']:.3f}+-{row['test_mae_std']:.3f}, RMSE={row['test_rmse_mean']:.3f}+-{row['test_rmse_std']:.3f}, R2log={row['test_r2_mean']:.3f}+-{row['test_r2_std']:.3f} | Ext37k: MAE={row['ext_mae_mean']:.3f}+-{row['ext_mae_std']:.3f}, Spearman={row['ext_spearman_mean']:.3f}+-{row['ext_spearman_std']:.3f}")
        
    return sal_results

def main():
    df_test, df_ext = load_data()
    train_df = D.load_split('train')
    labels = sorted(set(train_df['category'].dropna()))
    
    embeddings_dict = {}
    print('\n=== TIẾN HÀNH KIỂM TRA & NHÚNG POOLED LAST4 CHO 5 ENCODER ===')
    for enc, spec in ENCODER_SPECS.items():
        emb_test, emb_ext = get_or_create_embeddings(df_test, df_ext, enc, spec)
        embeddings_dict[enc] = (emb_test, emb_ext)
        
    mtl_res = eval_mtl(df_test, df_ext, embeddings_dict, labels)
    sal_res = eval_salary_dense(df_test, df_ext, embeddings_dict)
    
    with open(C.ARTIFACT_DIR / 'eval_dense_results_summary.json', 'w') as f:
        json.dump({'mtl': mtl_res, 'salary': sal_res}, f, indent=2, ensure_ascii=False)
        
    print('\n=== TOÀN BỘ TIẾN TRÌNH CHẤM DENSE ĐÃ HOÀN TẤT THÀNH CÔNG ===')

if __name__ == '__main__':
    main()
