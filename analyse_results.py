import glob, json, os
from operator import itemgetter

IN_DIR = "/content/drive/MyDrive/deepcompress_ckpt1/sweep_out"

def load_results(indir=IN_DIR):
    files = sorted(glob.glob(os.path.join(indir, "*_result.json")))
    results = []
    for f in files:
        try:
            j = json.load(open(f))
            results.append(j)
        except Exception as e:
            print("Skipping", f, ":", e)
    return results

def format_bytes(b):
    return f"{b/1e6:.3f} MB" if b is not None else "N/A"

def compute_best(results, tol=1.0):
    # tol = allowed absolute accuracy drop in percentage points
    best_model = None
    best_weights = None
    best_acts = None
    for r in results:
        base = r.get('acc_baseline', 0.0)
        acc = r.get('acc_compressed', 0.0)
        if acc >= base - tol:
            # model ratio (huffman if available)
            mr = r.get('ratio_model_huffman', r.get('ratio_model_raw', None))
            wr = r.get('ratio_weights_raw', None)
            ar = r.get('ratio_activations', None)
            if mr is not None:
                if best_model is None or mr < best_model['ratio']:
                    best_model = {'ratio': mr, 'run': r}
            if wr is not None:
                if best_weights is None or wr < best_weights['ratio']:
                    best_weights = {'ratio': wr, 'run': r}
            if ar is not None:
                if best_acts is None or ar < best_acts['ratio']:
                    best_acts = {'ratio': ar, 'run': r}
    return best_model, best_weights, best_acts

def summarize(results, tol=1.0):
    total = len(results)
    print(f"Loaded {total} runs.")
    # Basic summary: print top 5 by compression ratio (huffman) sorted ascending
    sorted_by_model = sorted([r for r in results if r.get('ratio_model_huffman')], key=lambda x: x['ratio_model_huffman'])
    print("\nTop 5 runs by model compression ratio (Huffman):")
    for r in sorted_by_model[:5]:
        print(f"  {r['name']}: ratio={r['ratio_model_huffman']:.4f}, acc={r['acc_compressed']:.2f}, final_MB={r['final_MB_huffman']:.3f}")

    best_model, best_weights, best_acts = compute_best(results, tol=tol)
    out = {
        'total_runs': total,
        'accuracy_tolerance': tol,
        'best_model': None,
        'best_weights': None,
        'best_activations': None
    }
    if best_model:
        r = best_model['run']
        out['best_model'] = {
            'name': r['name'],
            'ratio_model_huffman': best_model['ratio'],
            'final_MB_huffman': r.get('final_MB_huffman'),
            'acc_compressed': r.get('acc_compressed'),
            'acc_baseline': r.get('acc_baseline')
        }
    if best_weights:
        r = best_weights['run']
        out['best_weights'] = {
            'name': r['name'],
            'ratio_weights_raw': best_weights['ratio'],
            'final_MB_huffman': r.get('final_MB_huffman'),
            'acc_compressed': r.get('acc_compressed'),
            'acc_baseline': r.get('acc_baseline')
        }
    if best_acts:
        r = best_acts['run']
        out['best_activations'] = {
            'name': r['name'],
            'ratio_activations': best_acts['ratio'],
            'act_fp32_bytes_per_sample': r.get('act_fp32_bytes_per_sample'),
            'act_quant_bytes_per_sample': r.get('act_quant_bytes_per_sample'),
            'acc_compressed': r.get('acc_compressed'),
            'acc_baseline': r.get('acc_baseline')
        }
    # Also compute absolute best model ratio regardless of accuracy (for reference)
    all_with_mr = [r for r in results if r.get('ratio_model_huffman') is not None]
    if all_with_mr:
        best_overall = min(all_with_mr, key=lambda x: x['ratio_model_huffman'])
        out['best_overall_model'] = {'name': best_overall['name'], 'ratio_model_huffman': best_overall['ratio_model_huffman'], 'acc_compressed': best_overall['acc_compressed']}
    # write summary
    with open('analysis_summary.json','w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved analysis_summary.json")
    return out

if __name__ == '__main__':
    res = load_results()
    summary = summarize(res, tol=1.0)
    print("\nSummary (printed):")
    import pprint; pprint.pprint(summary)
