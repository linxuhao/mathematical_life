def run_single_union(npy_a, npy_b, eps_pred_ref, independent_eps=False):
    """Run one union test. 
    If independent_eps=True: each cloud uses its OWN eps_pred (for cross-model unions).
    If independent_eps=False: all use ref_eps from A (for same-model prompt-type unions).
    """
    t0 = time.time()
    
    if independent_eps:
        # Each cloud independently
        points_a, ba, eps_a, eps_pred_a = load_and_betti(npy_a, None)
        if points_a is None:
            return {'error': 'NaN in A'}
        points_b, bb, eps_b, eps_pred_b = load_and_betti(npy_b, None)
        if points_b is None:
            return {'error': 'NaN in B'}
        points_u = np.concatenate([points_a, points_b], axis=0)
        _, bu, eps_u, eps_pred_u = load_and_betti_temp(points_u, None)
        
        eps_pred_ref_used = f"indep: A={eps_pred_a:.4f} B={eps_pred_b:.4f} U={eps_pred_u:.4f}"
    else:
        # Reference eps from A
        if eps_pred_ref is None:
            eps_pred_ref = compute_eps_pred_ref(npy_a)
        points_a, ba, eps_a, _ = load_and_betti(npy_a, eps_pred_ref)
        if points_a is None:
            return {'error': 'NaN in A'}
        points_b, bb, eps_b, _ = load_and_betti(npy_b, eps_pred_ref)
        if points_b is None:
            return {'error': 'NaN in B'}
        points_u = np.concatenate([points_a, points_b], axis=0)
        os.environ.setdefault('OMP_NUM_THREADS', '4')
        from ripser import ripser
        dgms_u = ripser(points_u, maxdim=1)['dgms']
        bu = sum(1 for d in dgms_u[1] if (d[1] - d[0]) > eps_pred_ref)
        eps_pred_ref_used = round(eps_pred_ref, 4)
    
    intersection = ba + bb - bu
    overlap_pct = round(intersection / max(ba, bb) * 100, 1) if max(ba, bb) > 0 else 0
    verdict = union_verdict(ba, bb, bu)
    dt = time.time() - t0
    
    result = {
        'A_beta1': ba, 'B_beta1': bb, 'U_beta1': bu,
        'sum': ba + bb, 'intersection': intersection,
        'overlap_pct': overlap_pct, 'verdict': verdict,
        'eps_pred': eps_pred_ref_used,
        'n_A': points_a.shape[0], 'n_B': points_b.shape[0], 'n_U': points_u.shape[0],
        'time_s': round(dt, 1),
    }
    return result
