# Profiling gate (spec section 11)

### lurestar

| metric | lurestar-gpt | lurestar-nextlat |
|---|---|---|
| median s/step (compute) | 0.1006 | 0.1154 |
| p95 s/step (compute) | 0.1009 | 0.1156 |
| wall s/step | 0.2351 | 0.2516 |
| examples/s | 5089.6 | 4438.2 |
| tokens/s | 351185 | 306237 |
| peak allocated VRAM (GB) | 7.45 | 8.59 |
| peak reserved VRAM (GB) | 7.60 | 9.23 |
| VRAM headroom | 80.8% | 76.6% |
| physical batch fits | yes | yes |
| GPU util % (median) | 77.0 | 77.0 |
| host-input wait (s) | 12.90 | 12.90 |
| host-input wait (frac wall) | 11.0% | 10.3% |
| checkpoint writes | 5 | 5 |
| checkpoint write (s, median) | 0.60 | 0.64 |
| checkpoint size (MB) | 244.1 | 250.9 |
| steps summarized | 399 | 399 |
| val_(5, 5)/test_accuracy @ step 0 (lurestar-gpt) | 0.0000 | - |
| val_(5, 5)/test_accuracy @ step 0 (lurestar-nextlat) | - | 0.0000 |
| UNMEASURED (spec sec.11) | none | none |

NextLat vs GPT: nextlat_step_time_overhead = 1.147x, nextlat_throughput_ratio = 0.872x, nextlat_peak_allocated_overhead = 1.153x, nextlat_peak_reserved_overhead = 1.215x, nextlat_checkpoint_bytes_overhead = 1.028x

### hmm

| metric | hmm-gpt | hmm-nextlat |
|---|---|---|
| median s/step (compute) | 0.0127 | 0.0163 |
| p95 s/step (compute) | 0.0142 | 0.0182 |
| wall s/step | 0.0923 | 0.0965 |
| examples/s | 20161.6 | 15704.6 |
| tokens/s | 1310505 | 1020801 |
| peak allocated VRAM (GB) | 0.24 | 0.27 |
| peak reserved VRAM (GB) | 0.28 | 0.34 |
| VRAM headroom | 99.3% | 99.1% |
| physical batch fits | yes | yes |
| GPU util % (median) | 0.0 | 0.0 |
| host-input wait (s) | 2.38 | 2.36 |
| host-input wait (frac wall) | 8.6% | 8.2% |
| checkpoint writes | 5 | 5 |
| checkpoint write (s, median) | 0.03 | 0.03 |
| checkpoint size (MB) | 9.8 | 10.6 |
| steps summarized | 239 | 239 |
| UNMEASURED (spec sec.11) | none | none |

NextLat vs GPT: nextlat_step_time_overhead = 1.284x, nextlat_throughput_ratio = 0.779x, nextlat_peak_allocated_overhead = 1.141x, nextlat_peak_reserved_overhead = 1.227x, nextlat_checkpoint_bytes_overhead = 1.077x

### projected end-to-end runtime

- lurestar: 66.01 GPU-h (bst 59.87, gpt 2.86, nextlat 3.28)
- adapt: 3.32 GPU-h (bst 3.00, gpt 0.15, nextlat 0.17)
- hmm: 0.12 GPU-h (gpt 0.05, nextlat 0.07)
- subtotal: 69.45 GPU-h
- with 20% interruption margin: 83.34 GPU-h
