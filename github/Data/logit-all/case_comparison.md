# Logit-All 12-Case Summary

## Case Order (script execution order)

| # | Mode | Sub | Tag | baseline | serve_k | init_k | out_tok/s | total_tok/s | mean_ttft_ms | draft_acc | sys_eff | align_accept | adp_switch/min |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | basedraft | k2 | A0_k2 | A0 | 2 |  | 408.36 | 806.33 | 231.07 | 0.770 | 0.799 | 0.8872 |  |
| 2 | basedraft | k4 | A0_k4 | A0 | 4 |  | 309.06 | 610.26 | 366.26 | 0.749 | 0.655 | 0.7151 |  |
| 3 | basedraft | k8 | A0_k8 | A0 | 8 |  | 214.57 | 423.68 | 996.72 | 0.724 | 0.452 | 0.4865 |  |
| 4 | align_only | k2 | A2_k2 | A2 | 2 |  | 395.06 | 780.07 | 259.47 | 0.775 | 0.803 | 0.8885 |  |
| 5 | align_only | k4 | A2_k4 | A2 | 4 |  | 342.10 | 675.51 | 324.76 | 0.753 | 0.659 | 0.7203 |  |
| 6 | align_only | k8 | A2_k8 | A2 | 8 |  | 221.15 | 436.68 | 583.34 | 0.716 | 0.444 | 0.4794 |  |
| 7 | adaptive_only | init_k2 | A1_k8 | A1 | 8 | 2 | 380.40 | 751.42 | 266.41 | 0.764 | 0.675 | 0.8169 | 8.3276 |
| 8 | adaptive_only | init_k4 | A1_k8 | A1 | 8 | 4 | 388.45 | 767.02 | 254.72 | 0.761 | 0.719 | 0.8401 | 9.2749 |
| 9 | adaptive_only | init_k8 | A1_k8 | A1 | 8 | 8 | 356.34 | 703.61 | 386.02 | 0.738 | 0.562 | 0.6996 | 9.1587 |
| 10 | joint | init_k2 | A2_k8 | A2 | 8 | 2 | 394.03 | 778.04 | 258.91 | 0.771 | 0.745 | 0.8567 | 8.0183 |
| 11 | joint | init_k4 | A2_k8 | A2 | 8 | 4 | 392.57 | 775.15 | 248.53 | 0.770 | 0.744 | 0.8561 | 9.4633 |
| 12 | joint | init_k8 | A2_k8 | A2 | 8 | 8 | 380.29 | 750.91 | 516.77 | 0.739 | 0.546 | 0.6795 | 10.3455 |

## Horizontal Compare: basedraft vs align_only (same k)

| k | basedraft out_tok/s | align_only out_tok/s | delta | basedraft TTFT | align_only TTFT | delta | basedraft draft_acc | align_only draft_acc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 408.36 | 395.06 | -13.3 | 231.07 | 259.47 | 28.4 | 0.770 | 0.775 |
| 4 | 309.06 | 342.10 | 33.04 | 366.26 | 324.76 | -41.5 | 0.749 | 0.753 |
| 8 | 214.57 | 221.15 | 6.58 | 996.72 | 583.34 | -413.38 | 0.724 | 0.716 |

## Horizontal Compare: adaptive_only vs joint (same init_k)

| init_k | adaptive_only out_tok/s | joint out_tok/s | delta | adaptive_only TTFT | joint TTFT | delta | adaptive_only draft_acc | joint draft_acc | adaptive_only switch/min | joint switch/min |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 380.40 | 394.03 | 13.63 | 266.41 | 258.91 | -7.5 | 0.764 | 0.771 | 8.3276 | 8.0183 |
| 4 | 388.45 | 392.57 | 4.12 | 254.72 | 248.53 | -6.19 | 0.761 | 0.770 | 9.2749 | 9.4633 |
| 8 | 356.34 | 380.29 | 23.95 | 386.02 | 516.77 | 130.75 | 0.738 | 0.739 | 9.1587 | 10.3455 |

