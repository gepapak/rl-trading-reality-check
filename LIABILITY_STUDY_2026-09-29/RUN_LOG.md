2026-09-29 20:48:19 full run launched: python run_b.py --phase all --workers 7 (pre-registration hash b5400ed38505f742); smoke outputs (train split) kept in results/jobs_smoke
2026-09-29 21:53:20 runner stopped after 14 finished jobs (rules.csv complete); relaunched with run_b_order.py --workers 7: same jobs and code, NE market first, seed-major (submission order only; run_b.py unchanged)
2026-09-30 04:48:33 all 120 jobs finished; aggregate_b.py run
2026-09-30 05:08:37 Study C (scaling) launched: python scaling/run_c.py --workers 7 (pre-registration hash 127edf72f4ecba9b)
2026-09-30 05:18:56 Study D coding complete (coding_d.csv, 20 candidates, 18 included)
2026-09-30 18:55:56 Study C: all 24 runs finished 18:55; aggregate_c.py written afterwards and run
2026-09-30 19:48:13 Study D coding correction: Gym-Trading-Env recoded at defaults (positions [0,1], unlevered) per protocol; forgiveness noted as configuration-dependent
2026-10-02 08:59:01 Study E launched: python run_e.py --workers 7 (pre-registration hash ce9e00d62d8eb29e); smoke outputs in results_e/jobs_smoke
2026-10-02 15:18:24 Study E: all 120 runs finished; aggregate_e.py run
2026-10-03 10:10:08 Study F launched: python run_f.py --workers 7 (pre-registration hash 9e216c20d87de7f8); pilot results_pilot_f (train split) and smoke results_f/jobs_smoke kept
2026-10-03 10:40:37 Study F: background task stopped by the harness time limit at 10:40 after 7 of 60 runs (in-progress runs lost); relaunched unchanged as an independent process (PID 10332), resume-safe, log results_f/run_full_2.log
2026-10-03 13:15:43 Study F: all 60 runs finished 12:42; aggregate_f.py run
2026-10-03 13:31:10 Study H launched as independent process (PID 7312): python run_h.py --workers 7; pre-registration hashed 11:31Z
2026-10-03 13:47:15 Study G queued (chain_g.py PID 15144): starts run_g.py --workers 7 in the mtsim_study venv when Study H (PID 7312) exits; pre-registration hashed 11:46Z
2026-10-03 15:45:18 Study H: all 40 runs finished; aggregate_h.py run
2026-10-03 21:22:14 Study G: all 20 runs finished 16:03; aggregate_g.py run
2026-10-04 20:34:46 Study I launched as independent process (PID 11744): python run_i.py --workers 7 (90 runs; pre-registration hashed 18:34:38Z); pilot results_pilot_i (train split, 2 rounds) and smoke results_i/jobs_smoke kept
2026-10-04 22:23:09 Study I: all 90 runs finished 22:18 (no failures; hashes verified); aggregate_i.py run: 2/9 supported (I1 A2C-n64, I3 DQN); exploratory explore_i.py, figure make_studyI_outputs.py
2026-10-04 23:22:05 Study J launched as independent process (PID 12508): python run_j.py --workers 7 (60 runs, FI and NO2; pre-registration hashed 21:21:57Z); no pilot; smoke results_j/jobs_smoke kept
2026-10-05 01:59:54 Study J: all 60 runs finished 01:58 (no failures; hashes verified); aggregate_j.py run: 3/5 supported (J2, J3, J4); figure make_studyJ_outputs.py
