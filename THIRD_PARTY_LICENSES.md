# Third-party material and licenses

Checked on 29 Sep 2026; updated 5 Oct 2026 for Studies A–J.

| Material | Location in repository | Source | License / terms | How it is used here |
|---|---|---|---|---|
| Imbalance prices DK1 / DK2, 2025-03-04 → 2026-08-17 | `data/energinet/` | Energi Data Service, dataset `ImbalancePrice` | **CC BY 4.0**. Terms: https://www.energidataservice.dk/terms-and-conditions | **Redistributed.** Period subset of the published dataset; values unchanged; gzip-compressed. |
| Balancing activation volumes | `data/engine_liquidity/` | Derived from Energi Data Service datasets `RegulatingBalancePowerdata` and `ImbalancePrice` | **CC BY 4.0** (as above) | **Redistributed, modified.** Observed balancing-activation energy, summed up and down: mFRR MWh from `RegulatingBalancePowerdata`, and after its 2025 transition, aFRR MW from `ImbalancePrice` converted to MWh. Aligned onto the 10-minute 2025 evaluation grid; missing values set to 0. |
| Imbalance prices (EXP14) and imbalance volumes (EXP13), FI and NO1–NO5 | **not included**; fetched into `data/esett/` by `data/fetch_esett.py` | eSett Open Data, https://opendata.esett.com | Terms of use (https://opendata.esett.com/terms): public, no authorization required, provided "as is". **No explicit redistribution license.** | **Not redistributed.** Only the query manifests and verification hashes are included. |
| Imbalance prices, production and consumption settlement, and day-ahead wind forecasts, DK1 / DK2, 2025-03-04 → 2026-08-17 | `GENERALIZATION_STUDY_2026-09-29/data_energinet/` | Energi Data Service, datasets `ImbalancePrice`, `ProductionConsumptionSettlement`, `Forecasts_Hour` | **CC BY 4.0** (as above) | **Redistributed.** Period subsets; values unchanged; gzip-compressed. |
| 15-minute panels for DK1 / DK2 (imbalance and day-ahead price, balancing volume, scaled wind) | `GENERALIZATION_STUDY_2026-09-29/data_panel/` | Derived by the authors from the Energinet files above (`build_panel.py`) | **CC BY 4.0** (as above) | **Redistributed, modified.** Used by Studies A–I. |
| Imbalance prices (EXP14) and volumes (EXP13), FI and NO2, for Study J | **not included**; fetched by `data/fetch_esett.py`, placed by `LIABILITY_STUDY_2026-09-29/get_esett_j.py`, and turned into panels by `build_panel_j.py` | eSett Open Data | Terms of use as above | **Not redistributed**, nor are the panels derived from them. The re-download is verified against the study's content hashes. |
| Derived tail statistics per bidding zone | `data/derived/cross_market_structure.csv` | Computed by the authors from eSett EXP14 prices | Authors' derived statistics (CC BY 4.0, `LICENSE-docs.md`); the underlying data follows eSett's terms | Aggregate statistics only; no raw eSett records. |
| GitHub repository metadata (names, star counts, short descriptions, commit SHAs) | `code_audit/candidates*.csv`, `code_audit/stage2_repo_summary.csv` | GitHub REST API (public metadata) | Descriptions belong to the repository owners | Research documentation of the search and screening. **No third-party source code is included.** The verbatim excerpts used during coding were excluded and can be regenerated with `code_audit/scan_repos.py` from the pinned commits. |
| Bibliographic data (arXiv identifiers, short title snippets, keyword counts) | `literature/` | Public paper metadata | Belongs to the respective publishers / authors | Research documentation. **The paper PDFs and full text are not included.** |
| numpy, pandas, scipy, pypdf; engine stack (PyTorch, TensorFlow, Stable-Baselines3, Gymnasium, PettingZoo, scikit-learn, ...) | not bundled (`requirements*.txt`) | PyPI | Permissive open-source licenses (BSD, Apache-2.0, MIT) | Runtime dependencies only. |
| gym-mtsim 2.0.0 (AminHP) | not bundled (`requirements-mtsim.txt`) | PyPI / https://github.com/AminHP/gym-mtsim | MIT (repository license) | Runtime dependency of Study G. `mtsim_study/mtsim_common.py` subclasses its simulator to remove the balance floor; no gym-mtsim code is copied. |
| FinRL, TensorTrade and the 20 screened trading environments | not included | Public GitHub repositories | Their own licenses | Read at pinned commits for the audits (`LIABILITY_STUDY_2026-09-29/env_audit/`); only coding and file-and-line references are included. |
| Multi-agent engine source | `engine/` | The authors' own code | MIT (`LICENSE`) | Included: the patched audit copy. Local absolute paths in the provenance manifests are replaced by `<project_root>`. Datasets and checkpoints are not included. |

## Required attribution for the Energinet data
> Source: Energinet (www.energidataservice.dk). Data subset and, for `data/engine_liquidity/`, aggregated and resampled by the authors. Energinet does not endorse this work.

## eSett data
Run `python data/fetch_esett.py` before `analysis/cross_market_liquidity_test.py`. The script downloads exactly the queries used in the study (2025-03-04 → 2026-09-28) and checks every EXP14 API response against the SHA-256 recorded at download time, and each EXP13 file against its recorded content hash. It reports any file eSett has revised since.

**Re-download test (29 Sep 2026).**
- The EXP13 volume files are identical to the study files once cut at the study's last timestamp: 65–66 rows published since the study are dropped.
- In each zone, eSett corrected 1–2 EXP14 price values on 2026-09-25. These fall outside the analysed window: joined with the EXP13 volumes, the sample ends 2026-09-15.
- `analysis/cross_market_liquidity_test.py` run on the re-downloaded data reproduces `results/cross_market/` exactly (maximum difference 0.0 in every column).
