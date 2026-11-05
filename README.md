# MuleGuard - explainable mule-account investigation

MuleGuard looks at a bank's transaction history and answers one question an investigator actually asks:

> **Rs 5,00,000 just arrived in this account and left within two minutes. Where did it come from, where did it go,
> are these accounts mules, and should we freeze them?**

It finds suspicious accounts, traces the money, retrieves similar past cases, and writes an explainable
investigation report with a risk level. A human approves any freeze.

> **This is a prototype on a synthetic dataset.** No real bank data was used or is accessible. The results show that
> the system finds the patterns that were designed into the data. They are not real-world accuracy.
> See [Limitations](#limitations-read-this-before-quoting-any-number).

---

## 1. What it does

| Capability | How |
|---|---|
| Find mule-like accounts | XGBoost on SQL behaviour features + transaction-graph features |
| Catch unusual behaviour without labels | PyTorch autoencoder (reconstruction error -> anomaly percentile) |
| Trace the money | Recursive SQL, forward (where it went) and backward (where it came from) |
| Find similar past cases | 16-number behaviour embedding in pgvector, cosine nearest neighbours (HNSW index) |
| Explain the decision | Agent: code gathers evidence -> fixed rules set the risk level -> LLM writes the narrative -> every number is verified |
| Act safely | Agent only *recommends*. Freeze needs explicit human confirmation and is written to an audit log |
| Use it | FastAPI service + Streamlit dashboard |

---

## 2. Architecture

```
IBM AMLworld data -> prepare + inject mules/decoys -> PostgreSQL (+pgvector)
                                                          |
              +-------------------+-----------------------+------------------+
              v                   v                       v                  v
      SQL + graph features   fund trace (recursive    similar cases      anomaly model
      -> XGBoost score       SQL, fwd + back)         (pgvector)         (autoencoder)
              |                   |                       |                  |
              +-------------------+-----------+-----------+------------------+
                                              v
                              Investigation agent (src/agent.py)
                   evidence -> RULES decide risk -> LLM writes text -> numbers verified
                                              v
                               FastAPI  ->  Streamlit dashboard
                                              v
                      human confirms freeze -> accounts.status + audit_log
```

**Design rule:** the risk level and freeze candidates come from fixed rules in `assess()`, never from the LLM.
The LLM only writes the narrative. Every number in its text is checked against the evidence, and if it fails the
check (or the LLM is unavailable) a plain template report is used instead.

---

## 3. The dataset

Base data is **IBM AMLworld HI-Small** (`HI-Small_Trans.csv` + `HI-Small_Patterns.txt`, from Kaggle
`ealtman2019/ibm-transactions-for-anti-money-laundering-aml`). It is transformed as follows:

- Cleaned: same-currency transfers kept, self-transfers ("Reinvestment") removed, string account IDs.
- Amounts are IBM's values **relabelled** as INR (not converted).
- Random seconds/milliseconds added (the original has minute-level timestamps).
- Payment formats mapped to Indian channels: NEFT, RTGS, IMPS, UPI, CARD, CHEQUE, CASH (cosmetic).
- **Injected mule scenarios**: Rs 3-8 lakh arrives and 90-99% leaves. 85% forward within 20-120 s, the rest within 2-10 min.
  - `single_pass` (30%): one recipient. `fan_out` (30%): 3-10 recipients. `two_hop` (40%): recipients forward again 1-5 min later.
- **Injected decoys** (legitimate look-alikes): `biz_supplier`, `aggregator`, `one_off`, `trusted_forwarder`.
- Injected start times follow the base traffic distribution, and injected channels are sampled from the base mix so the
  channel column does not give the answer away.

| Fact | Value |
|---|---|
| Transactions in database | 4,512,323 |
| Accounts scored (train + val + test) | 987,449 |
| Split | by time (traffic quantiles 70 / 15 / 15), never random |
| Train | 412,054 accounts, 2,836 mules (0.688%) |
| Validation | 262,464 accounts, 782 mules (0.298%) |
| Test | 312,931 accounts, 1,013 mules (0.324%) |
| Test mules by source | 449 injected, 564 from IBM's own patterns |
| Test decoys | 729 |

---

## 4. Results (test window, all from a real run)

Threshold and early stopping were chosen on the validation window. The test window was used once.

### 4.1 Supervised model (XGBoost)

| Model | Features | PR-AUC | ROC-AUC | Precision | Recall | precision@100 | @500 | @1000 |
|---|---|---|---|---|---|---|---|---|
| Tabular (SQL features) | 26 | 0.5588 | 0.9461 | 1.000 | 0.4314 | 1.000 | 0.914 | 0.535 |
| **Graph** (main model) | 51 | **0.6072** | 0.9484 | 1.000 | 0.4304 | 1.000 | 0.970 | 0.579 |
| Graph, no invented columns | 48 | 0.6120 | 0.9484 | 1.000 | 0.4294 | 1.000 | 0.968 | 0.584 |
| Baseline rule `n_fast_pass >= 1` | - | - | - | 0.813 | 0.2833 | - | - | - |

- Graph features add +0.048 PR-AUC over tabular.
- Removing the invented columns (`account_age_days`, `kyc_level`, `account_type`) does not hurt (gap -0.0048), so they are not carrying the result.
- The model beats the simple rule on both precision (1.000 vs 0.813) and recall.

### 4.2 Who gets flagged (main model, threshold 0.981)

| Group | Accounts | Flagged | Share |
|---|---|---|---|
| Injected mules | 449 | 436 | 97.1% |
| IBM-pattern mules | 564 | 0 | 0.0% |
| Decoys (legit look-alikes) | 729 | 0 | 0.0% |
| Everyone else | 311,189 | 0 | 0.0% |

Recall by typology: `two_hop` 96.9% (312/322), `single_pass` 96.2% (50/52), `fan_out` 80.4% (74/92).
All IBM-pattern typologies (`scatter_gather`, `bipartite`, `cycle`, `stack`, `random`, `gather_scatter`, `fan_in`) are 0%.

### 4.3 Anomaly model (autoencoder, used on its own)

Trained on 409,218 normal train-window accounts, 23 features. ROC-AUC 0.8329, PR-AUC 0.2870 (random would be 0.0032).

| Group | Accounts | In top 1% | In top 5% |
|---|---|---|---|
| Injected mules | 449 | 94.9% | 97.8% |
| IBM-pattern mules | 564 | 2.7% | 16.1% |
| Decoys | 729 | 10.8% | 25.2% |
| Everyone else | 311,189 | 0.4% | 4.1% |

Of the 577 mules the supervised model missed, 94 (16.3%) are in the anomaly top 5% and 18 in the top 1%.
Used alone it would put 1,373 ordinary accounts in the top 1%, which is too noisy. It is therefore used as a
**supporting signal** in the agent's rules, not as a detector on its own.

### 4.4 Similar-case retrieval (pgvector)

3,829 past cases indexed (2,836 mules, 993 decoys) from the train window. On test accounts, the majority of the
top-5 neighbours matches the true class for 82.4% of mules (835/1,013) and 76.1% of decoys (555/729). For injected
mules, the neighbours' top typology equals the true typology 80.4% of the time (361/449).

### 4.5 End-to-end examples (agent, `--no-llm`)

| Account | Ground truth | Risk | ML score | Anomaly pct | Evidence |
|---|---|---|---|---|---|
| 015_803F7F5A0 | single_pass | **HIGH** | 0.9898 | 0.999 | Rs 746,433 98% forwarded, first outflow 103.6 s |
| 0342584_80F9519C0 | fan_out | **HIGH** | 0.991 | 0.999 | Rs 671,030 97% forwarded, first outflow 40.6 s |

Both reports list freeze candidates for human approval and the three most similar past cases (all confirmed mules).

---

## 5. Limitations 

1. **Synthetic data.** Injected-mule results (97% flagged, zero false positives) are a best case: the system recovers
   patterns that were designed into the data. Real accuracy is unknown.
2. **IBM-pattern mules are not detected (0% of 564).** They are slow, multi-day structures that look like ordinary accounts
   at account level. A diagnostic (`src/diagnose_ibm.py`) confirmed the current tables do not contain the signal
   (a model trained only on them reached PR-AUC 0.0226 against 0.0013 random). This was accepted rather than fitted to.
   Overall recall is therefore capped near 43%.
3. **Amount features dominate.** `max_in` and `max_out` are the top features. Injected mules all sit in a Rs 3-8 lakh band,
   so part of the score may be an artefact of that band. An ablation without raw amounts has not been run.
4. **Invented columns.** `opened_at`, `kyc_level`, `account_type` are generated. The ablation shows they do not drive results.
5. **Precision of 1.0 with zero false positives** reflects 729 designed decoys and a synthetic background, not production conditions.
6. **Similar-case "mule share" is inflated by the prior** (mules are over-represented in the case index). It is shown as context, not as evidence.
7. **The LLM narrative** (Groq, `llama-3.3-70b-versatile`) is implemented with number verification and a template fallback.
   The reports above use the template path.
8. Short time window; all splits are by time.

---

## 6. Techniques in one line each

- **XGBoost**: boosted decision trees; `scale_pos_weight` handles the 1-in-1,000 imbalance; early stopping on validation PR-AUC.
- **Graph features**: cycles, diamonds, shared counterparties, connected components, neighbour aggregates (scipy sparse; hubs excluded).
- **Recursive SQL trace**: follows money hop by hop, 30-minute hop window, no amount growth, no cycles, minimum 5% of the seed amount; backward lookback 24 h.
- **Autoencoder**: compresses and rebuilds behaviour; high reconstruction error means unlike normal accounts.
- **pgvector**: 16-dimension z-scored behaviour embedding, cosine distance, HNSW index.
- **Agent**: evidence from code, risk from rules, text from the LLM, number verification, template fallback.

## Data and licence notes

Source data: IBM AMLworld (Altman et al.), via Kaggle; check the dataset's licence before redistributing. All injected cases,
amounts relabelled as INR, and derived account attributes are synthetic.
