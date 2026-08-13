# A6 Pre-Registered Predictions

Written and committed **before** generating any EVAL01 results for bab_el_mandeb,
panama, suez, or malacca (Task 3 of the A6 prompt). The point of this document is
that these predictions are on record prior to seeing the numbers — Task 5's
results document must compare outcomes against this file as written, not against
a version of it edited after the fact.

Every figure cited below is read directly from the committed scenario YAMLs
(`config/benchmark/scenarios/*_P_CRIT.yaml`) or from measurements already
recorded in `docs/multiregion/BENCHMARK_SCHEMA_REFERENCE.md` §6 gap 18 during
A5. Nothing here is invented; where reasoning is genuinely uncertain, that
uncertainty is stated rather than resolved by guessing.

## Source data consulted

| region | P_CRIT active domains that move (non-null effect) | onset_day | ramp_days | duration_days |
|---|---|---|---|---|
| hormuz | shipping, market, geopolitical, routing, news (5) | 240 | 12 | 60 |
| bab_el_mandeb | shipping, market, geopolitical, routing, news (5) | 240 | 12 | 60 |
| panama | shipping, market, routing, news, disaster (5) | 150 | 90 | 200 |
| suez | shipping, market, news (3) | 240 | 1 | 11 |
| malacca | disaster, news (2) | 240 | 15 | 30 |

`lead_days` per domain (positive = leads the physical onset, negative = lags,
0 = contemporaneous):

| region | shipping | market | geopolitical | routing | news | disaster |
|---|---|---|---|---|---|---|
| hormuz | 0 | -5 | +7 | +5 | **+10** | n/a (null) |
| bab_el_mandeb | 0 | -5 | +7 | +5 | **+10** | n/a (absent) |
| panama | 0 | -38 | n/a (absent) | +38 | **+75** | 0 |
| suez | 0 | 0 | n/a (null) | n/a (null) | 0 | n/a (absent) |
| malacca | n/a (null) | n/a (absent) | n/a (null) | n/a (null) | **-3 (lags)** | 0 |

---

## 1. Complementarity ranking

The thesis claim under test: multi-agent fusion outperforms single-domain
methods only where the moving domains carry genuinely independent
information, not merely where many domains happen to move. Domain *count*
and domain *complementarity* are treated as separate axes here deliberately —
Suez is the explicit worked example in this prompt showing they can diverge
(3 domains move, but as near-simultaneous mirrors of one instantaneous
event, so complementarity is low despite non-trivial count).

**Predicted ranking, highest to lowest complementarity: Panama > Bab el-Mandeb
≳ Hormuz > Malacca > Suez.**

- **Panama (highest predicted).** 5 domains move, and `disaster` is a
  mechanistically distinct **leading physical indicator** (Gatun Lake level)
  from the four **lagging economic/behavioral consequences** it drives
  (shipping capacity cut, market auction price, routing diversion, news
  coverage). Critically, `news` leads the physical onset by **75 days** and
  `routing` by 38 — the longest lead structure of any region by a wide
  margin, giving a fusion system real, early, non-redundant information no
  single domain alone would surface until much later. This is the
  textbook case the thesis claim describes.

- **Bab el-Mandeb (high-moderate).** Same 5-domain shape as Hormuz (shipping,
  market, geopolitical, routing, news; same onset/ramp/lead structure,
  deliberately mirrored from Hormuz per `bab_el_mandeb_P_CRIT.yaml`'s own
  header) — but `routing`'s target (0.85) is a **real, measured** diversion
  rate ("the single most precisely quantified domain in the evidence file"),
  not a modeled proxy the way Hormuz's routing figure is (Hormuz has no
  viable sea alternate at all). A genuinely measured behavioral channel is
  more likely to carry real independent information than a proxy. Predicted
  to land at or slightly above Hormuz.

- **Hormuz (moderate — the calibration anchor).** Same 5-domain shape, but
  used here as the known reference point: already-observed ~12% FPR
  improvement from fusion despite touching 5 domains. This is the clearest
  warning against naive domain-counting — geopolitical/routing/shipping/market
  are all downstream reflections of one escalating-tension root cause with a
  comparatively short lead structure (max +10 days, on `news`), so much of
  the apparent 5-domain diversity is correlated, not complementary. Both
  Bab el-Mandeb and Hormuz share this structural ceiling; Hormuz's own
  already-measured result is the anchor the other four regions are being
  predicted relative to, not a new prediction itself.

- **Malacca (low).** Only 2 domains move at all — the sparsest of any
  region. `disaster` (root cause, lead_days=0) and `news` (lead_days=**-3**,
  i.e. it *lags* the physical onset) — meaning **no domain leads the
  physical onset in Malacca's P_CRIT at all**. There is very little for a
  fusion system to triangulate: one leading signal and one lagging,
  low-information echo of it. Predicted low complementarity, though not the
  lowest, since disaster and news are at least mechanistically distinct
  (an environmental measurement vs. a media-attention channel), unlike
  Suez's simultaneous-mirror structure below.

- **Suez (lowest).** Only 3 domains move, and per this prompt's own framing,
  they are near-simultaneous reactions to one zero-warning instantaneous
  blockage — `shipping` is the direct physical consequence, `market` is an
  immediate economic reaction, `news` lags by only ~24-48h (`lead_days=0`
  for every domain). No domain leads the onset at all, and unlike Malacca
  there isn't even a lead-lag structure between the two active non-shipping
  domains — market and news both sit at `lead_days=0`. Suez's `routing`
  classification is DOMINANT at the region level, but (per gap 18, already
  established in A5) that rating is earned by a different, unmodeled event
  entirely. Lowest predicted complementarity of the five.

**Falsification condition:** this ranking is falsified if the measured
multi-agent gain (fused-system FPR/TPR/F1 improvement over the best single
domain baseline) does not track this order — e.g. if Suez shows a *larger*
gain than Hormuz or Bab el-Mandeb, or if Malacca's gain exceeds Panama's, the
domain-count-and-lead-structure reasoning above is wrong or incomplete and
should be reported as such, not rationalized after the fact.

---

## 2. Per-region predicted direction of multi-agent gain

- **Hormuz** — shipping-dominant; already shown ~12% FPR improvement from
  fusion. **Prediction: gain remains small** on this re-run (same scenario
  files, same magnitude). Falsified if the re-measured gain differs
  materially from ~12% under identical methodology (a large swing would
  itself be a finding worth investigating, e.g. an evaluation-pipeline
  change since that number was produced).

- **Suez** — P_CRIT (Ever Given) moves only shipping, market, and news;
  routing and geopolitical are documented-null (the evidence file states
  outright neither rerouting nor a quantifiable geopolitical signal applies
  to a 6-day physical blockage). This makes Suez **low-complementarity
  despite its DOMINANT routing classification**, which derives from a
  different, unmodeled event (the 2023-2024 Red Sea knock-on). **Prediction:
  weak multi-agent gain, comparable to or smaller than Hormuz's.** A
  Hormuz-like weak gain here would **support**, not undermine, the thesis
  claim — it is the predicted outcome for a structurally low-complementarity
  region, not a failure of fusion. Falsified if Suez shows a *strong*
  fusion gain despite this domain-sparse, no-lead structure — that would
  mean complementarity is not actually gating fusion benefit the way the
  thesis claims, since Suez has the least basis for it structurally.

- **Bab el-Mandeb** — predicted **moderate gain, at or slightly above
  Hormuz's**, per the complementarity reasoning above (same domain shape as
  Hormuz, but a genuinely measured, non-proxy routing signal). Falsified if
  Bab el-Mandeb's gain is smaller than Hormuz's, or if it matches Suez's low
  end — either would mean the proxy-vs-measured routing distinction doesn't
  actually matter for fusion performance.

- **Panama** — predicted **largest multi-agent gain of the five regions**,
  per the leading-disaster-indicator reasoning above (75-day news lead,
  38-day routing lead, mechanistically distinct leading/lagging domain
  structure). Falsified if Panama's gain is not the largest, or is
  comparable to Suez's/Malacca's — that would mean long lead times and
  leading/lagging domain diversity don't translate into fusion advantage,
  a direct hit against the thesis's central mechanism claim.

- **Malacca** — predicted **small gain**, likely the second-smallest after
  Suez, given only 2 active domains and no domain that leads the physical
  onset (`news` lags by 3 days; `disaster` is contemporaneous). Falsified
  if Malacca's gain is comparable to Panama's or Bab el-Mandeb's — that
  would mean domain *count* and lead structure matter less than predicted,
  since Malacca has the sparsest domain participation of any region.

---

## 3. Lead-time ranking

**Predicted ranking, best (longest usable lead) to worst: Panama > Hormuz ≈
Bab el-Mandeb > Malacca ≈ Suez (tied for worst, genuinely uncertain which is
lower).**

- **Panama best** — `news` leads onset by 75 days, `routing` by 38; by far
  the largest structural lead of any region, plus the longest ramp (90
  days) giving the most time for a rising signal to cross a threshold
  before the labeled onset.
- **Hormuz ≈ Bab el-Mandeb, moderate** — identical lead structure (max +10
  days on `news`), identical 12-day ramp.
- **Suez worst by design** — every active domain has `lead_days=0` and
  `ramp_days=1`; the evidence file itself states this event has
  "effectively zero warning time." No structural lead exists to detect.
- **Malacca, genuinely uncertain vs. Suez** — `disaster` is contemporaneous
  (`lead_days=0`, same as Suez's domains) and `news` actually **lags** the
  onset by 3 days. Malacca therefore also has **zero domains that lead the
  physical onset** — structurally not better than Suez on this metric, and
  arguably worse on the one domain that does move non-contemporaneously
  (lagging rather than merely simultaneous). This is stated as a genuine,
  unresolved prediction rather than forced into an order: it is plausible
  Malacca measures *worse* than Suez on lead time despite its longer
  (15-day) ramp, because ramp length alone doesn't create lead time without
  a domain that starts moving before the labeled onset.

**Falsification condition:** falsified if Panama is not best, if Suez is not
at or near the worst end, or if Malacca measures clearly better than Suez —
the last of these wouldn't falsify the broader lead-time mechanism (a longer
ramp could still help even without positive `lead_days`, which is itself a
useful thing to learn), but it would falsify the specific "Malacca ≈ Suez"
call made here, and should be reported as exactly that rather than absorbed
silently into "the ranking roughly worked."

---

## 4. False-positive-rate (FPR) ranking

**Predicted ranking, best (lowest FPR) to worst (highest FPR): Malacca >
{Hormuz, Bab el-Mandeb, Panama} (clustered) > Suez (worst).**

- **Suez worst** — its only decoy domain (`market`) is the same domain its
  true P_CRIT event moves most: decoy peak `5.7947` sits within ~3% of
  P_CRIT's own peak (`5.9869`), already measured and recorded in gap 18.
  A detector tuned to catch the real event on `market` is structurally
  likely to also fire on the decoy.
- **Hormuz, Bab el-Mandeb, Panama — clustered in the middle.** All three
  use a `news` decoy sized via the same σ/ratio conventions, and all three
  cleared gap 18's matched-window check with a comfortable margin against
  their own P_CRIT (unlike Suez's ~3% margin) — but `news` is still an
  *active* mover in all three regions' own P_CRIT, so some decoy/positive
  confusability exists, just far less acute than Suez's.
- **Malacca best** — its decoy domain (`geopolitical`) is one `P_CRIT`
  **never touches at all** (`effect: null`, confirmed in A5) — perfect
  domain orthogonality between the decoy and the true positive. Its decoy
  also stays **sub-threshold** against the real `geopolitical` agent's own
  gate (peak `0.4259` vs. threshold `0.50`, per gap 18's threshold table),
  making it the hardest of the five decoys to confuse with a real event.

**Falsification condition:** falsified if Suez's FPR is not the worst of
the five, if Malacca's is not the best, or if the three middle regions are
not clustered closer to Malacca than to Suez. A specific sharper failure
mode to watch for: if Malacca's FPR is *not* the best despite perfect decoy
orthogonality, that would mean orthogonality on paper doesn't translate to
detector-level discrimination — worth reporting explicitly rather than
folding into "roughly as predicted."
