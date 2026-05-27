# How well can composition alone predict battery voltage?
### …and how I almost fooled myself with the wrong train/test split

In the previous post, I built a cleaned dataset of 2,634 Li insertion-electrode cathodes from the Materials Project, filtering obvious computational artifacts while preserving chemically meaningful variation. This post is about the first machine-learning baseline — and more importantly, about how easy it is to overestimate model performance if the evaluation setup is careless.

At first, the model looked surprisingly good. Then I checked the train/test split more carefully, and realized I was almost measuring chemical similarity instead of real generalization.

---

## From formulas to features

A model cannot learn directly from a string like `"LiCoO2"`. It needs a fixed-length numerical representation of the chemistry.

To do that, I used the `ElementProperty` featurizer from `matminer`, configured with the Magpie preset. For each composition, Magpie computes stoichiometry-weighted statistics over elemental properties such as:

- electronegativity
- atomic radius
- valence electron counts
- melting temperature
- covalent radius
- magnetic moment
- periodic-table position

For every property, it calculates descriptors like:

- mean
- maximum
- minimum
- range
- average deviation

This converts a variable-size chemical composition into a fixed-length numerical vector.

I featurized the **`framework_formula`** — the host structure without Li — rather than the charged or discharged compositions.

That choice was deliberate.

Voltage is primarily governed by the host framework's redox chemistry: the transition metal, its electronic environment, and the surrounding anion network. Since every row in the dataset already uses Li as the working ion, explicitly including Li adds a nearly constant, low-information feature across the dataset.

Using the discharged composition also risks a softer form of leakage: the amount of inserted Li is mechanically tied to the target voltage through the underlying electrochemistry. The framework formula is therefore the cleanest representation of the question:

> “What host chemistry gives rise to this voltage?”

Result:

- **132 Magpie features**
- **0 missing values**
- across all **2,634 materials**

One small quirk I noticed: a handful of single-element frameworks naturally produced zero-valued spread statistics, since quantities like range and deviation collapse when only one element is present. But these represented less than 1% of the dataset and caused no issues.

---

## A baseline that beats “guess the average”

Before evaluating any ML model, I established the baseline floor.

If a model simply predicts the average voltage for every material, how wrong is it?

| model | MAE |
|-------|-----|
| Dummy regressor (predict mean) | 0.880 V |
| LightGBM (single 80/20 split) | 0.486 V |

The LightGBM model reduced the mean absolute error by almost half relative to the dummy baseline, showing that composition alone carries substantial predictive signal.

Additional metrics:

| metric | value |
|--------|------|
| RMSE | 0.697 V |
| R² | 0.601 |

The fact that RMSE is noticeably larger than MAE suggests the model still produces a tail of larger prediction errors on some materials.

At this point, the model looked genuinely promising.

Then I evaluated it properly.

---

## The part where I almost fooled myself

A single train/test split is fragile. Performance can depend heavily on one lucky partition.

So I moved to 5-fold cross-validation.

The first result was confusing:

| evaluation | MAE |
|------------|-----|
| 5-fold CV (default, unshuffled) | 0.590 ± 0.110 V |

The error became both larger *and* much more variable.

That variability was the important clue.

The issue turned out not to be chemistry — it was data ordering.

Scikit-learn’s default `KFold` does **not** shuffle data before splitting. My dataset happened to be ordered by composition complexity: single-element frameworks first, then binaries, ternaries, and so on. As a result, some folds became chemically unrepresentative slices of the dataset.

This is a subtle but important failure mode: cross-validation is only trustworthy if each fold resembles the overall data distribution.

Once I enabled shuffling, the result stabilized immediately:

| evaluation | MAE |
|------------|-----|
| 5-fold CV (shuffled) | 0.483 ± 0.014 V |

Now the cross-validation score matched the original 80/20 split and showed a tight error bar.

But there was still a deeper problem.

Even shuffled cross-validation allows closely related materials to appear in both training and testing sets. In materials datasets, that can artificially inflate performance because the model is effectively interpolating within known chemistry families.

For a real screening model, the harder and more meaningful question is:

> Can the model predict voltages for chemistries it has never seen before?

So I introduced a grouped split.

Instead of random folds, I grouped materials by `chemsys` and held out entire chemical systems during testing. That prevented any element-combination family from appearing in both train and test.

The difference was immediate:

| evaluation | MAE | meaning |
|------------|-----|---------|
| Shuffled CV | 0.483 ± 0.014 V | optimistic, chemical leakage allowed |
| Grouped CV | 0.584 ± 0.040 V | held-out chemistries |

The gap — roughly **0.10 V (~21%)** — is the optimism introduced by chemistry-family leakage.

And here is the part I initially misunderstood:

The earlier unshuffled CV score (`0.590`) happened to land numerically close to the grouped result (`0.584`). For a moment, that misleading coincidence made it look like leakage might not matter much.

But that comparison was invalid.

The correct comparison is:

- shuffled CV → grouped CV

Only then does the leakage gap become visible.

That was the real lesson of the experiment.

**Evaluation methodology can completely change your conclusion.**

Shuffle-vs-no-shuffle is not a minor implementation detail. The only metric that matters is the one whose split reflects the way the model will actually be used.

---

## What did the model actually learn?

A useful predictive model should capture real chemistry, not just statistical noise.

To investigate that, I compared two feature-importance methods.

LightGBM’s built-in split-frequency importance ranked:

- `avg_dev_SpaceGroupNumber`

as the most important feature.

But built-in tree importance is known to be biased toward high-cardinality features.

Permutation importance — which measures how much test accuracy degrades when a feature is randomly shuffled — told a more physically meaningful story.

Under permutation importance:

- `SpaceGroupNumber` dropped from #1 to #7

The features that remained consistently important fell into broader chemical themes:

- periodic-table position (`Column`, `Number`)
- atomic size (`CovalentRadius`, `GSvolume`)
- electronegativity
- magnetic moment (`GSmagmom`)

Those descriptors map naturally onto transition-metal redox chemistry and electronic structure, which is encouraging evidence that the model learned physically meaningful trends rather than arbitrary correlations.

One important caveat: Magpie features are strongly correlated with each other.

That makes feature attribution fundamentally difficult. If two correlated features carry nearly identical information, shuffling one may barely affect the prediction because the other still supplies the signal.

So the mature interpretation is not:

> “this exact feature controls voltage”

but rather:

> “the model relies on broader chemical themes related to periodic position, size, electronegativity, and magnetism.”

Neither importance method provides perfect per-feature truth under heavy correlation.

---

## The honest scorecard

- Dummy baseline: **0.88 V**
- Composition + LightGBM (random leakage allowed): **0.48 V**
- Composition + LightGBM (held-out chemistry): **0.58 V**
- Leakage optimism: **~0.10 V**
- R² ≈ **0.60**

Composition alone is therefore a strong and surprisingly informative baseline for coarse battery screening.

But it also clearly leaves information on the table.

The remaining unexplained variance likely contains the physics composition alone cannot fully encode:

- crystal structure
- coordination environment
- oxidation-state geometry
- local bonding topology

---

## What’s next

The structure-based model now has a clear benchmark to beat:

- **MAE < 0.58 V**
- **R² > 0.60**
- under grouped, chemistry-aware evaluation

Next step: moving beyond composition and testing whether crystal-structure graph neural networks can capture the missing physics.