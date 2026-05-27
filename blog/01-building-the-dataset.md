# Building a clean battery-materials dataset from the Materials Project  
### …and why I kept the “bad” entries

I’m a computational materials researcher working at the intersection of electronic-structure simulation, scientific computing, and machine learning. For this project, I started building a machine-learning dataset to predict the average voltage of Li insertion-electrode cathodes using data from the Materials Project.

This post is about the least glamorous part of the pipeline — the data work. Before models, embeddings, or benchmarks, there’s a more important question: *is the dataset actually trustworthy?*

---

## Why cathode voltage?

Average insertion voltage is one of the key quantities that determines the energy density of a lithium-ion battery cathode. Higher voltage generally means higher energy storage, but only within physically and chemically stable limits.

Traditionally, estimating voltages for new materials requires density functional theory (DFT) calculations, which are computationally expensive at scale. A reliable ML model could help screen thousands of candidate materials before running high-cost simulations, accelerating the search for better battery chemistries.

---

## Getting the data: inspect before you ingest

I used the Materials Project API through the Python client provided by the `mp-api` package.

Before downloading thousands of entries, I first inspected the schema:

- queried `available_fields`
- pulled a small 10-record sample
- checked real field names, data types, and nested structures

That step matters more than people think. You should never bulk-download an unfamiliar API blindly and hope the fields behave the way you expect.

The final query returned:

- **2,774 Li insertion-electrode records**

One immediate surprise: several fields were not plain Python values. For example:

- `working_ion` came back as a `pymatgen.Element`
- `battery_id` was an `MPID` object
- `elements` was a list of `Element` objects

Useful scientifically, inconvenient operationally.

---

## Flattening to a tidy table

To build a usable ML dataset, those objects had to be converted into standard serializable types.

For example:

- `Element("Li") → "Li"`
- lists of `Element` objects → lists of strings

There were two practical reasons for this:

1. clean, filterable tabular columns  
2. Parquet storage only supports standard serializable data types — not arbitrary Python objects

I also avoided repeatedly appending rows to a DataFrame inside a loop. Instead, I built a list of dictionaries and constructed the DataFrame once at the end, which is both cleaner and significantly faster for larger datasets.

---

## The decision that mattered: filter on physics, not on flags

This ended up being the most important part of the entire preprocessing pipeline.

The raw voltage distribution immediately showed obvious issues:

| stat | raw value |
|------|-----------|
| median | 3.54 V |
| std | 1.53 V |
| min | −7.75 V |
| max | 33.07 V |

A negative insertion voltage is unphysical for a Li cathode. A 33 V cathode is not a breakthrough battery chemistry — it is almost certainly a computational artifact or problematic entry.

The raw dataset contained:

- **72 entries** with negative voltage
- **976 entries (35%)** marked with warnings

At that point, the obvious temptation was:

> “Drop all warned rows.”

But removing a third of the dataset without checking whether the warnings actually correlate with bad physics is not data cleaning — it is data destruction.

So I checked.

### Voltage distributions by warning group

| group | count | mean | median |
|-------|-------|------|--------|
| warned | 976 | 3.51 V | 3.70 V |
| unwarned | 1798 | 3.37 V | 3.48 V |

The two distributions were remarkably similar.

The warnings were *not* acting as a meaningful quality signal.

Even more telling:

- only **48 of the 140 physically implausible entries** were warned

So the warning flag and the actual problematic voltages were largely independent.

The two dominant warning types were:

- “More than one working ion per transition metal” (**823 entries**)
- “Transition metal not found” (**153 entries**)

These are descriptive metadata warnings, not indicators that the calculation itself is broken.

### Final decision

Instead of filtering on warnings, I filtered on physics:

```python
0 < voltage <= 5.5
````

and preserved warning information as potential *features* for downstream models.

The 5.5 V upper cutoff came from both:

* domain knowledge of realistic Li cathode chemistry
* the empirical distribution itself (75th percentile ≈ 4.2 V)

So the cutoff removes only a small high-voltage artifact tail while preserving the physically meaningful distribution.

---

## Result

* Removed **140 entries (5%)**
* Final clean dataset: **2,634 cathodes**
* Standard deviation reduced from **1.53 → 1.07 V**
* Final voltage range: **0.02 – 5.50 V**

The important part is not that the dataset became “cleaner.”

It became *physically consistent* without throwing away a large fraction of potentially useful data.

---

## How it’s built

What started as exploratory notebook work has now been refactored into a small Python package structure:

* separate `fetch` and `clean` functions
* unit tests for cleaning logic
* CI for validation
* Parquet export pipeline

The goal was reproducibility: rerunning the pipeline should regenerate the exact same cleaned dataset from the raw Materials Project query.

You can explore the project and code structure on GitHub.

---

## Next

Next step: featurization, compositional descriptors, and the first baseline ML models for voltage prediction.

```
```