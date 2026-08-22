# News Negativity Dominance Scoring

A lexicon-based algorithm for detecting **hidden negativity** in news text: cases where an article's overall tone reads as neutral or positive, but contains a genuinely negative event that a plain compound sentiment score would mask by averaging it away.

Built as a self-directed project to explore computational methods for analyzing negativity in news content, in the context of research on news diets and negativity exposure.

---

## Motivation

Standard sentiment analysis reduces a document to a single average score. But averaging can hide important information: a mostly positive article that briefly mentions a death, a layoff, or a tragedy will often average out to "neutral," even though a human reader would clearly register the negative event.

This project asks: can we detect negativity that is diluted by surrounding positive framing, rather than just measuring overall tone?

---

## Method

**Dataset:** [Kaggle News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) (~140,000 HuffPost articles, 2012-2022). Each document is constructed by merging a headline and its short description into a single text field.

**Sentiment engine:** VADER (via NLTK), chosen for its speed, interpretability, and suitability for short text.

**Pipeline:**
1. Merge `headline` and `short_description` into one `text` field per article.
2. Split `text` into individual sentences.
3. Score each sentence with VADER, producing a compound score, a negative-sentiment proportion, and a positive-sentiment proportion.
4. Compute two document-level metrics from these sentence scores (described below).

### Metrics

**`overall_tone`**: the average of the compound scores across all sentences in the document. Range: `[-1, 1]`. This reflects the document's general sentiment.

**`negativity_dominance`**: computed in three steps.
1. For each sentence, check whether its negative-sentiment proportion exceeds its positive-sentiment proportion. If so, mark that sentence as "negativity-dominant" and record its negative score.
2. Collect the negative scores of all negativity-dominant sentences in the document.
3. Take the maximum of those scores. If no sentence was negativity-dominant, the value is 0.

Range: `[0, 1]`. This reflects the single worst negative moment in the document, independent of how large a share of the text it occupies.

Reporting these as **two separate axes**, rather than merging them into one number, is a deliberate design choice: `overall_tone` answers "what's the general sentiment of this document," while `negativity_dominance` answers "did something negative happen here, even briefly." Using the maximum (rather than, say, a proportion-weighted average) ensures that a single sharp negative sentence is not diluted just because the rest of the document is positively framed, which is the core behavior needed to surface hidden negativity in short, two- to three-sentence documents like a headline paired with its description.

---

## Results

### Distributions (full dataset, n = 140,854)

![Histograms](histograms.png)

- **`overall_tone`** is roughly bell-shaped and centered slightly above zero (mean approximately 0.08), consistent with the generally neutral-to-positive framing conventions of headline writing. A sharp spike exactly at 0 corresponds to documents with no sentiment-bearing words detected at all.
- **`negativity_dominance`** is zero for the majority of documents (approximately 59%), meaning most headline-and-description pairs contain no sentence where negativity outweighs positivity. Where dominance is nonzero, scores spread across the full range, including a small but distinct cluster at 1.0 (documents containing at least one maximally negative sentence).

### Tone vs. dominance

![Quadrant plot](quadrant_plot.png)

- At the negative-tone extreme (tone approaching -1), dominance is consistently high. This is expected, since an overall negative average is usually driven by sentences that are individually negative.
- The more analytically interesting region is tone at or above 0 combined with elevated dominance: documents that read as neutral or positive on average but contain a genuine negative spike. This is the hidden negativity phenomenon the metric was designed to surface, and its presence, visible as points scattered above the positive-tone region, including a horizontal band at dominance = 1.0 spanning positive tone values, confirms the metric captures something `overall_tone` alone misses.
- Two flat horizontal bands appear at the top and bottom of the plot. The bottom band (dominance = 0) holds every document where no sentence was negativity-dominant, regardless of overall tone. The top band (dominance = 1.0) holds documents where at least one sentence, often short and blunt, hit VADER's negativity ceiling; a short headline needs very few negative words to reach this maximum, which is why the band spans a wide range of tone values rather than sitting only on the negative side.

### Constructed test cases

| Case | Sentence 2 | `overall_tone` | `negativity_dominance` |
|---|---|---|---|
| Procedural framing | "...forced to lay off a third of its teaching staff." | +0.057 | 0.15 |
| Emotional framing (same event) | "...outrage and heartbreak after brutal layoffs devastated..." | -0.186 | 0.58 |
| Hidden negativity | "...tragic warehouse fire killed two workers." (paired with a positive headline) | +0.012 | 0.43 |
| Contrastive structure | "Despite early fears of disaster, the surgery was a complete success." | +0.117 | 0.00 |

These controlled examples surface two distinct properties of the underlying lexicon, discussed below.

---

## Limitations

- **Sensitivity to vocabulary, not event severity.** The same underlying event (staff layoffs) produced a nearly fourfold difference in `negativity_dominance` (0.15 vs. 0.58) depending on whether it was described procedurally or with emotionally charged language. This is inherited directly from VADER, which is tuned on social-media-style text and reacts more strongly to visceral wording than to clinical descriptions of real harm, such as layoffs, displacement, or closures.
- **Document length assumption.** With most documents limited to two sentences, `negativity_dominance` reduces to "the worse of at most two scores." This is appropriate at this scale but would need to be redesigned for longer, multi-paragraph text, where a single maximum is less informative about whether negativity is sustained or a one-off spike.
- **Sentence-level scoring, not full-document semantics.** VADER scores sentences independently; it does not track discourse-level context across sentences, such as whether an earlier sentence's negativity is resolved or reversed later in the document.
- **Contrastive structures are sometimes handled correctly, sometimes not.** In one test, VADER correctly resolved "despite fears of disaster, the surgery was a complete success" as non-negative, avoiding a false positive from the negative-sounding setup clause. This suggests VADER's behavior here is inconsistent rather than uniformly biased, and shouldn't be assumed to generalize without further testing.

## Future Work

- Extend to full-length article text and compare the maximum-based scoring used here against an average of the top-*k* most negative sentences, which should better capture sustained (rather than single-spike) negativity in longer documents.
- Validate against a small hand-labeled sample to quantify agreement with human judgments of hidden negativity.
- Compare against a classic machine learning classifier (for example, TF-IDF features with logistic regression) trained on labeled news sentiment data, to see whether a supervised approach reduces the vocabulary-sensitivity issue observed here.
- Break results down by category or outlet to explore whether negativity dominance varies systematically across news topics.

---

## Requirements

- NLTK (with the VADER lexicon and the Punkt sentence tokenizer)
- pandas
- NumPy
- Matplotlib

## Repository Structure

```
├── negativity_dominance_scoring.ipynb   # full analysis notebook
├── histograms.png                        # distribution plots
├── quadrant_plot.png                     # tone vs. dominance scatter
└── README.md
```
