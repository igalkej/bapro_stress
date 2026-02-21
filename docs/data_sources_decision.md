# Data Sources Decision

## Context

The BAPRO financial stress system requires a reliable, cost-free stream of
Argentina-focused financial news to serve as inputs for the TiDE model.
This document records the evaluated sources and the rationale for selecting or
rejecting each one.

---

## Selected Sources

### 1. GDELT (Primary)

- **Type:** Global event database, machine-curated from open web sources
- **Coverage:** Global, with country and theme filters (`sourcecountry:AR`)
- **Cost:** Free, no API key required
- **API:** GDELT DOC 2.0 — `https://api.gdeltproject.org/api/v2/doc/doc`
- **Fields used:** `title` (headline), `url`, `tone`, `themes`, `seendate`
- **Embed text:** `headline + " " + " ".join(gdelt_themes)`
- **Rationale:** Broadest coverage of Argentina sovereign/macro events;
  automatically tags stress-relevant GDELT themes (e.g. `ECON_BANKRUPTCY`,
  `FIN_MARKETS_STOCKS`). Free tier is sufficient for daily and historical
  backfill at business-day granularity.
- **Limitations:** Headlines are often in English; no full article body;
  occasional duplicates across news agencies.

### 2. RSS Feeds (Secondary — Spanish-language local sources)

| Feed key | URL | Focus |
|---|---|---|
| `ambito_rss` | `https://www.ambito.com/rss/economia.xml` | Economy, FX, bonds |
| `cronista_rss` | `https://www.cronista.com/rss/mercados.rss` | Markets, equities |
| `infobae_rss` | `https://www.infobae.com/feeds/rss/economia/` | General economy |

- **Type:** Standard RSS/Atom feeds from Argentine financial media
- **Coverage:** Argentina-only, Spanish language
- **Cost:** Free, publicly accessible
- **Fields used:** `title` (headline only), `link`, `published`
- **Embed text:** Headline only (no themes available from RSS)
- **Rationale:** Complements GDELT with local Spanish-language perspective.
  Ambito and Cronista are the two most widely cited financial wire services
  in Argentina. Infobae provides high-volume general economy coverage.
- **Limitations:** No tone/theme metadata; feed availability may vary;
  archives typically only go back 30-60 days.

---

## Rejected Sources

### Bloomberg Terminal / Reuters Eikon
- **Reason:** Paid subscription required (~USD 2,000+/month). Not viable for
  open-source or research use. Would introduce licensing constraints.

### FRED API (Federal Reserve Economic Data)
- **Reason:** Argentina-specific series are limited; EMBI+ spread requires
  manual download or JP Morgan subscription. Replaced by `yfinance` proxy
  (`EMB` ETF) in the FSI target build.

### Infobae Full Article Scraping
- **Reason:** Robots.txt disallows automated full-text scraping. Headlines
  via RSS are allowed and sufficient for embedding quality.

### Twitter / X API
- **Reason:** API access is no longer free; real-time sentiment from social
  media adds noise without clear predictive benefit for next-day FSI at this
  stage.

---

## Embedding Strategy

1. **GDELT articles:** `headline + " " + " ".join(gdelt_themes)`
   - Themes act as structured tags that capture event type even when the
     headline is vague.
2. **RSS articles:** headline only.
3. **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
   Pre-downloaded into the Docker image to avoid runtime latency.
4. **Per-day aggregation:** Mean-pool all article embeddings for a given
   business day to produce one 384-dim vector used as the TiDE past covariate.

---

## Idempotency Guarantee

Both ingestors use `url` as a UNIQUE key (`INSERT OR IGNORE` / `ON CONFLICT DO NOTHING`).
Re-running the same date range will not create duplicate rows.
