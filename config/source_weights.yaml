# Source reliability tiers used by weighting/source_weight.py.
# Matching is by longest-substring match of the publisher/domain string against
# these keys (case-insensitive). A publisher that is present in `verified_domains`
# but not explicitly listed below falls back to `default_verified`. A missing/blank
# source falls back to `unknown`.

reliability:
  Reuters: 1.00
  Bloomberg: 1.00
  "The Economic Times": 0.95
  "Business Standard": 0.95
  Mint: 0.92
  "Financial Express": 0.90
  "Moneycontrol": 0.90
  "Livemint": 0.90
  "Business Today": 0.88
  "CNBC-TV18": 0.88
  "CNBC TV18": 0.88
  "The Hindu BusinessLine": 0.88
  "Hindustan Times": 0.85
  "Times of India": 0.85
  "India Today": 0.83
  NDTV: 0.83
  "The Indian Express": 0.83
  "Zee Business": 0.80
  "Press Trust of India": 0.95
  PTI: 0.95
  ANI: 0.90

default_verified: 0.80
unknown: 0.65

# Domains treated as "verified but not individually rated" -> default_verified.
verified_domains:
  - moneycontrol.com
  - economictimes.indiatimes.com
  - business-standard.com
  - livemint.com
  - financialexpress.com
  - businesstoday.in
  - cnbctv18.com
  - thehindubusinessline.com
  - hindustantimes.com
  - timesofindia.indiatimes.com
  - indiatoday.in
  - ndtv.com
  - indianexpress.com
  - zeebiz.com
  - reuters.com
  - bloomberg.com
  - ptinews.com
  - aninews.in
