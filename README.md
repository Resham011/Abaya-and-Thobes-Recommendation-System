# 🧕👘 Abaya & Thobe Recommendation System

> **Hybrid ML Recommender** — SVD Collaborative Filter · TF-IDF Content Model · MobileNetV2 Visual Search

A full-stack AI-powered recommendation engine tailored for the Middle Eastern fashion industry. It combines three complementary ML models to deliver personalized thobe and abaya suggestions based on user behavior, product metadata, and garment visuals — all served through a production-ready FastAPI backend.

---

## 🧠 How It Works

The system uses a **hybrid recommendation pipeline** with three models working together:

| Model | Type | What it does |
|---|---|---|
| **SVD Collaborative Filter** | User-behavior | Learns from purchase/click/view history to surface items liked by similar users |
| **TF-IDF Content Model** | Content-based | Matches products by textual similarity — fabric, style, occasion, price range |
| **MobileNetV2 Visual Search** | Computer vision | Fine-tuned on garment images; finds visually similar items from an uploaded photo |

---

## 📁 Project Structure

```
thobe_recommender/
├── data/
│   ├── raw/
│   │   ├── images/
│   │   │   ├── abaya/          # Downloaded garment images
│   │   │   └── thobe/
│   │   └── metadata/
│   │       ├── products.csv         # 100-item synthetic product catalog
│   │       └── interactions.parquet # Synthetic user interaction log
│   └── processed/
│       └── images/             # Resized & normalized images (224×224)
├── models/
│   └── cv_model/               # Saved MobileNetV2 weights + feature embeddings
├── api.py                      # FastAPI application (written to disk by notebook)
├── api.log                     # Server log
└── tunnel_url.txt              # ngrok public URL
```

---

## ⚙️ Tech Stack

- **ML / DL**: PyTorch, TorchVision (MobileNetV2), Scikit-learn (TF-IDF, SVD), NumPy, Pandas
- **Image collection**: icrawler (Bing Image Crawler)
- **Image processing**: Pillow (PIL)
- **API**: FastAPI + Uvicorn
- **Environment**: Google Colab + Google Drive

---

## 🚀 Quickstart (Google Colab)

### 1. Open the notebook

Upload `Abaya_and_Thobes_Recommendation_System.ipynb` to Google Colab.

### 2. Run cells top to bottom

Every cell is self-contained and idempotent — re-running a cell skips work that's already done (e.g. already-downloaded images, already-trained models).

| # | Cell | What it does |
|---|------|--------------|
| 1 | **Setup** | Mount Google Drive, create folder structure |
| 2 | **Dependencies** | Install FastAPI, icrawler, and other packages |
| 3 | **Product Catalog** | Generate a 100-item synthetic products CSV |
| 4 | **User Interactions** | Generate synthetic view/click/purchase logs |
| 5 | **Image Download** | Crawl Bing for abaya & thobe images, resize to 224×224 |
| 6 | **SVD Model** | Train collaborative filter on user interaction ratings |
| 7 | **TF-IDF Model** | Build content-based cosine similarity model |
| 8 | **MobileNetV2** | Fine-tune CV model, extract and save feature embeddings |
| 9 | **API** | Write `api.py` to disk |
| 10 | **Launch** | Start Uvicorn server + expose via ngrok tunnel |
| 11 | **Test** | Hit all endpoints and print results |

### 3. Get your public URL

After launching, the notebook prints a public ngrok URL, e.g.:

```
Server live at: https://xxxx-xx-xx.ngrok-free.app
```

---

## 📡 API Reference

Base URL: `https://<your-ngrok-url>`

### `GET /v1/health`
Health check.

```json
{ "status": "ok" }
```

---

### `GET /v1/recommend/collaborative?user_id={id}&top_n={n}`
SVD-based recommendations for a user based on their interaction history.

**Params:**
| Param | Type | Default | Description |
|---|---|---|---|
| `user_id` | `int` | required | User identifier |
| `top_n` | `int` | `5` | Number of recommendations |

**Example:**
```bash
curl "https://<url>/v1/recommend/collaborative?user_id=42&top_n=5"
```

---

### `GET /v1/recommend/content?product_id={id}&top_n={n}`
TF-IDF content-based recommendations similar to a given product.

**Params:**
| Param | Type | Default | Description |
|---|---|---|---|
| `product_id` | `int` | required | Source product ID |
| `top_n` | `int` | `5` | Number of recommendations |

**Example:**
```bash
curl "https://<url>/v1/recommend/content?product_id=12&top_n=5"
```

---

### `POST /v1/recommend/visual`
Upload a garment image to find visually similar products.

**Body:** `multipart/form-data` with field `file` (JPG/PNG)

**Example:**
```bash
curl -X POST "https://<url>/v1/recommend/visual" \
  -F "file=@my_abaya.jpg"
```

---

## 🗂️ Data Details

### Product Catalog (`products.csv`)
100 synthetic garment records with:
- `product_id`, `name`, `category` (`abaya` / `thobe`)
- `color`, `fabric`, `pattern`, `occasion`, `price_range`
- `description` — used as TF-IDF input
- `image_path` — linked to downloaded images

### User Interactions (`interactions.parquet`)
Synthetic event log with implicit feedback converted to ratings:
| Event | Rating weight |
|---|---|
| `view` | 1 |
| `click` | 2 |
| `add_to_cart` | 4 |
| `purchase` | 5 |

---

## 📊 Model Details

### SVD Collaborative Filter
- Built with `numpy` matrix factorization
- Ratings matrix: users × products
- Factors are learned via Singular Value Decomposition
- Predicts ratings for unseen user–product pairs

### TF-IDF Content Model
- Vectorizes product `description` fields with `sklearn.TfidfVectorizer`
- Similarity computed via cosine distance
- Returns top-N most similar products by text features

### MobileNetV2 Visual Model
- Pre-trained MobileNetV2 backbone (ImageNet weights) from `torchvision`
- Fine-tuned classification head for abaya/thobe categories
- Feature embeddings (512-dim) extracted from penultimate layer
- Visual search via nearest-neighbor cosine similarity over saved embeddings

---

## 📈 Evaluation

The notebook evaluates models on:
- **Precision@K**, **Recall@K**, **F1-score** for ranking quality
- **A/B testing** hooks for comparing model variants in deployment
- **User feedback loop** — interactions feed back into model retraining

---

## 🔧 Configuration

Key constants defined at the top of each cell:

```python
BASE = '/content/drive/MyDrive/thobe_recommender'  # Root directory
IMAGE_SIZE = (224, 224)                             # Input size for MobileNetV2
TOP_N = 5                                           # Default recommendation count
```

---

## 📝 Notes

- **Google Colab required** — the notebook uses `google.colab.drive` for persistence. If running locally, replace `BASE` with a local path and remove the `drive.mount()` call.
- **ngrok** is used for tunneling. You may need a free ngrok account and auth token for the tunnel step.
- All product and interaction data is **synthetic** — replace with real vendor data for production use.

---

## 📄 License

This project is developed as part of a case study for an AI-powered recommendation system for the Middle Eastern fashion industry. See `LICENSE` for details.

---

## 🙏 Acknowledgements

- [MobileNetV2](https://arxiv.org/abs/1801.04381) — Howard et al., Google
- [Surprise / SVD](http://surpriselib.com/) — inspiration for collaborative filtering design
- [FastAPI](https://fastapi.tiangolo.com/) — modern Python API framework
- [icrawler](https://github.com/hellock/icrawler) — image dataset collection
