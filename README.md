
# 🧠 HysoLLM – Makine Öğrenimi & LLM Eğitim Altyapısı

HysoLLM, makine öğrenimi ve özellikle büyük dil modeli (LLM) projelerinde **eğitim süreçlerini düzenlemek, deneyleri kaydetmek, yeniden üretilebilir hale getirmek ve modüler bir mimari üzerine oturtmak** amacıyla geliştirilmiş hafif bir framework’tür.

Bu altyapı, modern ML projelerinde ihtiyaç duyulan temel bileşenleri tek çatı altında toplar:

- ⚙️ Modüler model mimarileri  
- 🔤 Tokenizer sistemi (BPE + Simple)  
- 📦 Deney yönetimi (run paths, manifest, logger, checkpoints, seed)  
- ⚡ Eğitim döngüsü (trainer, callback sistemi)  
- 🧾 Config yönetimi (YAML/JSON + CLI override)  
- 📁 Harici config dosyaları (configs/)  

Amaç:  
**Model kodunu eğitim mekanizmalarından tamamen ayırarak**, farklı projelerde yeniden kullanılabilir, düzenli ve sürdürülebilir bir yapı oluşturmak.
```python
from hyso.core.config import load_config_with_overrides
from hyso.core.storage import (
    RunPathFactory,
    get_logger,
    Manifest,
    save_manifest,
    CheckpointConfig,
    CheckpointManager,
    set_global_seed,
)
from hyso.core.tokenizer import HysoBPETokenizer
from hyso.core.models.encoder_only import HysoEncoderOnly
from hyso.core.train import Trainer

import torch
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------
# 1) Config yükleme (+ override desteği)
# ---------------------------------------------------------

cfg = load_config_with_overrides(
    path="configs/base.yaml",
    override_pairs=["training.lr=0.0001", "model.dim=512"]
)


# ---------------------------------------------------------
# 2) Run path oluşturma
# ---------------------------------------------------------

factory = RunPathFactory.from_root("runs")
run_paths = factory.create()

logger = get_logger("train", log_dir=run_paths.logs_dir)
logger.info(f"Run başlatıldı: {run_paths.run_id}")


# ---------------------------------------------------------
# 3) Seed ayarı (deterministik eğitim için)
# ---------------------------------------------------------

set_global_seed(cfg.training.seed)


# ---------------------------------------------------------
# 4) Manifest oluşturma
# ---------------------------------------------------------

manifest = Manifest.new(
    run_id=run_paths.run_id,
    model=cfg.model,
    training=cfg.training,
    data={"name": "dummy"},
)

save_manifest(run_paths.manifest_path, manifest)
logger.info("Manifest kaydedildi.")


# ---------------------------------------------------------
# 5) Tokenizer yükleme
# ---------------------------------------------------------

tokenizer = HysoBPETokenizer(
    lowercase=True,
    normalize="NFKC",
    cache_size=5000,
)


# ---------------------------------------------------------
# 6) Model oluşturma
# ---------------------------------------------------------

model = HysoEncoderOnly(
    dim=cfg.model.dim,
    num_layers=cfg.model.layers,
    vocab_size=tokenizer.vocab_size,
)

model = model.to(cfg.training.device)


# ---------------------------------------------------------
# 7) Dataset & Dataloader
# ---------------------------------------------------------

X = torch.randint(0, tokenizer.vocab_size, (100, 32))
y = torch.randint(0, tokenizer.vocab_size, (100, 32))

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)


# ---------------------------------------------------------
# 8) Optimizer & Scheduler
# ---------------------------------------------------------

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


# ---------------------------------------------------------
# 9) Checkpoint yöneticisi
# ---------------------------------------------------------

ckpt_cfg = CheckpointConfig.from_dir(
    directory=run_paths.checkpoints_dir,
    best_metric_name="val_loss",
    best_mode="min",
)

checkpoint_manager = CheckpointManager(ckpt_cfg)


# ---------------------------------------------------------
# 10) Trainer oluşturma
# ---------------------------------------------------------

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    tokenizer=tokenizer,
    config=cfg,
    run_paths=run_paths,
    checkpoint_manager=checkpoint_manager,
    logger=logger,
)


# ---------------------------------------------------------
# 11) Eğitim döngüsü
# ---------------------------------------------------------

trainer.fit(
    train_loader=loader,
    val_loader=loader,
    epochs=cfg.training.epochs,
)

logger.info("Eğitim tamamlandı.")

```
---
🚀 Kullanım Örneği

Aşağıda HysoLLM bileşenlerinin birlikte nasıl çalıştığını gösteren gerçek bir örnek bulunmaktadır.


---

# 📁 Proje Yapısı
Her klasör aşağıda açıklanmaktadır.
```bash
core/
  models/
  tokenizer/
  storage/
  train/
  config/
  configs/
```

---

# 🔤 1. Tokenizer Modülü (`hyso/core/tokenizer/`)

Tokenizer yapısı metni modele uygun ID dizilerine dönüştürür. HysoLLM iki tokenizer içerir:

---

### **• HysoBPETokenizer (BPE)**  
Byte Pair Encoding tabanlı tokenizer.

**Yöntemler & yetenekler:**
- BPE merge kurallarına dayalı alt birim üretimi  
- Özel token ID yönetimi  
- Batch encode/decode  
- Seq2Seq encoder ve decoder için ayrı encode fonksiyonları  
- UTF-8 byte-level işleme desteği  

**Kullanılan Teknolojiler:**
- Python  
- PyTorch (tensor dönüşümleri)  
- Unicode processing  
- BPE algoritması  

---

### **• HysoSimpleTokenizer (Word-level)**  
Kelime temelli sade tokenizer.

**Yöntemler:**
- Temel whitespace tokenizasyonu  
- Kelime frekansına göre vocabulary oluşturma  
- Batch encode/decode  
- Mask üretimi  

**Kullanılan Teknolojiler:**
- Python  
- PyTorch  
- Temel NLP tokenizasyon teknikleri  

---

# 🧩 2. Model Modülü (`hyso/core/models/`)

Bu klasör HysoLLM model mimarilerini içerir. Tüm modeller modüler ve genişletilebilir yapıdadır.

---

### **Model mimarileri:**
- **Encoder-Decoder Transformer**  
- **Encoder-only LLM** (BERT/LLM tarzı)  
- **Decoder-only Model** (GPT tarzı)

---

### **Her modelin sunduğu yöntemler:**
- `forward()`  
- `generate()` (decoder-only)  
- `encode()` / `decode()` (enc-dec yapıları)  
- RoPE pozisyonel embedding  
- Attention katmanları  
- DropPath, LayerNorm, RMSNorm gibi modern bileşenler  

**Kullanılan Teknolojiler:**
- PyTorch (nn.Module)  
- Multi-Head Attention  
- Rotary Positional Embedding (RoPE)  
- SwiGLU / GeLU aktivasyonları  
- Residual / PreNorm mimarileri  

---

# 📦 3. Storage Modülü (`hyso/core/storage/`)

Bu modül HysoLLM’in gerçek “deney yönetimi” çekirdeğidir.
Klasör yapısını otomatik oluşturur.

**Yöntemler:**  
- `create()`  
- `run_id` üretimi  
- klasör kurulumu  

## **• Logger**  
Hem konsola hem dosyaya temiz log yazan yapılandırılabilir logger.

**Yöntemler:**  
- `get_logger(name, log_dir)`  
- Formatlama (timestamp, level, name, message)  
- Rotating log desteği  


## **• Manifest**  
Her eğitimin kimlik kartıdır.

İçerik:  
- model bilgisi  
- eğitim bilgisi  
- veri seti bilgisi  
- ortam (Python, OS, CUDA)  
- hyperparametreler  
- timestamp  

**Yöntemler:**  
- `Manifest.new()`  
- `save_manifest()`  
- `load_manifest()`  

## **• CheckpointManager**  
Eğitim sırasında modelleri güvenli şekilde kaydeder.

**Yöntemler:**
- `save(epoch, model, optimizer, scheduler)`  
- `load_latest()`  
- `load_best()`  
- `restore_objects()`  
- max_to_keep ile eski checkpoint silme  


## **• Seed Yönetimi**
Tüm ortamı deterministik hale getirir.

**Yöntemler:**  
- `set_global_seed(seed)`  

**Kullanılan Teknolojiler (tüm storage):**
- PyTorch  
- Python logging  
- JSON  
- UUID  
- Pathlib  
- Datetime  

---

# ⚡ 4. Train Modülü (`hyso/core/train/`)

Bu modül model eğitimi için gerekli tüm yapılara sahiptir.

### **İçerik:**

### **• Trainer**
Model, veri, optimizer, scheduler, storage ve config bileşenlerini bir araya getiren ana eğitim sınıfı.

**Yöntemler:**
- `train_epoch()`  
- `validate()`  
- `fit()`  
- Loss hesaplama  
- Metrics kaydetme  


### **• Callback Sistemi**
Eğitimin belirli aşamalarında çalışacak küçük kancalar.

**Yöntemler:**
- `on_train_start()`  
- `on_epoch_end()`  
- `on_step_end()`  
- CallbackList ile çoklu callback desteği  

### **• Metrics Logging**
Eğitim ve validasyon metriklerini CSV’e yazar.

**Kullanılan Teknolojiler:**
- PyTorch (dataset, dataloader, optim, scheduler)  
- Callback pattern  
- CSV logging  
- Checkpoint entegrasyonu  

# 🧾 5. Config Modülü (`hyso/core/config/`)

Config sistemi ayarları koddan ayırır ve tüm eğitim süreçlerini yapılandırılabilir hale getirir.


### **Yapılan işlemler:**
- YAML / JSON config dosyası yükleme  
- CLI override desteği  
  (`training.lr=0.0001 model.layers=12` gibi)  
- Deep merge  
- Config kaydetme / okuma  
- Attribute-style erişim:  
  `cfg.training.lr`  


### **Yöntemler:**
- `load_config(path)`  
- `parse_overrides(argv)`  
- `merge_config(base, override)`  
- `load_config_with_overrides()`  
- `save_config()`  

**Kullanılan Teknolojiler:**
- PyYAML  
- JSON  
- Python AST parsing  
- Recursive merge algoritması  

---

# 📁 6. Configs Klasörü (`configs/`)

Bu klasör train modülü için harici ayar dosyalarını içerir.

Örnek dosyalar:
- `base.yaml`
- `encoder_small.yaml`
- `encoder_large.yaml`
- `lr_sweep.yaml`

Bu sayede:

- Deney ayarları versiyonlanabilir,  
- Farklı modeller için hızlı switching yapılabilir,  
- Manifest + config birleşince tam reproducibility sağlanır.

**Kullanılan Teknolojiler:**  
- YAML  
- JSON  

---

# 🧩 Özet

HysoLLM yalnızca model mimarisi değil;  
**tam bir eğitim altyapısı, deney yönetim sistemi, config framework ve modüler ML yapı setidir.**

Bu repo:

- 🔤 Tokenizer sistemini  
- ⚙️ Model mimarisini  
- 📦 Storage ve deney yönetimini  
- ⚡ Eğitim altyapısını  
- 🧾 Config yönetimini  

birbirinden ayrılmış, temiz ve profesyonel şekilde organize eder.

--- 

# ✨ Lisans
MIT License



