import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import models, transforms
from PIL import Image

# ---------------- CONFIG ----------------
WEIGHTS_PATH = r"New_weights_metadata\densenet201_with_metadata.pt"

# Must be EXACTLY the same classes order as training (ImageFolder order)
CLASS_NAMES = ['LDUAL', 'LFINF', 'LFM', 'LN', 'LWAL', 'TW']

META_COLS = ("country", "city")

# ---------------- Metadata encoder (same as training) ----------------
class ManualOneHotMeta:
    def __init__(self, cols):
        self.cols = cols
        self.vocab = {}
        self.index = {}
        self.offsets = {}
        self.meta_dim = 0

    def fit(self, df_train):
        offset = 0
        for col in self.cols:
            cats = sorted(df_train[col].unique().tolist())
            if "UNKNOWN" not in cats:
                cats.append("UNKNOWN")
            self.vocab[col] = cats
            self.index[col] = {v: i for i, v in enumerate(cats)}
            self.offsets[col] = offset
            offset += len(cats)
        self.meta_dim = offset

    def encode(self, row):
        vec = np.zeros((self.meta_dim,), dtype=np.float32)
        for col in self.cols:
            v = row.get(col, "UNKNOWN")
            if v is None:
                v = "UNKNOWN"
            v = str(v).strip()
            i = self.index[col].get(v, self.index[col]["UNKNOWN"])
            vec[self.offsets[col] + i] = 1.0
        return vec

# ---------------- Model (same as training) ----------------
class DenseNet201WithMetadata(nn.Module):
    def __init__(self, num_classes, meta_dim, fusion_mode="concat", meta_hidden=128, dropout=0.5):
        super().__init__()
        self.fusion_mode = fusion_mode

        self.backbone = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        img_feat_dim = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, meta_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        if fusion_mode == "concat":
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(img_feat_dim + meta_hidden, num_classes),
            )
        else:
            raise ValueError("This inference example expects fusion_mode='concat'")

    def forward(self, x, meta):
        img_feat = self.backbone(x)
        meta_feat = self.meta_mlp(meta)
        feat = torch.cat([img_feat, meta_feat], dim=1)
        return self.classifier(feat)

# ---------------- Image preprocessing (same as val/test) ----------------
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Minimal "input files" ----
    # 1) One image file
    image_path = r"test\LFINF\705099166_1671_0.jpg"

    # 2) Minimal metadata CSV (ONE row)
    #    (Must contain country, city for the SAME image filename)
    meta_csv = "proof_inference.csv"

    # 3) Minimal training metadata CSV used ONLY to recreate the metadata vocab
    #    (Because manual one-hot needs the same vocab as training.)
    train_meta_csv = r"train_data.csv"

    # Load metadata
    df_train = pd.read_csv(train_meta_csv)
    df_train["country"] = df_train["country"].fillna("UNKNOWN").astype(str).str.strip()
    df_train["city"]    = df_train["city"].fillna("UNKNOWN").astype(str).str.strip()

    encoder = ManualOneHotMeta(META_COLS)
    encoder.fit(df_train)
    meta_dim = encoder.meta_dim

    df_meta = pd.read_csv(meta_csv)
    row = df_meta.iloc[0].to_dict()
    row["country"] = str(row.get("country", "UNKNOWN")).strip()
    row["city"]    = str(row.get("city", "UNKNOWN")).strip()
    meta_vec = encoder.encode(row)

    # Build model + load weights
    model = DenseNet201WithMetadata(num_classes=len(CLASS_NAMES), meta_dim=meta_dim).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    # Load + preprocess image
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)                # (1,3,256,256)
    m = torch.tensor(meta_vec).unsqueeze(0).to(device)         # (1,meta_dim)

    # Predict
    with torch.no_grad():
        logits = model(x, m)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    print("Prediction:", CLASS_NAMES[pred_idx])
    print("Probabilities:", {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))})

if __name__ == "__main__":
    main()
