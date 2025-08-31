Glaucoma Detection using Deep Learning

📌 Project Overview

This project explores AI-driven glaucoma detection and progression prediction using various deep learning and machine learning architectures. The study compares multiple models applied to fundus photographs, OCT images, visual field (VF) data, intraocular pressure (IOP), and electronic health records (EHR). The objective is to identify efficient, accurate, and generalizable models for early glaucoma diagnosis and progression monitoring.

The project also emphasizes the advantages of Multiclass UNet (for segmentation) and ResNet150 + Inception V3 (for classification), highlighting their superior performance over existing models.


🧠 Models Studied

MFDL (Multi-Feature Deep Learning) – Combines IOP, CFP, and VF data.

3D Deep Learning Systems – Uses Fundus, OCT, and SAP for glaucoma detection.

CNN-based OD/OC Segmentation (Veena HN et al.) – Optic disc/cup segmentation.

AI Framework Reviews – Report up to 100% accuracy but face generalizability issues.

DeiT vs ResNet-50 – DeiT offers better cross-ethnicity generalizability.

DL on EHR Data – Predicts glaucoma progression using structured + NLP features.

ResNet, Vision Transformer, gMLP with Adversarial Robustness.

Schuman et al. AI Models – OCT, VF, and fundus–based detection.

ML with Corneal Densitometry – Detects glaucoma risk (83.93% accuracy).

AI with Fundus + OCT – AUC > 0.90, but quality/ground-truth issues.

Explainable AI (LIME, SHAP) – Improves transparency.

CoG-NET (Modified Xception) – Lightweight glaucoma classifier.

C2P EMS2 (Ensemble Bagging Classifier) – VF-based early glaucoma detection.

UNet++ + ResNet-GRU – Combined segmentation + classification.

Triggerfish Sensor + ML – Ocular + cardiac data for NTG detection.

Multimodal ML Models – Predict retinal thickness for trial optimization.



🚀 Why Multiclass UNet + ResNet150 Inception V3?

Multiclass UNet: Excels at segmenting optic disc/cup regions with high precision.

ResNet150: Deep residual architecture captures fine retinal details, minimizing vanishing gradients.

Inception V3: Efficiently processes multiple image scales, improving robustness.

Better Generalizability: Handles diverse datasets with reduced overfitting.

Clinical Readiness: Optimized for real-world application across populations.

Multiclass Capability: Enables detection of different glaucoma stages.



⚙️ Tech Stack

Languages: Python

Frameworks: TensorFlow, PyTorch, Scikit-learn

Visualization: Matplotlib, Seaborn, Grad-CAM

NLP (for EHRs): spaCy, Hugging Face Transformers

Explainability: LIME, SHAP


📌 Future Work

Multi-center validation for broader generalizability.

Integration of multimodal data (OCT + VF + Fundus + EHR).

Advanced NLP for unstructured medical text.

Deployment as a lightweight clinical decision-support system.

