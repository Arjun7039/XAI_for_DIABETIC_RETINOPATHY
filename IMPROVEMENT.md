# RetinaScreen AI — Planned Improvements & Engineering Roadmap

> **Document Purpose:** This document outlines pending engineering enhancements, research extensions, and enterprise hospital features planned for *RetinaScreen AI*.

---

## 🚀 Planned Improvements & Roadmap

### 1. Live Multimodal VLM Clinical Dialogue (Gemini 1.5 Flash / Med-PaLM)
* **Objective:** Enable interactive, multi-turn clinical question-answering between the clinician and an integrated Medical Vision-Language Model directly within the workstation.
* **Technical Blueprint:**
  * Add an optional external API toggle (`GEMINI_API_KEY` or `OPENAI_API_KEY`) in the backend settings.
  * Construct a structured multimodal prompt containing:
    * The high-resolution retinal fundus crop.
    * The localized Grad-CAM++ heatmap overlay.
    * Structured findings from the 3-Agent Clinical Loop (ICD-10 code, hallmark lesions, conformal prediction set).
  * **Clinician Workflow:** The ophthalmologist can ask targeted follow-up questions in a real-time consultation drawer:
    * *"Is there evidence of clinically significant macular edema (CSME) involving the foveal avascular zone?"*
    * *"What are the differential considerations between severe NPDR and early PDR in the superior temporal quadrant?"*
  * Ground all answers with citations to specific image coordinates and ETDRS / AAO study protocols.

---

### 2. Direct DICOM & PACS Hospital Ingestion (PyDICOM & DCM4CHEE)
* **Objective:** Allow direct image acquisition from hospital Picture Archiving and Communication Systems (PACS) and ophthalmic fundus cameras (Zeiss, Topcon, Heidelberg, Canon).
* **Technical Blueprint:**
  * Implement a DICOM receiver endpoint using `pydicom` and `pynetdicom`.
  * Support DICOM C-STORE protocol to receive `.dcm` files directly from clinic network cameras.
  * Extract standardized DICOM metadata:
    * Patient ID (`(0010, 0020)`), Patient Age/Sex, Study Date (`(0008, 0020)`).
    * Image Laterality (`(0020, 0062)`: `R` for Right Eye / OD, `L` for Left Eye / OS).
    * Photometric Interpretation (`RGB` vs `MONOCHROME2`) with auto-conversion.
  * Automatically de-identify protected health information (PHI) before forwarding to the inference pipeline.

---

### 3. Automated Microvascular Morphometry & Cup-to-Disc Ratio (CDR)
* **Objective:** Expand diagnostic utility beyond diabetic retinopathy to evaluate hypertensive retinopathy and glaucoma risk from the same fundus photograph.
* **Technical Blueprint:**
  * **Arteriolar-to-Venular Ratio (AVR) Calculation:**
    * Implement a U-Net / SegFormer retinal vessel segmentation model to separate retinal arterioles from venules.
    * Measure vessel calibers at 0.5 to 1.0 disc diameters from the optic disc margin.
    * Calculate the Parr-Hubbard-Knudtson AVR index (AVR < 0.65 indicates generalized arteriolar narrowing from systemic hypertension).
  * **Optic Cup-to-Disc Ratio (CDR) Measurement:**
    * Segment the optic disc boundary and optic cup excavation.
    * Calculate vertical CDR (CDR > 0.7 or asymmetry > 0.2 indicates high risk of glaucomatous optic neuropathy).

---

### 4. Deep Out-of-Distribution (OOD) Feature Embeddings
* **Objective:** Prevent adversarial, corrupted, or non-retinal images from producing spurious predictions when they pass initial color/blur filters.
* **Technical Blueprint:**
  * Extract 1792-dimensional latent feature embeddings from the penultimate layer of the EfficientNet-B4 backbone.
  * Fit an empirical Gaussian class-conditional distribution on the training set feature space.
  * Compute the **Mahalanobis Distance** of incoming test embeddings to the nearest class centroid.
  * If the minimum Mahalanobis distance exceeds a calibrated threshold $\tau_{\text{OOD}}$, flag the image as out-of-distribution (`HTTP 422: Out-of-Distribution Image Detected - Please submit an authentic retinal fundus photograph`).

---

### 5. Production Observability & Prometheus/Grafana Dashboard
* **Objective:** Enable enterprise Site Reliability Engineering (SRE) and drift monitoring in production hospital environments.
* **Technical Blueprint:**
  * Integrate `prometheus-fastapi-instrumentator` in `backend/app/main.py` exposing a `/metrics` scrape target.
  * Expose key operational metrics:
    * `dr_inference_latency_seconds_bucket` ($P_{50}, P_{90}, P_{95}, P_{99}$ latency histograms).
    * `dr_quality_rejection_total` (counter of blur, glare, and document rejections).
    * `dr_prediction_class_total` (counter partitioned by predicted ICDR class to detect distribution drift).
    * `dr_agent_consensus_contradictions_total` (counter tracking instances where Agent 3 overrides Agent 2).
  * Provide pre-configured Grafana dashboard JSON templates for 1-click monitoring.

---

### 6. FHIR (Fast Healthcare Interoperability Resources) & EHR Integration
* **Objective:** Export screening outputs directly into hospital Electronic Health Record (EHR) systems like Epic, Cerner, and Allscripts.
* **Technical Blueprint:**
  * Implement an HL7 FHIR v4.0.1 export service (`backend/app/integrations/fhir.py`).
  * Map diagnostic outputs to standard FHIR resources:
    * `DiagnosticReport`: Overall DR screening report, status, and clinician recommendation.
    * `Observation`: Individual observations for ICDR stage (LOINC code `74768-3`: Diabetic retinopathy severity), ICD-10 code, and conformal confidence set.
    * `DocumentReference`: Encapsulated base64 PDF medical report for direct attachment to the patient's electronic medical chart.

---

### 7. Offline Edge Screening & TFLite / ONNX Runtime Support
* **Objective:** Allow portable fundus camera attachments (e.g. smartphone-based ophthalmoscopes) to run screening in rural camps without internet connectivity.
* **Technical Blueprint:**
  * Quantize the model using INT8 post-training quantization via TensorFlow Lite or ONNX Runtime.
  * Package as a lightweight WebAssembly (WASM) or mobile edge module (<25MB) running fully client-side on iOS / Android or within modern web browsers.

---

## 📈 Summary Phasing

```
Phase 1: Live VLM Dialogue & Gemini 1.5 Pro Interactive Consultation
   │
   ▼
Phase 2: DICOM PACS Ingestion & Multi-Organ Morphometry (AVR & Glaucoma CDR)
   │
   ▼
Phase 3: Deep OOD Feature Embeddings & Edge TFLite/ONNX Quantization
   │
   ▼
Phase 4: HL7/FHIR EHR Integration & Enterprise Prometheus/Grafana SRE
```
