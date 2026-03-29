
<div align="center"\>
<h1 align="center"\>Face-Recognition-System 🧠👁️\</h1\>
<h3\>نظام متقدم للتعرف على الوجوه الحيوية بالزمن الفعلي\</h3\>
</div\>

\<p align="center"\>
مشروع متكامل يجمع بين قوة الذكاء الاصطناعي (Deep Learning) وتصميم واجهات المستخدم المستقبلية (Sci-Fi HUD) للتعرف على الوجوه والتحقق من الهوية بشكل فوري ومذهل بصرياً.
\</p\>

-----

## 📖 نظرة عامة ومواصفات المشروع

هذا المشروع عبارة عن نظام أمني وتعريفي يعتمد على رؤية الحاسب (Computer Vision) والشبكات العصبية العميقة. يقوم النظام باكتشاف الوجوه عبر كاميرا الويب، وتحويل ملامح الوجه إلى أرقام أو بصمات رياضية (Embeddings) باستخدام بنية **EfficientNetV2M**، ثم يقارنها بالأشخاص المعرفين مسبقاً في قاعدة الحسابات لمنح أو رفض الدخول.

**أبرز المواصفات التقنية:**

  * **التعرف الفوري على الوجوه (Real-Time Recognition):** معالجة فائقة السرعة لإطارات الفيديو باستخدام نموذج `PyTorch` (`best_model.pth`).
  * **اكتشاف الوجه (Face Detection):** باستخدام مصنفات Haar Cascade لاقتطاع الوجه المستهدف.
  * **تحليل وتوليد البصمة الحيوية:** باستخدام `TensorFlow / Keras` لتحويل آلاف الصور إلى متجهات دقيقة (Vectors).
  * **واجهة مستخدم تفاعلية (Cyberpunk HUD):** مبنية بـ `PyQt6` وتتضمن تأثيرات بصرية متقدمة مثل الشبكة الحيوية ومحاكاة الجسيمات.

-----

## 📂 بنية المشروع ووظيفة كل ملف

لكي تفهم أين تبدأ وأين تنتهي، هذه نبذة عن الملفات الأساسية بترتيب الأهمية العملياتية:

1.  **`taking_photos.py` (نقطة البداية):** السكربت الخاص بجمع البيانات (التقاط 2500 صورة).
2.  **`model.py` (التدريب المحلي):** تدريب الشبكة العصبية لإنتاج نموذج `my_face_model.keras`.
3.  **`colab_training_code.py`:** بدائل للتدريب السحابي باستخدام Google Colab.
4.  **`generate_all_embeddings.py` (التشفير الرقمي):** يحول صور كل شخص إلى ملف بصمة رقمية `.npy`.
5.  **`main_app.py` (نقطة النهاية):** الواجهة الرسومية التي تشغل الكاميرا الحية لتأكيد الهوية.

-----

## 🚀 المسار الكامل للتشغيل (دليل الخطوات)

### الخطوة 1: إعداد بيئة العمل

تثبيت المكتبات اللازمة عبر الأمر:

```bash
pip install -r requirements.txt
```

### الخطوة 2: تسجيل الوجوه (تجميع البيانات)

قم بتشغيل `taking_photos.py` لإدخال اسمك والتقاط صور وجهك (2500 صورة). تحرك ببطء وغير تعابير وجهك لضمان الدقة.

### الخطوة 3: تدريب الموديل

استخدم `model.py` للتدريب المحلي أو ملفات Colab المرفقة للتدريب السحابي للحصول على أوزان النموذج.

### الخطوة 4: توليد البصمات الحيوية

قم بتشغيل `generate_all_embeddings.py` لتحويل الصور إلى بيانات رياضية (Embeddings) مخزنة في مجلد `embeddings/`.

### الخطوة 5: التشغيل النهائي

شغل `main_app.py` لفتح الواجهة السايبرية والبدء في عملية التعرف الحية.

-----
\<div align="center"\>
\<h1 align="center"\>Face-Recognition-System🧠👁️\</h1\>
\<h3\>Advanced Real-Time Biometric Face Recognition System\</h3\>
\</div\>

\<p align="center"\>
An integrated project combining the power of Deep Learning with futuristic Sci-Fi HUD design for instant and visually stunning facial recognition and identity verification.
\</p\>

-----

## 📖 Overview and Project Specifications

This project is a security and identification system based on Computer Vision and Deep Neural Networks. The system detects faces via webcam, converts facial features into numerical vectors (Embeddings) using the **EfficientNetV2M** architecture, and compares them against registered individuals in the database.

**Key Technical Specifications:**

  * **Real-Time Recognition:** Ultra-fast video frame processing using a `PyTorch` model (`best_model.pth`).
  * **Face Detection:** Utilizing Haar Cascade classifiers to crop target faces.
  * **Biometric Embedding Generation:** Using `TensorFlow / Keras` to transform thousands of images into precise vectors.
  * **Futuristic HUD Interface:** Built with `PyQt6`, featuring Biometric Grids, Particle Simulations, and a Live Terminal.

-----

## 📂 Project Structure and File Functions

To understand the workflow, here is a summary of the core files in operational order:

1.  **`taking_photos.py` (Start Point):** Data collection script (captures 2500 images).
2.  **`model.py` (Local Training):** Script to train the neural network and produce `my_face_model.keras`.
3.  **`colab_training_code.py`:** Cloud training alternatives for Google Colab.
4.  **`generate_all_embeddings.py` (Digital Encoding):** Iterates through images to create `.npy` biometric fingerprint files.
5.  **`main_app.py` (End Point):** The main GUI core that runs the live camera for identity verification.

-----

## 🚀 Comprehensive Setup Guide

### Step 1: Environment Setup

Install the necessary dependencies using:

```bash
pip install -r requirements.txt
```

### Step 2: Face Enrollment (Data Collection)

Run `taking_photos.py`, enter your name, and let the system capture 2500 images. Move slowly and change expressions for better accuracy.

### Step 3: Model Training

Run `model.py` for local training or use the provided Colab files for cloud training to generate the model weights.

### Step 4: Generate Biometric Embeddings

Run `generate_all_embeddings.py` to convert images into mathematical data stored in the `embeddings/` folder.

### Step 5: Final Execution

Run `main_app.py` to launch the **NEURAL NEXUS** interface and start live recognition.

-----

## 🛠 Troubleshooting

  * **Camera Error:** Ensure no other app is using the webcam.
  * **Missing Model:** If `best_model.pth` is missing, you must run the training script or download it manually.
  * **Slow Recognition:** This is expected on systems without GPU acceleration due to the heavy VFX processing.

-----
