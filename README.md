<div align="center">

<h1 align="center">
    Face-Recognition-System 🧠👁️</h1>

<h3>نظام متقدم للتعرف على الوجوه الحيوية بالزمن الفعلي</h3>

</div>
<p align="center">

مشروع متكامل يجمع بين قوة الذكاء الاصطناعي (Deep Learning) وتصميم واجهات المستخدم المستقبلية (Sci-Fi HUD) للتعرف على الوجوه والتحقق من الهوية بشكل فوري ومذهل بصرياً.

</p>
---
## 📖 نظرة عامة ومواصفات المشروع

هذا المشروع عبارة عن نظام أمني وتعريفي يعتمد على رؤية الحاسب (Computer Vision) والشبكات العصبية العميقة. يقوم النظام باكتشاف الوجوه عبر كاميرا الويب، وتحويل ملامح الوجه إلى أرقام أو بصمات رياضية (Embeddings) باستخدام بنية **EfficientNetV2M**، ثم يقارنها بالأشخاص المعرفين مسبقاً في قاعدة الحسابات لمنح أو رفض الدخول.

**أبرز المواصفات التقنية:**

- **التعرف الفوري على الوجوه (Real-Time Recognition):** معالجة فائقة السرعة لإطارات الفيديو باستخدام نموذج `PyTorch` (`best_model.pth`).
    
- **اكتشاف الوجه (Face Detection):** باستخدام مصنفات Haar Cascade لاقتطاع الوجه المستهدف بدقة.
    
- **تحليل وتوليد البصمة الحيوية:** استخدام `TensorFlow / Keras` لتحويل آلاف الصور إلى متجهات دقيقة (Vectors).
    
- **واجهة مستخدم تفاعلية (Cyberpunk HUD):** مبنية بـ `PyQt6` وتتضمن:
    
    - **الشبكة الحيوية (Biometric Grid):** لتحديد أبعاد الوجه برمجياً.
        
    - **محاكاة الجسيمات (Particles & Ripples):** تأثيرات بصرية تفاعلية أثناء التعرف.
        
    - **شاشة أوامر حية (Live Terminal):** لعرض سجل النظام اللحظي للمستخدم.
        

---

## 📂 بنية المشروع ووظيفة كل ملف

هذه نبذة عن الملفات الأساسية بترتيب الأهمية العملياتية:

1. **`taking_photos.py`:** سكربت جمع البيانات؛ يلتقط 2500 صورة لوجه المستخدم ويحفظها في مجلد `dataset/`.
    
2. **`local_model_trining.py`:** سكربت التدريب المحلي لإنتاج نموذج `my_face_model.keras`.
    
3. **`embeddings.py`:** المحرك الذي يحول الصور إلى ملفات بصمة رقمية `.npy`.
    
4. **`main_app.py`:** قلب المشروع؛ الواجهة الرسومية التي تدير الكاميرا الحية وعملية المطابقة.
    
5. **`requirements.txt`:** قائمة المكتبات الضرورية لتشغيل النظام.
    

---
## 🚀 دليل التشغيل الشامل (Step-by-Step)

### الخطوة 1: إعداد بيئة العمل

قم بتثبيت المكتبات اللازمة عبر موجه الأوامر:

```
pip install -r requirements.txt
```

### الخطوة 2: تسجيل الوجوه (تجميع البيانات)

قم بتشغيل `taking_photos.py` وأدخل اسمك بالإنجليزية. سيقوم النظام بالتقاط 2500 صورة؛ تحرك ببطء وغيّر تعابير وجهك لضمان جودة التدريب.

### الخطوة 3: تدريب الموديل والترميز الرقمي

بعد جمع الصور، قم بتدريب الموديل ثم قم بتشغيل سكربت توليد البصمات:



```
python embeddings.py
```

### الخطوة 4: الإقلاع والتشغيل النهائي

الآن، قم بتشغيل واجهة القيادة المركزية:



```
python main_app.py
```

قف أمام الكاميرا، وسيقوم النظام بتحليل ملامحك ومطابقتها مع البصمات المسجلة فورياً.

---
<div align="center">

<h1 align="center">Face-Recognition-System 🧠👁️</h1>

<h3>Advanced Real-Time Biometric Face Recognition System</h3>

</div>

<p align="center">

An integrated project combining Deep Learning power with futuristic Sci-Fi HUD design for stunning, instant identity verification.

</p>

---

## 📖 Overview and Specifications

This project is a security and identification system based on Computer Vision and Deep Neural Networks. It detects faces via webcam, transforms facial features into numerical **Embeddings** using the **EfficientNetV2M** architecture, and compares them against registered users to grant or deny access.

**Key Technical Features:**

- **Real-Time Recognition:** Ultra-fast frame processing using `PyTorch` (`best_model.pth`).
    
- **Face Detection:** Precise cropping using Haar Cascade classifiers.
    
- **Biometric Generation:** Using `TensorFlow / Keras` to convert thousands of images into high-precision vectors.
    
- **Cyberpunk HUD Interface:** Built with `PyQt6`, featuring:
    
    - **Biometric Grid:** Dynamic facial mapping.
        
    - **Visual Particles:** Interactive VFX during the recognition process.
        
    - **Live Terminal:** Real-time system logs displayed on-screen.
        

---

## 📂 Project Structure

1. **`taking_photos.py`:** Data collection script; captures 2500 images per person.
    
2. **`local_model_trining.py`:** Local training script for creating the `.keras` model.
    
3. **`generate_all_embeddings.py`:** The engine that transforms images into `.npy` biometric fingerprints.
    
4. **`main_app.py`:** The main GUI core that handles live recognition and the visual interface.
    

---

## 🚀 Step-by-Step Guide

### Step 1: Environment Setup

Install dependencies:


```
pip install -r requirements.txt
```

### Step 2: Face Enrollment

Run `taking_photos.py`, enter your name, and let the camera capture your facial features. Change your expressions and move slightly for better results.

### Step 3: Biometric Generation

Transform your captured images into mathematical data:

```
python embeddings.py
```

### Step 4: Final Execution

Launch the system:

```
python main_app.py
```

---

## 🛠 Troubleshooting

- **Camera Error:** Ensure no other software is using your webcam.
    
- **Missing Model:** Make sure `best_model.pth` and `my_face_model.keras` are in the root directory.
    
- **Performance:** High-quality VFX might cause lag on systems without GPU acceleration.
    
