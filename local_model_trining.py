import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import os

# --- تهيئة الـ GPU (للتدريب المحلي) ---
# هذا الجزء يمنع TensorFlow من حجز كل ذاكرة الكرت دفعة واحدة ويسمح بنموها حسب الحاجة
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"تم اكتشاف GPU: {len(gpus)}")
    except RuntimeError as e:
        print(e)
else:
    print("لم يتم اكتشاف GPU، سيتم استخدام الـ CPU (سيكون التدريب أبطأ)")

# --- إعدادات ---
DATA_DIR = 'dataset'
IMG_SIZE = (380, 380) 

# إعدادات مرنة للتدريب (غير القيمة لـ True إذا كنت تدرب محلياً وجهازك متوسط القوة)
IS_LOCAL_TRAINING = True 

if IS_LOCAL_TRAINING:
    BATCH_SIZE = 4 # للتدريب المحلي، 4 آمنة جداً لغالبية الكروت
    EPOCHS = 15
else:
    BATCH_SIZE = 16 # لـ Google Colab T4/L4
    EPOCHS = 30

print("جاري تحميل الصور...")

# تحميل البيانات
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"Classes found: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- Data Augmentation ---
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip('horizontal'),
  layers.RandomRotation(0.3),
  layers.RandomZoom(0.3),
  layers.RandomBrightness(0.3),
  layers.RandomContrast(0.3),
])

# --- بناء الموديل (EfficientNetV2M) ---
# EfficientNetV2M يتوقع صور بقيم 0-255 داخلياً (يحتوي على طبقة Rescaling)
base_model = tf.keras.applications.EfficientNetV2M(input_shape=IMG_SIZE + (3,),
                                                  include_top=False,
                                                  weights='imagenet')

#  تجميد الطبقات للتدريب العميق
base_model.trainable = True 

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)

# ملاحظة: حذفنا layers.Rescaling لأن EfficientNet يقوم بذلك داخلياً

x = base_model(x, training=True) 
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)

# طبقة الـ Embedding 
embeddings = layers.Dense(256, activation=None, name='embedding_layer')(x) 

outputs = layers.Dense(len(class_names), activation='softmax')(embeddings)

model = models.Model(inputs, outputs)

optimizer = optimizers.Adam(learning_rate=1e-5) 

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

print("بدء تدريب EfficientNetV2M...")

checkpoint = tf.keras.callbacks.ModelCheckpoint("my_face_model.keras", 
                                                monitor='val_accuracy', 
                                                save_best_only=True, 
                                                mode='max', 
                                                verbose=1)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                                  patience=5, 
                                                  restore_best_weights=True)

history = model.fit(train_ds, 
                    validation_data=val_ds, 
                    epochs=EPOCHS,
                    callbacks=[checkpoint, early_stopping])

print("تم الانتهاء! EfficientNet ready.")