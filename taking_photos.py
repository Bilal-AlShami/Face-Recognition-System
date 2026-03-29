import cv2
import os
import time

# --- إعدادات ---
NUM_IMAGES = 2500  
# طلب الاسم من المستخدم
user_name = input("أدخل اسم الشخص الذي تريد تسجيله (باللغة الإنجليزية): ").strip()
if not user_name:
    print("يجب إدخال اسم صحيح!")
    exit()

SAVE_PATH = f"dataset/{user_name}"

# إنشاء المجلد
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    print(f"تم إنشاء مجلد جديد: {SAVE_PATH}")
else:
    print(f"المجلد {SAVE_PATH} موجود مسبقاً، سيتم إضافة الصور إليه.")

# تشغيل الكاميراا
cap = cv2.VideoCapture(2)

# تحميل مصنف الوجه (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print(f"--- بدء جمع الصور لـ: {user_name} ---")
print("حاول تغيير تعابير وجهك، وحرك رأسك قليلاً، وغير الإضاءة إن أمكن.")
print("اضغط 'q' للخروج مبكراً.")

count = 0
# نحسب عدد الصور الموجودة أصلاً لكي لا نكتب فوقها
existing_files = os.listdir(SAVE_PATH)
start_index = len(existing_files)

while count < NUM_IMAGES:
    ret, frame = cap.read()
    if not ret:
        print("فشل في قراءة الكاميرا")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # قص الوجه
        face_img = frame[y:y+h, x:x+w]
        
        # حفظ الصورة
        img_name = os.path.join(SAVE_PATH, f"{user_name}_{start_index + count}.jpg")
        cv2.imwrite(img_name, face_img)
        count += 1

        # رسم مربع
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        break 

    # عرض الفيديو
    cv2.putText(frame, f"Count: {count}/{NUM_IMAGES}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    time.sleep(0.05)

cap.release()
cv2.destroyAllWindows()
print(f"\nتم الانتهاء! تم حفظ {count} صورة في {SAVE_PATH}")