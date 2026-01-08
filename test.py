import numpy as np
# استيراد المكونات التي برمجناها بأنفسنا
from nn_library.core.network import NeuralNetwork
from nn_library.core.layer import Dense
from nn_library.losses.functions import MSE
from nn_library.optimizers.optimizers import Adam, SGD

# 1. تجهيز البيانات (مشكلة XOR)
# المدخلات: (0,0), (0,1), (1,0), (1,1)
x_train = np.array([[0,0], [0,1], [1,0], [1,1]])
# المخرجات المتوقعة: 0, 1, 1, 0
y_train = np.array([[0], [1], [1], [0]])

# 2. بناء الشبكة العصبية
model = NeuralNetwork()
model.add(Dense(2, 3, activation='tanh')) # طبقة مخفية بـ 3 أعصاب
model.add(Dense(3, 1, activation='sigmoid')) # طبقة المخرجات

# 3. تحديد المحسن ودالة الخسارة
model.set_loss(MSE())
optimizer = Adam(learning_rate=0.1)

# 4. تدريب الشبكة
print("Starting Training...")
model.train(x_train, y_train, epochs=1000, optimizer=optimizer)

# 5. اختبار الشبكة بعد التدريب
print("\nTesting Results:")
for x in x_train:
    output = model.predict(x)
    print(f"Input: {x}, Predicted Output: {output.round()}, Raw: {output}")