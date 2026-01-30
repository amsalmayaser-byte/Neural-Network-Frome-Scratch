import numpy as np

class Loss:
    """كلاس القاعدة لكل دوال الخسارة"""
    def forward(self, y_pred, y_true):
        raise NotImplementedError
        
    def backward(self):
        raise NotImplementedError

class MSE(Loss):
    """
    متوسط مربع الخطأ (Mean Squared Error)
    TODO: تنفيذ الـ forward والـ backward
    """
    def forward(self, y_pred, y_true):
        # حفظ القيم لأننا سنحتاجها في حساب المشتقة
        self.y_pred = y_pred
        self.y_true = y_true
        
        # المعادلة: متوسط (التوقع - الحقيقة) تربيع
        # np.mean تحسب المتوسط لكل العينات
        return np.mean((y_pred - y_true)**2)

    def backward(self):
        # مشتقة MSE بالنسبة للتوقع (y_pred)
        # رياضياً: 2 * (y_pred - y_true) مقسوماً على عدد العناصر
        n = self.y_true.shape[0] # نأخذ عدد العينات (N)
        return 2 * (self.y_pred - self.y_true) / n
    
