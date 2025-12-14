import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
VAL_DIR = os.path.join(BASE_DIR, "data", "val")

def count_files(directory):
    if not os.path.exists(directory):
        return {}
    counts = {}
    for cls in os.listdir(directory):
        cls_path = os.path.join(directory, cls)
        if os.path.isdir(cls_path):
            cnt = len([f for f in os.listdir(cls_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))])
            counts[cls] = cnt
    return counts

def check_health():
    print("🏥 CHEQUEO DE SALUD DE DATOS")
    
    train_counts = count_files(TRAIN_DIR)
    val_counts = count_files(VAL_DIR)
    
    print("\n📂 ENTRENAMIENTO:")
    for cls, count in train_counts.items():
        print(f"   - {cls}: {count}")
        
    print("\n📂 VALIDACIÓN:")
    for cls, count in val_counts.items():
        print(f"   - {cls}: {count}")

    # Verificar si alguna clase está vacía o crítica
    all_counts = list(train_counts.values()) + list(val_counts.values())
    if any(c < 10 for c in all_counts):
        print("\n❌ ALERTA CRÍTICA: Algunas clases están casi vacías.")
    else:
        print("\n✅ Cantidades parecen operativas (aunque revisa si el balance cambió mucho).")

if __name__ == "__main__":
    check_health()