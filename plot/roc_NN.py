def plot_multi_roc_NN(X_test, y_test, model, batch_size=32,torch_type="float32"):
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import LabelBinarizer
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    """
    ROC eğrisi çizmek için:
    - Modelden çıkan tahmin olasılıklarını batch size kullanarak hesaplar.
    - ROC ve AUC değerlerini çizer.
    """
    # Cihazı belirle (GPU varsa kullan, yoksa CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Modeli GPU'ya taşı
    model = model.to(device)
    model.eval()  # Modeli değerlendirme moduna al

    # Binarize etiketler
    label_binarizer = LabelBinarizer()
    y_test_bin = label_binarizer.fit_transform(y_test)

    # TensorDataset ve DataLoader ile test verisini hazırlayın
    if torch_type == "float32":
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_bin, dtype=torch.float32)
    if torch_type == "long":
        X_test_tensor = torch.tensor(X_test, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test_bin, dtype=torch.long)
        
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Modelden tahmin olasılıklarını topla
    all_probabilities = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)  # Tahmin al
            probabilities = F.softmax(logits, dim=1).cpu().numpy()  # Softmax ve CPU'ya taşı
            all_probabilities.append(probabilities)

    # Tahminleri birleştir
    all_probabilities = np.vstack(all_probabilities)

    # ROC eğrisi hesapla
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(6, 5))

    # Multi-class ROC için her sınıfın ROC eğrisini çizin
    for i in range(len(label_binarizer.classes_)):
        color = 'aqua' if i % 2 == 0 else 'darkorange'
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    # Rastgele tahmin çizgisi
    plt.plot([0, 1], [0, 1], 'r--', lw=1, label='Random Guess')

    # Grafik ayarları
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multiclass ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
