def plot_multi_roc_bert(y_test, raw_outputs, n_classes):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    """
    Plots multi-class ROC-AUC curves and calculates AUC scores for each class.

    Parameters:
    y_test : Ground truth labels (in integer format).
    raw_outputs : Predicted probabilities from the model (for each class).
    n_classes : Number of classes.
    """
    # Binarize the labels for multi-class ROC
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

    # Variables for storing ROC curves and AUC scores
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Set the figure size
    plt.figure(figsize=(6, 5))

    # Calculate ROC curve and AUC score for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], raw_outputs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

    # Calculate micro-average ROC curve
    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), raw_outputs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, linestyle="--", color="gray", label=f"Micro-average (AUC = {roc_auc_micro:.2f})")

    # Add the random guess line
    plt.plot([0, 1], [0, 1], 'r--', lw=1, label='Random Guess')

    # Add labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    # Return AUC scores for each class
    return roc_auc
