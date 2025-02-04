def plot_multi_roc(X_test, y_test ,model):
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    """
    This function works with binary classification by:
    - Applying label binarization.
    - Calculating ROC and AUC for each class.
    - Visualizing the ROC curve and AUC values.
    """
    print(f"y_test unique value :{np.unique(y_test)}")
    # Binarize the labels
    y_test_bin = label_binarize(y_test, classes = np.unique(y_test))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict() ; tpr = dict() ;    roc_auc = dict()
     
    plt.figure(figsize=(6, 5)) 
    #multi class calculate 
    for i in range(len(set(y_test))):   #model.classes_.shape[0]
        if i%2==0:
            color='aqua'
        else:
            color='darkorange'
        temp =  model.predict_proba(X_test)[:, i]
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i] , temp)
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{model.__class__.__name__} - Class {i} (AUC = {roc_auc[i]:.2f})')
    
    # Plot random guess line
    plt.plot([0, 1], [0, 1], 'r--', lw=1, label='Random Guess')
    
    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multiclass ROC Curve with {model.__class__.__name__} ')
    plt.legend(loc="lower right")
    plt.show()