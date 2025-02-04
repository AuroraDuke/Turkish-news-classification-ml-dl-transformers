def model_scoring (y_test,y_pred ,results,model_name,vectorization_method):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import os
    import pandas as pd
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results['Model'].append(model_name)
    results['Vectorization Method'].append(vectorization_method)
    results['Accuracy'].append(accuracy)
    results['Precision'].append(precision)
    results['Recall'].append(recall)
    results['F1 Score'].append(f1)
    
    print(model_name + ' | ' + vectorization_method + ':' )
    print(f"F1 Score {f1:.4f} | Accuracy:{accuracy:.4f} | Precision:{precision:.4f} | Recall:{recall:.4f}  ")

    score_df = pd.DataFrame(results)
    #save measurement of model 
    os.makedirs('eval', exist_ok=True)
    score_file_path = f'eval/score.csv'
    score_df.to_csv(score_file_path, index=False)
    print(f"Results saved to {score_file_path}")