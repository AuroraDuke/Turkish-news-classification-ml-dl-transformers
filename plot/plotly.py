def plotly_eval(results):
    import plotly.express as px
    import pandas as pd
    scores = pd.DataFrame(results)
    scores['Model and Vectorization'] = scores['Model'] + " (" + scores['Vectorization Method'] + ")"
    fig = px.bar(
        scores.melt(
            id_vars='Model and Vectorization', 
            value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score'],  # Performans metriklerini seç
            var_name='Metric', 
            value_name='Value'
        ),
        x='Model and Vectorization',
        y='Value',
        color='Metric',
        barmode='group',
        title='Model Performance Metrics'
    )
    fig.update_layout(
        yaxis=dict(
            range=[0, 1],  # Y ekseni 0 ile 1 arasında olacak
            tickformat=".4f"  # Y ekseni etiketleri ondalık olacak
        )
    )
    fig.show()
