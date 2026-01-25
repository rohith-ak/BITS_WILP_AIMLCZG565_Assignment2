import plotly.graph_objects as go
import numpy as np
import pandas as pd


def plot_confusion_matrix(cm, class_names):
    """Create a plotly confusion matrix heatmap."""
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    annotations = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=f"{cm[i, j]}<br>({cm_normalized[i, j]:.2%})",
                    font=dict(color='white' if cm_normalized[i, j] > 0.5 else 'black'),
                    showarrow=False
                )
            )

    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        showscale=True
    ))

    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        annotations=annotations,
        width=500,
        height=500
    )

    return fig


def plot_feature_importance(importances, feature_names, top_n=10):
    """Create a plotly bar chart for feature importances."""
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    fig = go.Figure(go.Bar(
        x=feature_importance_df['Importance'][:top_n],
        y=feature_importance_df['Feature'][:top_n],
        orientation='h'
    ))

    fig.update_layout(
        title=f'Top {top_n} Feature Importances',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=400
    )

    return fig
