import plotly.graph_objects as go
import numpy as np
import pandas as pd


def plot_confusion_matrix(cm, class_names):
    """Create a plotly confusion matrix heatmap.
    
    Generates an interactive confusion matrix visualization with both absolute
    counts and normalized percentages for each cell.
    
    Args:
        cm (numpy.ndarray): Confusion matrix with shape (n_classes, n_classes)
        class_names (list): List of class labels for the axes
    
    Returns:
        plotly.graph_objects.Figure: Interactive confusion matrix heatmap
    """
    # Normalize the confusion matrix by row (true labels) to get percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create annotations for each cell showing both count and percentage
    annotations = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            annotations.append(
                dict(
                    x=j,  
                    y=i,  
                    text=f"{cm[i, j]}<br>({cm_normalized[i, j]:.2%})",  
                    # Use white text on dark cells, black text on light cells for readability
                    font=dict(color='white' if cm_normalized[i, j] > 0.5 else 'black'),
                    showarrow=False
                )
            )

    # Create the heatmap figure using normalized values
    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,  
        x=class_names,    
        y=class_names,    
        colorscale='Blues',  
        showscale=True    
    ))

    # Customize the layout with titles and annotations
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        annotations=annotations,  # Add cell annotations (counts and percentages)
        width=500,
        height=500
    )

    return fig


def plot_feature_importance(importances, feature_names, top_n=10):
    """Create a plotly bar chart for feature importances.
    
    Generates a horizontal bar chart showing the most important features
    from tree-based models (e.g., Random Forest, XGBoost, Decision Tree).
    
    Args:
        importances (array-like): Feature importance values from the model
        feature_names (list): List of feature names corresponding to importance values
        top_n (int, optional): Number of top features to display. Defaults to 10.
    
    Returns:
        plotly.graph_objects.Figure: Interactive horizontal bar chart of feature importances
    """
    # Create a DataFrame and sort features by importance in descending order
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    # Create horizontal bar chart with top N features
    fig = go.Figure(go.Bar(
        x=feature_importance_df['Importance'][:top_n],  
        y=feature_importance_df['Feature'][:top_n],     
        orientation='h'  
    ))

    # Customize the layout with titles and dimensions
    fig.update_layout(
        title=f'Top {top_n} Feature Importances',
        xaxis_title='Importance',  
        yaxis_title='Feature',     
        height=400  
    )

    return fig
