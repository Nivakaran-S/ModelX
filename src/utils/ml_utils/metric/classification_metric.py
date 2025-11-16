from src.entity.artifact_entity import ClassificationMetricArtifact
from src.exception.exception import DementiaException
from sklearn.metrics import f1_score,precision_score,recall_score, average_precision_score, roc_auc_score
import sys

def get_classification_score(y_true,y_pred)->ClassificationMetricArtifact:
    try:
            
        model_f1_score = f1_score(y_true, y_pred, average='binary')
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score=precision_score(y_true,y_pred)
        model_pr_auc_score=average_precision_score(y_true,y_pred)
        model_roc_auc_score=roc_auc_score(y_true,y_pred)

        classification_metric =  ClassificationMetricArtifact(f1_score=model_f1_score,
                    precision_score=model_precision_score, 
                    recall_score=model_recall_score,
                    pr_auc_score=model_pr_auc_score,
                    roc_auc_score=model_roc_auc_score 
                    )
        return classification_metric
    except Exception as e:
        raise DementiaException(e,sys)