import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )


            models = {
                "Random Forest": RandomForestClassifier(class_weight='balanced'),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(class_weight='balanced'),
                "XGBClassifier": XGBClassifier(scale_pos_weight=3),
                
            }
            params={
                "Decision Tree": {
                   'criterion':['gini','entropy','log_loss'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                    
                },
                "Random Forest":{
                    'criterion':['gini','entropy','log_loss'],
                    "max_depth": [4, 6, 8],
                    "min_samples_split": [2, 5, 10],
                 
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256],
                    
                },
                "Gradient Boosting":{
                    # 'loss':['log_loss,'exponential'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    "max_depth": [3, 4, 5],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Logistic Regression":{},
                "XGBClassifier":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256],
                    "max_depth": [3, 5, 7],
                }
                
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            print("\n================ MODEL PERFORMANCE REPORT ================")
            for model, score in model_report.items():
                print(f"-> {model}: {score*100:.2f}%")
            print("==========================================================\n")
            
            print(f" {best_model_name} with a score of {best_model_score*100:.2f}%\n")

            if best_model_score<0.6:
                raise CustomException("No best model found",sys)
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            roc_auc= roc_auc_score(y_test, predicted)
            return roc_auc
            



            
        except Exception as e:
            raise CustomException(e,sys)