{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "44f9f044-cffb-4ba6-bb80-70642637be0c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\allib\\anaconda3\\Lib\\site-packages\\xgboost\\training.py:183: UserWarning: [11:51:30] WARNING: C:\\actions-runner\\_work\\xgboost\\xgboost\\src\\learner.cc:738: \n",
      "Parameters: { \"use_label_encoder\" } are not used.\n",
      "\n",
      "  bst.update(dtrain, iteration=i, fobj=obj)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "XGBoost Accuracy: 0.3\n",
      "\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "     group A       0.00      0.00      0.00        18\n",
      "     group B       0.22      0.21      0.21        38\n",
      "     group C       0.34      0.38      0.36        64\n",
      "     group D       0.34      0.42      0.38        52\n",
      "     group E       0.29      0.21      0.24        28\n",
      "\n",
      "    accuracy                           0.30       200\n",
      "   macro avg       0.24      0.24      0.24       200\n",
      "weighted avg       0.28      0.30      0.29       200\n",
      "\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "['scaler.pkl']"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# student_model.py\n",
    "\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "from sklearn.metrics import classification_report, accuracy_score\n",
    "from xgboost import XGBClassifier\n",
    "import joblib\n",
    "\n",
    "# load the dataset\n",
    "df = pd.read_csv(r'C:\\Users\\allib\\OneDrive\\Desktop\\MS Data Science\\ANA680\\Week2\\StudentsPerformance.csv')\n",
    "df = df[['math score', 'reading score', 'writing score', 'race/ethnicity']]\n",
    "\n",
    "# encode the target labels\n",
    "le = LabelEncoder()\n",
    "df['race/ethnicity_encoded'] = le.fit_transform(df['race/ethnicity'])\n",
    "\n",
    "X = df[['math score', 'reading score', 'writing score']]\n",
    "y = df['race/ethnicity_encoded']\n",
    "\n",
    "# train/test split (stratified)\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=42, stratify=y)\n",
    "\n",
    "# scale the features\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "# train XGBoost classifier\n",
    "model = XGBClassifier(\n",
    "    objective='multi:softprob',\n",
    "    num_class=5,\n",
    "    eval_metric='mlogloss',\n",
    "    use_label_encoder=False,\n",
    "    random_state=42\n",
    ")\n",
    "model.fit(X_train_scaled, y_train)\n",
    "\n",
    "# evaluate performance\n",
    "y_pred = model.predict(X_test_scaled)\n",
    "print(\"XGBoost Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "print(\"\\nClassification Report:\\n\", classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))\n",
    "\n",
    "# save my model, label encoder, and scaler\n",
    "joblib.dump(model, 'student_model.pkl')\n",
    "joblib.dump(le, 'label_encoder.pkl')\n",
    "joblib.dump(scaler, 'scaler.pkl')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "da4afdfc-a5b2-4c41-bf95-29db4583d3ba",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
