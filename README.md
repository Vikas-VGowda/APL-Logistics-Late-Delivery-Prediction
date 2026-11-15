# APL Logistics – Late Delivery Risk Prediction

Streamlit dashboard for the machine-learning project **Machine Learning-Based Late Delivery Risk Prediction in Global Supply Chain Operations**.

The app downloads its trained model artifacts from Hugging Face and runs predictions on a CSV uploaded by the user.

## Dashboard modules

1. **Delay Risk Overview** – predicted risk distribution and probability histogram.
2. **Order-Level Risk Prediction** – probability and risk category for a selected order.
3. **Region & Mode Risk Analysis** – average predicted risk by region, shipping mode, and customer segment.
4. **Operations Action Panel** – high-risk order queue and CSV export.
5. **Model Insights** – top global feature importances when supported by the saved estimator.

## Repository structure

Keep these application files in your GitHub repository:

```text
APL-Logistics-Late-Delivery-Prediction/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

The trained model files are hosted separately in the Hugging Face model repository:

- `Vvg0007/apl-logistics-model/best_late_delivery_model.joblib`
- `Vvg0007/apl-logistics-model/feature_columns.joblib`

The app currently expects that Hugging Face repository to be **public**. Do not make it public if you do not have permission to publish the model. For a private repository, use a Hugging Face read token stored in Streamlit Secrets rather than putting the token in source code.

## Run locally

Use a Python environment compatible with the versions used to train the model in Google Colab.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, `README.md`, and `.gitignore` to GitHub.
2. Do **not** upload the original customer dataset or the model files to a public GitHub repository.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Select **Create app**, then choose your repository and branch.
5. Set the main file path to `app.py`.
6. Deploy.

On startup, the app downloads the two model artifacts from Hugging Face and caches them.

## Input data

Upload the project CSV through the app's sidebar. The CSV must contain every input column listed in `feature_columns.joblib` and have compatible data types with the training data.

The app uses the feature columns in the exact order saved during training. If any required feature is missing, the app displays an error before prediction.

## Risk categories

The dashboard applies these display thresholds to the model's predicted probability:

- **Low Risk:** probability below 0.30.
- **Medium Risk:** probability from 0.30 up to (but not including) the selected high-risk threshold.
- **High Risk:** probability at or above the selected high-risk threshold.

The high-risk threshold can be adjusted in the sidebar. These categories are operational display labels, not calibrated guarantees of delivery outcomes.

## Important model compatibility note

Joblib/pickle model files are sensitive to Python and package versions. The `requirements.txt` above lists the required packages but does not pin versions because the exact versions from the Colab training runtime have not yet been supplied.

Before deployment, run this in the Colab notebook that trained the model and use the printed versions to pin the corresponding dependencies in `requirements.txt`:

```python
import sys, sklearn, xgboost, numpy, pandas, joblib
print("Python:", sys.version)
print("scikit-learn:", sklearn.__version__)
print("XGBoost:", xgboost.__version__)
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Joblib:", joblib.__version__)
```

Do not assume that a model trained with one scikit-learn/XGBoost version will load correctly with another.

## Privacy and responsible use

The source dataset may contain customer names, street addresses, postal codes, and commercially sensitive information.

- Do not commit the raw dataset to a public GitHub repository.
- Upload only data you are authorized to process.
- Do not publish customer-level personal information in screenshots, reports, or exports.
- The exported high-risk queue contains selected operational columns rather than every source column.
- Predictions are estimates for decision support, not confirmed delivery outcomes. Operations personnel should verify risk signals before taking action.
