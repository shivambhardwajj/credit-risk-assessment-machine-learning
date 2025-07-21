import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')


# CHAID Implementation (Simplified)
class SimpleChaidNode:
    def __init__(self, feature=None, threshold=None, samples=None, prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.samples = samples
        self.prediction = prediction
        self.left = None
        self.right = None
        self.is_leaf = prediction is not None


class SimpleChaidTree:
    def __init__(self, max_depth=5, min_samples_split=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _chi_square_test(self, X, y, feature_idx):
        """Simplified chi-square test for feature selection"""
        feature_values = X[:, feature_idx]
        unique_vals = np.unique(feature_values)
        if len(unique_vals) < 2:
            return 0

        chi_square = 0
        for val in unique_vals:
            mask = feature_values == val
            if np.sum(mask) > 0:
                observed = np.bincount(y[mask], minlength=2)
                expected = np.sum(mask) * np.bincount(y, minlength=2) / len(y)
                expected = np.maximum(expected, 1e-10)  # Avoid division by zero
                chi_square += np.sum((observed - expected) ** 2 / expected)

        return chi_square

    def _find_best_split(self, X, y):
        best_feature = None
        best_chi_square = 0
        best_threshold = None

        for feature_idx in range(X.shape[1]):
            chi_square = self._chi_square_test(X, y, feature_idx)
            if chi_square > best_chi_square:
                best_chi_square = chi_square
                best_feature = feature_idx
                # Use median as threshold for continuous variables
                best_threshold = np.median(X[:, feature_idx])

        return best_feature, best_threshold, best_chi_square

    def _build_tree(self, X, y, depth=0):
        # Check stopping conditions
        if depth >= self.max_depth or len(X) < self.min_samples_split or len(np.unique(y)) == 1:
            prediction = np.argmax(np.bincount(y))
            return SimpleChaidNode(samples=len(X), prediction=prediction)

        # Find best split
        best_feature, best_threshold, chi_square = self._find_best_split(X, y)

        if best_feature is None or chi_square < 3.841:  # Chi-square critical value at 0.05
            prediction = np.argmax(np.bincount(y))
            return SimpleChaidNode(samples=len(X), prediction=prediction)

        # Create node
        node = SimpleChaidNode(feature=best_feature, threshold=best_threshold, samples=len(X))

        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        if np.sum(left_mask) > 0:
            node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        if np.sum(right_mask) > 0:
            node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def fit(self, X, y):
        self.root = self._build_tree(X, y)
        return self

    def _predict_sample(self, x, node):
        if node.is_leaf:
            return node.prediction

        if x[node.feature] <= node.threshold:
            if node.left:
                return self._predict_sample(x, node.left)
            else:
                return node.prediction if node.prediction is not None else 0
        else:
            if node.right:
                return self._predict_sample(x, node.right)
            else:
                return node.prediction if node.prediction is not None else 0

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])


def generate_sample_data(n_samples=1000):
    """Generate synthetic credit risk data"""
    st.write("📊 **Step 1: Data Generation**")
    st.write("Generating synthetic credit loan data with the following features:")
    st.write("- Age, Income, Credit Score, Debt-to-Income Ratio, Loan Amount, Employment Length")

    np.random.seed(42)

    # Generate correlated features
    age = np.random.normal(40, 12, n_samples).clip(18, 80)
    income = np.random.lognormal(10.5, 0.5, n_samples).clip(20000, 200000)
    credit_score = np.random.normal(650, 100, n_samples).clip(300, 850)
    debt_to_income = np.random.beta(2, 5, n_samples) * 0.6  # Max 60% DTI
    loan_amount = np.random.lognormal(9.5, 0.8, n_samples).clip(1000, 50000)
    employment_length = np.random.exponential(3, n_samples).clip(0, 20)

    # Create target variable (default risk)
    # Higher risk for: low credit score, high DTI, low income relative to loan
    risk_score = (
            -0.01 * (credit_score - 300) +  # Lower credit score = higher risk
            2 * debt_to_income +  # Higher DTI = higher risk
            0.05 * (loan_amount / income) +  # Higher loan-to-income = higher risk
            -0.02 * employment_length +  # Less employment = higher risk
            np.random.normal(0, 0.5, n_samples)  # Random noise
    )

    # Convert to binary classification
    default_risk = (risk_score > np.percentile(risk_score, 80)).astype(int)

    data = pd.DataFrame({
        'age': age.round(0),
        'income': income.round(0),
        'credit_score': credit_score.round(0),
        'debt_to_income_ratio': debt_to_income.round(3),
        'loan_amount': loan_amount.round(0),
        'employment_length': employment_length.round(1),
        'default_risk': default_risk
    })

    st.write(f"✅ Generated {n_samples} samples with {len(data.columns) - 1} features")
    st.write("**Sample Data Preview:**")
    st.dataframe(data.head(10))

    return data


def preprocess_data(data):
    """Preprocess the data"""
    st.write("🔧 **Step 2: Data Preprocessing**")
    st.write("Performing data preprocessing steps:")

    # Handle missing values
    missing_count = data.isnull().sum().sum()
    st.write(f"- Missing values: {missing_count}")

    # Basic statistics
    st.write("- Computing basic statistics and checking for outliers")
    st.dataframe(data.describe())

    # Check class distribution
    class_dist = data['default_risk'].value_counts()
    st.write("**Class Distribution:**")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"- No Default (0): {class_dist[0]} ({class_dist[0] / len(data) * 100:.1f}%)")
        st.write(f"- Default (1): {class_dist[1]} ({class_dist[1] / len(data) * 100:.1f}%)")

    with col2:
        fig = px.pie(values=class_dist.values, names=['No Default', 'Default'],
                     title="Class Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # Feature correlation
    st.write("**Feature Correlation Matrix:**")
    corr_matrix = data.corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                    title="Feature Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

    st.write("✅ Data preprocessing completed")

    return data


def prepare_features(data):
    """Prepare features for modeling"""
    st.write("⚙️ **Step 3: Feature Preparation**")
    st.write("Preparing features for machine learning models:")

    # Separate features and target
    X = data.drop('default_risk', axis=1)
    y = data['default_risk']

    st.write(f"- Features shape: {X.shape}")
    st.write(f"- Target shape: {y.shape}")
    st.write(f"- Feature names: {list(X.columns)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    st.write(f"- Training set: {X_train.shape[0]} samples")
    st.write(f"- Test set: {X_test.shape[0]} samples")

    # Scale features for some models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.write("✅ Feature preparation completed")

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """Train different models"""
    st.write("🤖 **Step 4: Model Training**")
    st.write("Training multiple models for comparison:")

    models = {}
    model_scores = {}

    # 1. CHAID Decision Tree
    st.write("**1. Training CHAID Decision Tree:**")
    st.write("- Using chi-square test for feature selection")
    st.write("- Maximum depth: 5, Minimum samples per split: 20")

    chaid_model = SimpleChaidTree(max_depth=5, min_samples_split=20)
    chaid_model.fit(X_train.values, y_train.values)
    chaid_pred = chaid_model.predict(X_test.values)
    chaid_auc = roc_auc_score(y_test, chaid_pred)

    models['CHAID'] = chaid_model
    model_scores['CHAID'] = chaid_auc
    st.write(f"- CHAID AUC Score: {chaid_auc:.3f}")

    # 2. Random Forest
    st.write("**2. Training Random Forest:**")
    st.write("- Number of trees: 100")
    st.write("- Using all features with bootstrap sampling")

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_pred)

    models['Random Forest'] = rf_model
    model_scores['Random Forest'] = rf_auc
    st.write(f"- Random Forest AUC Score: {rf_auc:.3f}")

    # 3. Logistic Regression
    st.write("**3. Training Logistic Regression:**")
    st.write("- Using L2 regularization")
    st.write("- Features are standardized for optimal performance")

    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_pred)

    models['Logistic Regression'] = lr_model
    model_scores['Logistic Regression'] = lr_auc
    st.write(f"- Logistic Regression AUC Score: {lr_auc:.3f}")

    # Model comparison
    st.write("**Model Performance Summary:**")
    results_df = pd.DataFrame({
        'Model': list(model_scores.keys()),
        'AUC Score': list(model_scores.values())
    }).sort_values('AUC Score', ascending=False)

    st.dataframe(results_df)

    # Best model
    best_model_name = results_df.iloc[0]['Model']
    st.write(f"🏆 **Best Model: {best_model_name}**")

    st.write("✅ Model training completed")

    return models, model_scores, best_model_name


def analyze_feature_importance(models, X_train):
    """Analyze feature importance"""
    st.write("📈 **Step 5: Feature Importance Analysis**")
    st.write("Analyzing which features are most important for predictions:")

    # Random Forest feature importance
    if 'Random Forest' in models:
        rf_importance = models['Random Forest'].feature_importances_
        importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': rf_importance
        }).sort_values('Importance', ascending=False)

        st.write("**Random Forest Feature Importance:**")
        fig = px.bar(importance_df, x='Importance', y='Feature',
                     orientation='h', title="Feature Importance (Random Forest)")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(importance_df)

    # Logistic Regression coefficients
    if 'Logistic Regression' in models:
        lr_coef = models['Logistic Regression'].coef_[0]
        coef_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Coefficient': lr_coef,
            'Abs_Coefficient': np.abs(lr_coef)
        }).sort_values('Abs_Coefficient', ascending=False)

        st.write("**Logistic Regression Coefficients:**")
        fig = px.bar(coef_df, x='Coefficient', y='Feature',
                     orientation='h', title="Feature Coefficients (Logistic Regression)")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(coef_df)

    st.write("✅ Feature importance analysis completed")


def model_evaluation(models, X_test, y_test, X_test_scaled):
    """Detailed model evaluation"""
    st.write("📊 **Step 6: Model Evaluation**")
    st.write("Detailed evaluation of model performance:")

    evaluation_results = {}

    for model_name, model in models.items():
        st.write(f"**Evaluating {model_name}:**")

        # Make predictions
        if model_name == 'CHAID':
            y_pred = model.predict(X_test.values)
            y_pred_proba = y_pred  # CHAID returns binary predictions
        elif model_name == 'Logistic Regression':
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        if model_name != 'CHAID':  # For probabilistic predictions
            auc_score = roc_auc_score(y_test, y_pred_proba)
        else:
            auc_score = roc_auc_score(y_test, y_pred)

        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)

        evaluation_results[model_name] = {
            'auc': auc_score,
            'accuracy': class_report['accuracy'],
            'precision': class_report['1']['precision'],
            'recall': class_report['1']['recall'],
            'f1_score': class_report['1']['f1-score']
        }

        st.write(f"- AUC Score: {auc_score:.3f}")
        st.write(f"- Accuracy: {class_report['accuracy']:.3f}")
        st.write(f"- Precision (Default): {class_report['1']['precision']:.3f}")
        st.write(f"- Recall (Default): {class_report['1']['recall']:.3f}")
        st.write(f"- F1-Score (Default): {class_report['1']['f1-score']:.3f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, aspect="auto",
                        title=f"Confusion Matrix - {model_name}",
                        labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(fig, use_container_width=True)

    # Summary comparison
    st.write("**Model Performance Comparison:**")
    comparison_df = pd.DataFrame(evaluation_results).T
    st.dataframe(comparison_df.round(3))

    st.write("✅ Model evaluation completed")

    return evaluation_results


def prediction_interface(models, scaler, X_train):
    """Interactive prediction interface"""
    st.write("🎯 **Step 7: Make Predictions**")
    st.write("Enter customer information to assess credit risk:")

    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=80, value=35)
            income = st.number_input("Annual Income ($)", min_value=20000, max_value=200000, value=50000)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

        with col2:
            debt_to_income = st.slider("Debt-to-Income Ratio", min_value=0.0, max_value=0.6, value=0.3, step=0.01)
            loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=50000, value=20000)
            employment_length = st.number_input("Employment Length (years)", min_value=0.0, max_value=20.0, value=3.0)

        submitted = st.form_submit_button("Assess Credit Risk")

    if submitted:
        # Prepare input data
        input_data = np.array([[age, income, credit_score, debt_to_income, loan_amount, employment_length]])
        input_scaled = scaler.transform(input_data)

        st.write("**Processing Input:**")
        st.write(f"- Customer Age: {age}")
        st.write(f"- Annual Income: ${income:,}")
        st.write(f"- Credit Score: {credit_score}")
        st.write(f"- Debt-to-Income Ratio: {debt_to_income:.2%}")
        st.write(f"- Loan Amount: ${loan_amount:,}")
        st.write(f"- Employment Length: {employment_length} years")

        # Make predictions with each model
        st.write("**Model Predictions:**")

        predictions = {}

        for model_name, model in models.items():
            if model_name == 'CHAID':
                pred = model.predict(input_data)[0]
                pred_proba = pred  # Binary prediction
                risk_score = pred * 100
            elif model_name == 'Logistic Regression':
                pred = model.predict(input_scaled)[0]
                pred_proba = model.predict_proba(input_scaled)[0, 1]
                risk_score = pred_proba * 100
            else:
                pred = model.predict(input_data)[0]
                pred_proba = model.predict_proba(input_data)[0, 1]
                risk_score = pred_proba * 100

            predictions[model_name] = {
                'prediction': pred,
                'risk_score': risk_score
            }

            risk_level = "HIGH" if pred == 1 else "LOW"
            st.write(f"- **{model_name}**: {risk_level} Risk ({risk_score:.1f}%)")

        # Overall assessment
        avg_risk = np.mean([p['risk_score'] for p in predictions.values()])
        overall_risk = "HIGH" if avg_risk > 50 else "MODERATE" if avg_risk > 30 else "LOW"

        st.write("**Overall Assessment:**")
        st.write(f"- Average Risk Score: {avg_risk:.1f}%")
        st.write(f"- Overall Risk Level: **{overall_risk}**")

        # Risk factors analysis
        st.write("**Risk Factor Analysis:**")
        if credit_score < 600:
            st.write("⚠️ Low credit score increases risk")
        if debt_to_income > 0.4:
            st.write("⚠️ High debt-to-income ratio increases risk")
        if loan_amount / income > 0.3:
            st.write("⚠️ High loan-to-income ratio increases risk")
        if employment_length < 2:
            st.write("⚠️ Short employment history increases risk")

        if overall_risk == "LOW":
            st.success("✅ Customer appears to be a good candidate for loan approval")
        elif overall_risk == "MODERATE":
            st.warning("⚠️ Customer has moderate risk - consider additional verification")
        else:
            st.error("❌ Customer has high default risk - loan approval not recommended")


def main():
    st.title("Credit Loan Risk Assessment System by Shivam Bhardwaj")
    st.write("A transparent machine learning application for credit risk evaluation using CHAID and other models")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    run_analysis = st.sidebar.button("Run Full Analysis")

    if run_analysis:
        # Step-by-step process
        data = generate_sample_data()

        data = preprocess_data(data)

        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = prepare_features(data)

        models, model_scores, best_model = train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)

        analyze_feature_importance(models, X_train)

        evaluation_results = model_evaluation(models, X_test, y_test, X_test_scaled)

        # Store in session state for prediction interface
        st.session_state['models'] = models
        st.session_state['scaler'] = scaler
        st.session_state['X_train'] = X_train

        st.success("✅ Analysis completed! You can now make predictions below.")

    # Prediction interface (always available if models are trained)
    if 'models' in st.session_state:
        st.markdown("---")
        prediction_interface(st.session_state['models'], st.session_state['scaler'], st.session_state['X_train'])
    else:
        st.info("Click 'Run Full Analysis' in the sidebar to start the credit risk assessment process.")

    # Footer
    st.markdown("---")
    st.write("**Process Summary:**")
    st.write("This application demonstrates a complete machine learning pipeline for credit risk assessment:")
    st.write("1. **Data Generation**: Synthetic credit data with realistic correlations")
    st.write("2. **Preprocessing**: Data cleaning, statistics, and visualization")
    st.write("3. **Feature Preparation**: Train/test split and scaling")
    st.write("4. **Model Training**: CHAID, Random Forest, and Logistic Regression")
    st.write("5. **Feature Analysis**: Understanding model decision factors")
    st.write("6. **Evaluation**: Comprehensive performance metrics")
    st.write("7. **Prediction**: Interactive risk assessment interface")


if __name__ == "__main__":
    main()