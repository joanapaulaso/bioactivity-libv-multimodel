import pickle
import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    explained_variance_score,
)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.utils import resample
from sklearn.feature_selection import SelectFromModel
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from utils.file_operations import filedownload
from utils.descriptors import desc_calc
from utils.data_processing import remove_low_variance
from utils.visualization import model_graph_analysis


# Função para listar os modelos disponíveis
def list_models():
    if not os.path.exists("models"):
        os.makedirs("models")

    models = [
        f.removesuffix(".pkl") for f in os.listdir("models") if f.endswith(".pkl")
    ]
    return models


# Interface para seleção de algoritmos
def select_algorithms_ui():
    st.sidebar.header("Seleção de Algoritmos")
    selected_algorithms = []
    if st.sidebar.checkbox("Random Forest", value=True):
        selected_algorithms.append("Random Forest")
    if st.sidebar.checkbox("Gradient Boosting", value=True):
        selected_algorithms.append("Gradient Boosting")
    if st.sidebar.checkbox("Support Vector Machine", value=True):
        selected_algorithms.append("Support Vector Machine")
    if st.sidebar.checkbox("Rede Neural", value=True):
        selected_algorithms.append("Neural Network")
    return selected_algorithms


def weighted_score(r2, mse, rmse, mae, mape, explained_var):
    normalized_r2 = r2  # Já está na escala desejada (0-1)
    normalized_mse = 1 / (1 + mse)  # Inverter porque valores menores são melhores
    normalized_rmse = 1 / (1 + rmse)  # Inverter porque valores menores são melhores
    normalized_mae = 1 / (1 + mae)  # Inverter porque valores menores são melhores
    normalized_mape = 1 / (1 + mape)  # Inverter porque valores menores são melhores
    normalized_explained_var = explained_var  # Já está na escala desejada (0-1)

    weight_r2 = 0.3
    weight_mse = 0.2
    weight_rmse = 0.15
    weight_mae = 0.15
    weight_mape = 0.1
    weight_explained_var = 0.1

    total_score = (
        weight_r2 * normalized_r2
        + weight_mse * normalized_mse
        + weight_rmse * normalized_rmse
        + weight_mae * normalized_mae
        + weight_mape * normalized_mape
        + weight_explained_var * normalized_explained_var
    )

    return total_score


def detect_outliers(df, target_column, threshold=1.5):
    Q1 = df[target_column].quantile(0.25)
    Q3 = df[target_column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    outliers = df[(df[target_column] < lower_bound) | (df[target_column] > upper_bound)]
    return outliers, lower_bound, upper_bound


def detect_and_remove_outliers(molecules_processed):
    st.subheader("Outlier Detection and Removal")

    # Detect outliers
    outliers, lower_bound, upper_bound = detect_outliers(molecules_processed, "pIC50")

    st.write("Debug: Outliers Detected")
    st.write(f"Outliers:\n{outliers}")
    st.write(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

    # Create columns for the layout
    col1, col2, col3 = st.columns([1, 3, 1])  # Adjust column widths as needed

    # Plot distribution with outliers highlighted in the middle column
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))  # Set the figure size for a smaller plot
        sns.histplot(
            data=molecules_processed,
            x="pIC50",
            kde=True,
            ax=ax,
        )
        ax.axvline(lower_bound, color="r", linestyle="dashed", linewidth=2)
        ax.axvline(upper_bound, color="r", linestyle="dashed", linewidth=2)
        ax.set_title("Distribution of pIC50 with Outlier Bounds")
        st.pyplot(fig)

    st.write(f"Number of outliers detected: {len(outliers)}")
    st.write(
        f"Percentage of outliers: {len(outliers) / len(molecules_processed) * 100:.2f}%"
    )

    # Button to remove outliers
    if st.button("Remove Outliers"):
        st.write("Debug: Remove Outliers Button Clicked")
        try:
            molecules_processed = molecules_processed[
                (molecules_processed["pIC50"] >= lower_bound)
                & (molecules_processed["pIC50"] <= upper_bound)
            ]
            st.success(f"Outliers removed. {len(outliers)} outliers were excluded.")
            st.write("Debug: Outliers Removed Successfully")
            st.write(molecules_processed)

            # Return the cleaned dataset
            return molecules_processed

        except Exception as e:
            st.error(f"Error removing outliers: {str(e)}")
            st.write("Debug: Error Removing Outliers")
            st.write(e)

    # If no removal is done, return the original dataset
    return molecules_processed


def build_model(input_data, load_data, selected_model, selected_model_name):
    try:
        with open(selected_model, "rb") as f:
            model_info = pickle.load(f)

        model = model_info["model"]
        feature_names = model_info["feature_names"]
        scaler = model_info["scaler"]

        # Ensure input_data has the correct features in the correct order
        input_data = input_data.reindex(columns=feature_names, fill_value=0)

        # Transform the input data using the same scaler used during training
        input_data_transformed = scaler.transform(input_data)

        prediction = model.predict(input_data_transformed)
        st.header(
            f"**Saída das predições - Bioatividade em relação ao modelo {selected_model_name}**"
        )
        prediction_output = pd.Series(prediction, name="pIC50")
        molecule_id = pd.Series(load_data["ID"], name="id_molecula")
        molecule_name = pd.Series(load_data["Name"], name="nome_molecula")
        df = pd.concat([molecule_id, molecule_name, prediction_output], axis=1)
        st.write(df)
        st.markdown(filedownload(df), unsafe_allow_html=True)

        # If the model has feature_importances_, display them
        if hasattr(model, "feature_importances_"):
            feature_importance = pd.DataFrame(
                {"feature": feature_names, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)
            st.subheader("Feature Importances")
            st.write(feature_importance.head(10))  # Show top 10 features
        elif hasattr(model, "coef_"):  # For linear models
            feature_importance = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": abs(
                        model.coef_[0]
                    ),  # Use absolute values for importance
                }
            ).sort_values("importance", ascending=False)
            st.subheader("Feature Importances (Coefficient Magnitudes)")
            st.write(feature_importance.head(10))  # Show top 10 features
        else:
            st.info(
                "Feature importance information is not available for this model type."
            )

        return df
    except Exception as e:
        st.error(f"Erro ao construir o modelo: {str(e)}")
        return pd.DataFrame()


def generate_class_models(
    molecules_processed,
    variance_input,
    estimators_input,
    base_model_name,
    selected_models,
    replicates=10,
    test_size=0.2,
    min_samples=5,  # New parameter for minimum sample size
):
    if "compound_class" not in molecules_processed.columns:
        st.error(
            "Error: 'compound_class' column not found in the data. Please run the classification first."
        )
        return {}

    grouped = molecules_processed.groupby("compound_class")
    class_molecule_counts = {}

    for class_name, group in grouped:
        if len(group) < min_samples:
            st.warning(
                f"Skipping class '{class_name}' due to insufficient data (only {len(group)} molecules). Minimum required: {min_samples}"
            )
            continue

        st.subheader(f"Generating model for class: {class_name}")

        try:
            results, full_model_name = model_generation(
                group,
                variance_input,
                estimators_input,
                f"{base_model_name}_{class_name.replace(' ', '_')}",
                selected_models,
                replicates,
                test_size,
            )

            class_molecule_counts[class_name] = len(group)

            st.success(f"Model for class '{class_name}' generated successfully.")
            st.write(f"Model name: {full_model_name}")

            # Display metrics for the best model
            best_model_type = full_model_name.split("_")[-1]
            best_model_metrics = results[best_model_type]
            st.write(f"Best model: {best_model_type}")
            st.write(f"Test R² score: {best_model_metrics['test_r2']:.3f}")
            st.write(
                f"Mean CV R² score: {np.mean(best_model_metrics['cv_scores']):.3f}"
            )

        except Exception as e:
            st.error(f"Error generating model for class '{class_name}': {str(e)}")

    return class_molecule_counts


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

    st.subheader(f"Model Performance: {model_name}")
    st.write(f"Training R²: {train_r2:.4f}")
    st.write(f"Test R²: {test_r2:.4f}")
    st.write(
        f"Cross-validation R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}"
    )

    return {
        "model": model,
        "train_mse": train_mse,
        "train_r2": train_r2,
        "test_mse": test_mse,
        "test_r2": test_r2,
        "cv_scores": cv_scores,
    }


def interpret_metric(metric_name, value, threshold, better="lower"):
    """Interprete o valor da métrica e classifique como satisfatório ou não."""
    if better == "lower":
        if value <= threshold:
            return f"{metric_name}: {value:.4f} (Satisfatório)"
        else:
            return f"{metric_name}: {value:.4f} (Não Satisfatório)"
    elif better == "higher":
        if value >= threshold:
            return f"{metric_name}: {value:.4f} (Satisfatório)"
        else:
            return f"{metric_name}: {value:.4f} (Não Satisfatório)"


def weighted_score(r2, mse, rmse, mae, mape, explained_var):
    # Normalizar cada métrica
    normalized_r2 = r2  # Já está na escala desejada (0-1)
    normalized_mse = 1 / (1 + mse)  # Inverter porque valores menores são melhores
    normalized_rmse = 1 / (1 + rmse)  # Inverter porque valores menores são melhores
    normalized_mae = 1 / (1 + mae)  # Inverter porque valores menores são melhores
    normalized_mape = 1 / (1 + mape)  # Inverter porque valores menores são melhores
    normalized_explained_var = explained_var  # Já está na escala desejada (0-1)

    # Pesos para cada métrica
    weight_r2 = 0.3
    weight_mse = 0.2
    weight_rmse = 0.15
    weight_mae = 0.15
    weight_mape = 0.1
    weight_explained_var = 0.1

    # Cálculo da pontuação ponderada
    total_score = (
        weight_r2 * normalized_r2
        + weight_mse * normalized_mse
        + weight_rmse * normalized_rmse
        + weight_mae * normalized_mae
        + weight_mape * normalized_mape
        + weight_explained_var * normalized_explained_var
    )

    return total_score


def model_generation(
    molecules_processed,
    variance,
    estimators,
    base_model_name,
    selected_models,
    replicates=10,
    test_size=0.2,
):
    try:
        selection = ["canonical_smiles", "molecule_chembl_id"]
        df_final_selection = molecules_processed[selection]
        df_final_selection.to_csv("molecule.smi", sep="\t", index=False, header=False)

        desc_calc()
        df_fingerprints = pd.read_csv("descriptors_output.csv")
        df_fingerprints = df_fingerprints.drop(columns=["Name"])
        df_Y = molecules_processed["pIC50"]
        df_training = pd.concat([df_fingerprints, df_Y], axis=1)
        df_training = df_training.dropna()
        X = df_training.drop(["pIC50"], axis=1)
        Y = df_training["pIC50"]
        X = remove_low_variance(X, variance)

        selector = SelectFromModel(
            RandomForestRegressor(n_estimators=100, random_state=42), threshold="median"
        )
        X_selected = selector.fit_transform(X, Y)
        selected_feature_names = X.columns[selector.get_support()].tolist()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)

        models = {
            "Random Forest": {
                "model": RandomForestRegressor(random_state=42),
                "param_dist": {
                    "n_estimators": sp_randint(100, 300),
                    "max_depth": [10, 15, 20],
                    "min_samples_split": sp_randint(5, 15),
                    "min_samples_leaf": sp_randint(5, 10),
                    "max_features": sp_uniform(0.3, 0.5),
                },
            },
            "Gradient Boosting": {
                "model": GradientBoostingRegressor(random_state=42),
                "param_dist": {
                    "n_estimators": sp_randint(200, 500),
                    "max_depth": [3, 5, 7],
                    "min_samples_split": sp_randint(5, 15),
                    "min_samples_leaf": sp_randint(5, 10),
                    "learning_rate": sp_uniform(0.01, 0.1),
                },
            },
            "Support Vector Machine": {
                "model": SVR(),
                "param_dist": {
                    "C": sp_uniform(0.01, 10),
                    "gamma": sp_uniform(0.001, 0.01),
                    "kernel": ["rbf"],
                },
            },
            "Neural Network": {
                "model": MLPRegressor(max_iter=1000, random_state=42),
                "param_dist": {
                    "hidden_layer_sizes": [(50,), (50, 30), (30, 30)],
                    "activation": ["relu", "tanh"],
                    "alpha": sp_uniform(0.001, 0.01),
                    "learning_rate_init": sp_uniform(0.001, 0.01),
                },
            },
        }

        results = {}
        best_model = None
        best_score = -np.inf
        best_model_type = ""

        for model_type, config in models.items():
            if model_type not in selected_models:
                continue  # Skip models that are not selected

            st.write(f"Treinando e ajustando {model_type}...")

            model_scores = []

            for _ in range(replicates):
                X_train, X_test, Y_train, Y_test = train_test_split(
                    X_scaled, Y, test_size=test_size, random_state=42
                )

                model = config["model"]
                param_dist = config["param_dist"]

                random_search = RandomizedSearchCV(
                    model,
                    param_distributions=param_dist,
                    n_iter=20,
                    cv=5,
                    random_state=42,
                    n_jobs=-1,
                )
                random_search.fit(X_train, Y_train)

                tuned_model = random_search.best_estimator_

                Y_test_pred = tuned_model.predict(X_test)
                test_mse = mean_squared_error(Y_test, Y_test_pred)
                test_r2 = r2_score(Y_test, Y_test_pred)
                test_rmse = np.sqrt(test_mse)
                test_mae = mean_absolute_error(Y_test, Y_test_pred)
                test_mape = mean_absolute_percentage_error(Y_test, Y_test_pred)
                explained_var = explained_variance_score(Y_test, Y_test_pred)

                model_score = weighted_score(
                    test_r2, test_mse, test_rmse, test_mae, test_mape, explained_var
                )
                model_scores.append(model_score)

            mean_model_score = np.mean(model_scores)

            if mean_model_score > best_score:
                best_score = mean_model_score
                best_model = tuned_model
                best_model_type = model_type

            st.subheader(f"Métricas de Desempenho do {model_type}")
            st.write(
                f"Pontuação Ponderada do Modelo: {mean_model_score:.4f} (Média de {replicates} replicatas)"
            )

            st.write(
                interpret_metric(
                    "Coeficiente de Determinação no Teste (R²)",
                    test_r2,
                    threshold=0.7,
                    better="higher",
                )
            )
            st.write(
                interpret_metric(
                    "Erro Quadrático Médio no Teste (MSE)",
                    test_mse,
                    threshold=0.5,
                    better="lower",
                )
            )
            st.write(
                interpret_metric(
                    "Erro Médio Quadrático no Teste (RMSE)",
                    test_rmse,
                    threshold=0.5,
                    better="lower",
                )
            )
            st.write(
                interpret_metric(
                    "Erro Médio Absoluto no Teste (MAE)",
                    test_mae,
                    threshold=0.3,
                    better="lower",
                )
            )
            st.write(
                interpret_metric(
                    "Erro Médio Absoluto Percentual no Teste (MAPE)",
                    test_mape,
                    threshold=10,
                    better="lower",
                )
            )
            st.write(
                interpret_metric(
                    "Variância Explicada", explained_var, threshold=0.7, better="higher"
                )
            )

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
                ax.scatter(Y_test_pred, Y_test - Y_test_pred)
                ax.axhline(0, color="r", linestyle="--")
                ax.set_xlabel("Valores Previstos")
                ax.set_ylabel("Resíduos")
                ax.set_title(f"Plotagem de Resíduos do {model_type}")
                st.pyplot(fig)

            st.subheader(f"Resultados da Validação Cruzada do {model_type}")
            cv_scores = cross_val_score(
                tuned_model, X_train, Y_train, cv=5, scoring="r2"
            )
            st.write(
                f"Pontuação Média de R²: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores) * 2:.3f})"
            )

            if hasattr(tuned_model, "feature_importances_"):
                feature_importance = pd.DataFrame(
                    {
                        "feature": selected_feature_names,
                        "importance": tuned_model.feature_importances_,
                    }
                ).sort_values("importance", ascending=False)
                st.subheader(f"Top 10 Características Mais Importantes do {model_type}")
                st.write(feature_importance.head(10))

            n_iterations = 1000
            metrics = []
            for i in range(n_iterations):
                X_resample, Y_resample = resample(X_test, Y_test)
                Y_pred_resample = tuned_model.predict(X_resample)
                rmse = np.sqrt(mean_squared_error(Y_resample, Y_pred_resample))
                metrics.append(rmse)
            st.write(
                f"RMSE Bootstrapped: Média={np.mean(metrics):.4f}, Desvio Padrão={np.std(metrics):.4f}"
            )

        full_model_name = f"{base_model_name}_{best_model_type}"

        model_info = {
            "model": best_model,
            "feature_names": selected_feature_names,
            "scaler": scaler,
        }
        pickle.dump(model_info, open(f"models/{full_model_name}.pkl", "wb"))

        final_descriptor_list = pd.DataFrame({"feature": selected_feature_names})
        final_descriptor_list.to_csv(
            f"descriptor_lists/{full_model_name}_descriptor_list.csv", index=False
        )

        st.success(
            f"Melhor modelo '{full_model_name}' criado e avaliado! Agora disponível para previsões."
        )
        st.info(
            f"Lista de descritores final salva em 'descriptor_lists/{full_model_name}_descriptor_list.csv'"
        )

        return results, full_model_name  # Retorna dois valores agora

    except Exception as e:
        st.error(f"Erro na geração do modelo: {str(e)}")
        raise


def interpret_results(test_mse, test_r2, cv_mean, cv_std):
    # Definir limites para interpretação
    r2_threshold_excellent = 0.8
    r2_threshold_good = 0.6
    r2_threshold_moderate = 0.4
    r2_difference_threshold = 0.1

    # Determinar se o desempenho é satisfatório
    if test_r2 >= r2_threshold_excellent:
        performance = "excelente"
    elif test_r2 >= r2_threshold_good:
        performance = "bom"
    elif test_r2 >= r2_threshold_moderate:
        performance = "moderado"
    else:
        performance = "insatisfatório"

    # Verificar overfitting
    overfitting = cv_mean - test_r2 > r2_difference_threshold

    st.write(
        f"""
    ## Interpretando os Resultados

    1. Erro Quadrático Médio (MSE) no Conjunto de Teste: {test_mse:.4f}
       - Representa a diferença quadrática média entre os valores previstos e reais de pIC50.
       - Valores mais baixos indicam melhor desempenho.

    2. Pontuação R² (Coeficiente de Determinação) no Conjunto de Teste: {test_r2:.4f}
       - Indica a proporção da variância na variável alvo que é previsível a partir das características.
       - Intervalo: 0 a 1, onde 1 é uma previsão perfeita e 0 significa que o modelo está apenas prevendo a média.
       - Seu modelo explica {test_r2*100:.2f}% da variância nos dados de teste.

    3. Pontuação R² da Validação Cruzada: {cv_mean:.3f} (+/- {cv_std * 2:.3f})
       - Esta é a pontuação R² média da validação cruzada 5-fold nos dados de treinamento.
       - O valor entre parênteses representa o intervalo de confiança de 95%.
       - Isso fornece uma estimativa mais robusta do desempenho do modelo e sua consistência.

    Interpretação:
    - O desempenho do modelo é considerado {performance} para modelos QSAR.
    - Uma pontuação R² de {test_r2:.4f} sugere que as previsões do modelo explicam {test_r2*100:.2f}% da variabilidade na variável alvo.
    """
    )

    if overfitting:
        st.write(
            """
    - O R² do Teste é significativamente menor que o R² da Validação Cruzada, o que pode indicar overfitting.
      Isso significa que o modelo pode estar se ajustando demais aos dados de treinamento e não generalizando bem para novos dados.
            """
        )

    if performance == "insatisfatório":
        st.write(
            """
    - A baixa pontuação R² pode indicar que:
      a) As características podem não ser fortemente preditivas da variável alvo.
      b) A relação entre as características e o alvo pode ser altamente não-linear.
      c) Pode haver muito ruído nos dados.

    Próximos Passos:
    1. Como o desempenho é insatisfatório para um modelo QSAR, considere:
       - Revisar a seleção de descritores moleculares e possivelmente incluir ou gerar novos descritores
       - Investigar a qualidade dos dados experimentais e remover outliers, se apropriado
       - Experimentar diferentes algoritmos de aprendizado de máquina, incluindo métodos não-lineares
       - Se possível, aumentar o tamanho do conjunto de dados
            """
        )
    elif performance == "moderado":
        st.write(
            """
    Próximos Passos:
    1. O desempenho do modelo é moderado para QSAR. Você pode:
       - Usar o modelo para triagem inicial de compostos, mas com cautela
       - Tentar melhorar o modelo através de seleção de características ou engenharia de características
       - Considerar ensemble methods ou técnicas de boosting para melhorar o desempenho
       - Validar o modelo com um conjunto de teste externo, se disponível
            """
        )
    elif performance == "bom":
        st.write(
            """
    Próximos Passos:
    1. O desempenho do modelo é bom para QSAR. Você pode:
       - Usar o modelo para previsões de bioatividade com confiança moderada
       - Realizar validação externa adicional para confirmar o desempenho
       - Considerar o uso do modelo em workflows de descoberta de medicamentos
       - Continuar refinando o modelo, possivelmente com dados adicionais ou técnicas avançadas
            """
        )
    else:  # excelente
        st.write(
            """
    Próximos Passos:
    1. O desempenho do modelo é excelente para QSAR. Você pode:
       - Confiar no modelo para fazer previsões precisas de bioatividade
       - Implementar o modelo em pipelines de descoberta de medicamentos
       - Publicar os resultados, pois este nível de desempenho é notável para modelos QSAR
       - Usar o modelo como benchmark para futuros desenvolvimentos
            """
        )
