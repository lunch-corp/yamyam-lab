import pandas as pd
import torch

from explanation.explain import ExplainableAI


def main():
    # Load your trained model
    model = torch.load("path_to_your_model.pt")

    # Load user and diner features
    user_features = pd.read_csv("path_to_user_features.csv")
    diner_features = pd.read_csv("path_to_diner_features.csv")

    # Initialize explainable AI
    explainer = ExplainableAI(
        model=model,
        feature_names=user_features.columns.tolist(),
        user_features=user_features,
        diner_features=diner_features,
    )

    # Example: Explain a recommendation for user 123 and diner 456
    user_id = 123
    diner_id = 456

    # Get explanation
    explanation = explainer.explain_recommendation(user_id, diner_id)
    print("Explanation details:", explanation)

    # Get human-readable explanation
    reason = explainer.generate_recommendation_reason(user_id, diner_id)
    print("\nRecommendation reason:", reason)

    # Visualize explanation
    explainer.visualize_explanation(user_id, diner_id)


if __name__ == "__main__":
    main()
