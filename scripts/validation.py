def evaluate_model(trainer: Trainer, dataset: Dataset) -> None:
    """
    Evaluate the model using the provided Trainer and dataset.
    
    Parameters:
    - trainer: Trainer instance for model evaluation.
    - dataset: Dataset instance for evaluation.
    """
    # Evaluate the model
    results = trainer.evaluate(eval_dataset=dataset)
    
    print("Validation Results:")
    print(results)
