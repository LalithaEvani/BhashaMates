"""
Models package initialization
"""
from models.bert_model import RegularizedBertForIntentClassification
from models.roberta_model import RegularizedRobertaForIntentClassification

def get_model(config, num_labels):
    """
    Factory function to create the appropriate model based on config
    
    Args:
        config: Configuration object
        num_labels: Number of intent classes
        
    Returns:
        Initialized model
    """
    if config.model.model_type.lower() == "bert":
        print(f'inside init get model function')
        return RegularizedBertForIntentClassification(
            model_name=config.model.model_name,
            num_labels=num_labels,
            config=config
        )
    elif config.model.model_type.lower() == "roberta":
        return RegularizedRobertaForIntentClassification(
            model_name=config.model.model_name,
            num_labels=num_labels,
            config=config
        )
    else:
        raise ValueError(f"Unsupported model type: {config.model.model_type}")