from services.node2vec_service import node2vec_service
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Node2Vec model retraining...")
    success = node2vec_service.load_or_train_model(force_retrain=True)
    if success:
        logger.info("Node2Vec model retraining completed successfully.")
    else:
        logger.error("Node2Vec model retraining failed.")
