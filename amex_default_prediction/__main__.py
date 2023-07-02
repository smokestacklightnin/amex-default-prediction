import logging
import amex_default_prediction.data


logging.basicConfig(
    level=logging.DEBUG,
)


if __name__ == "__main__":
    logging.info("Hello Kaggle")

    amex_default_prediction.data.download.from_kaggle()
