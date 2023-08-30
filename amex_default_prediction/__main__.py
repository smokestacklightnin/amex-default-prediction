import logging
import amex_default_prediction


logging.basicConfig(
    level=logging.DEBUG,
)


if __name__ == "__main__":
    amex_default_prediction.data.download.from_kaggle()
