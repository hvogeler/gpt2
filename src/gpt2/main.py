import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)


def main() -> None:
    logging.info(f"Starting the main function in {__file__}")


if __name__ == "__main__":  # pragma: no cover
    main()
