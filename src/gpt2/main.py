import logging

from gpt2.micrograd.micrograd import Value

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)


def main() -> None:
    logging.info(f"Starting the main function in {__file__}")

    a = Value(4.0)
    b = Value(5.0)
    c = Value(6.0)

    d = a * (b + c)

    logging.info(f"d.prev = {d.prev}")


if __name__ == "__main__":
    main()
