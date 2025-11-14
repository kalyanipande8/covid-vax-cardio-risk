"""Small smoke test runner to verify the starter package imports cleanly."""
from src.python import data_loader


def main():
    print(data_loader.example())


if __name__ == "__main__":
    main()
