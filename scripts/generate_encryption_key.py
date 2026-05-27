"""Print a fresh Fernet key. Paste the output into ENCRYPTION_KEY in your .env."""
from cryptography.fernet import Fernet

if __name__ == "__main__":
    print(Fernet.generate_key().decode())
