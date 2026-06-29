"""Create (or reset) the single login user + TOTP secret.

    python -m app.create_user

Prints an otpauth:// URI and an ASCII QR you can scan with any
authenticator app (Google Authenticator, Aegis, 1Password, ...).
"""
import getpass

from sqlmodel import Session, select

from app.db import engine, init_db
from app.models import User
from app.security import hash_password, new_totp_secret, totp_provisioning_uri


def main() -> None:
    init_db()
    username = input("Username: ").strip()
    if not username:
        print("Username required.")
        return

    password = getpass.getpass("Password: ")
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("Passwords do not match.")
        return
    if len(password) < 8:
        print("Use at least 8 characters.")
        return

    secret = new_totp_secret()
    with Session(engine) as db:
        existing = db.exec(select(User).where(User.username == username)).first()
        if existing:
            existing.password_hash = hash_password(password)
            existing.totp_secret = secret
            db.add(existing)
            print(f"\nUpdated existing user '{username}'.")
        else:
            db.add(
                User(
                    username=username,
                    password_hash=hash_password(password),
                    totp_secret=secret,
                )
            )
            print(f"\nCreated user '{username}'.")
        db.commit()

    uri = totp_provisioning_uri(secret, username)
    print("\nScan this with your authenticator app:")
    print(f"\nSecret (manual entry): {secret}")
    print(f"otpauth URI: {uri}\n")
    try:
        import qrcode

        qr = qrcode.QRCode(border=1)
        qr.add_data(uri)
        qr.make()
        qr.print_ascii(invert=True)
    except ImportError:
        print("(install 'qrcode' to render a scannable QR here)")


if __name__ == "__main__":
    main()
