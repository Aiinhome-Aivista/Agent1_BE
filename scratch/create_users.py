"""Create test users for mail escalation testing."""
from app.core.database import SessionLocal
from app.core.security import hash_password
from app.models import User
from app.models.user import UserRole

db = SessionLocal()
try:
    users = [
        {
            "email": "rohan.developer2001@gmail.com",
            "full_name": "Rohan Dev (L1 - DataOps/SRE)",
            "role": UserRole.dataops_engineer,
        },
        {
            "email": "rohandas9064@gmail.com",
            "full_name": "Rohan Das (L2 - Data Platform Lead)",
            "role": UserRole.data_platform_lead,
        },
        {
            "email": "rohandas8944@gmail.com",
            "full_name": "Rohan Das (L3 - Data Engineer)",
            "role": UserRole.data_engineer,
        },
    ]

    for u in users:
        existing = db.query(User).filter(User.email == u["email"]).first()
        if existing:
            print(f"SKIP: {u['email']} already exists (id={existing.id})")
            continue

        user = User(
            email=u["email"],
            full_name=u["full_name"],
            hashed_password=hash_password("test1234"),
            is_active=True,
            is_admin=False,
            role=u["role"],
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"CREATED: id={user.id} email={user.email} role={u['role'].value}")

    print("\nAll users:")
    for u in db.query(User).all():
        role_val = u.role.value if hasattr(u.role, "value") else str(u.role)
        print(f"  id={u.id} email={u.email} role={role_val} active={u.is_active}")
finally:
    db.close()
