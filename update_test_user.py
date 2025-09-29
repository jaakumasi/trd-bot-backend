"""
Generate and update password hash for testing
Run this to create a test user or update existing user password
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bcrypt
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, text
from app.models.user import User
from app.config import settings

async def update_user_password():
    """Update or create user with new password hash"""
    
    # Generate new hash for password "1234"
    password = "1234"
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    hashed_str = hashed.decode('utf-8')
    
    print(f"Generated hash for password '{password}': {hashed_str}")
    
    engine = create_async_engine(settings.database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        try:
            # Check if demo_user exists
            query = select(User).where(User.username == "demo_user")
            result = await session.execute(query)
            user = result.scalar_one_or_none()
            
            if user:
                # Update existing user
                user.password_hash = hashed_str
                print(f"✅ Updated password for existing user: {user.username}")
            else:
                # Create new test user
                user = User(
                    username="demo_user",
                    email="demo@example.com",
                    password_hash=hashed_str
                )
                session.add(user)
                print("✅ Created new test user: demo_user")
            
            await session.commit()
            
            print("\n🎉 Database updated successfully!")
            print("\n📋 Test Login Credentials:")
            print("Username: demo_user")
            print("Password: 1234")
            print(f"Hash: {hashed_str}")
            
        except Exception as e:
            print(f"❌ Database operation failed: {e}")
            await session.rollback()
            raise
    
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(update_user_password())