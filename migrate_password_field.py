"""
Database Migration Script: Add password_hash field and migrate data
Run this script to migrate from api_key_hash to password_hash field

Execute with: python migrate_password_field.py
"""

import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, text
from app.models.user import User
from app.config import settings

async def migrate_password_field():
    """Migrate user passwords from api_key_hash to password_hash field"""
    
    engine = create_async_engine(settings.database_url)
    
    async with engine.begin() as conn:
        try:
            # Check if password_hash column exists
            result = await conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'users' AND column_name = 'password_hash'
            """))
            
            if not result.fetchone():
                print("Adding password_hash column...")
                await conn.execute(text("""
                    ALTER TABLE users 
                    ADD COLUMN password_hash VARCHAR(255)
                """))
                print("✅ password_hash column added")
            else:
                print("✅ password_hash column already exists")
            
            # Migrate data from api_key_hash to password_hash
            print("Migrating password data...")
            await conn.execute(text("""
                UPDATE users 
                SET password_hash = api_key_hash 
                WHERE password_hash IS NULL AND api_key_hash IS NOT NULL
            """))
            print("✅ Password data migrated")
            
            # Make password_hash NOT NULL
            await conn.execute(text("""
                ALTER TABLE users 
                ALTER COLUMN password_hash SET NOT NULL
            """))
            print("✅ password_hash set to NOT NULL")
            
            print("\n🎉 Migration completed successfully!")
            print("\nNow you can:")
            print("1. Update existing user passwords in database")
            print("2. Use the new password_hash field for authentication")
            
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            raise
    
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(migrate_password_field())