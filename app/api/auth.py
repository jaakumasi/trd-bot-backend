from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional
import jwt
import bcrypt
from datetime import datetime, timedelta, timezone
import logging
from ..database import get_db
from ..models.user import User
from ..models.trade import TradingConfig
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()


# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        if len(v.encode('utf-8')) > 72:
            raise ValueError('Password is too long (max 72 bytes)')
        return v
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters long')
        if len(v) > 50:
            raise ValueError('Username is too long (max 50 characters)')
        return v


class UserLogin(BaseModel):
    username: str
    password: str
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if len(v.encode('utf-8')) > 72:
            raise ValueError('Password is too long')
        return v


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse


# Helper functions
def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    # Bcrypt has a 72-byte limit, so truncate if necessary
    if len(password.encode('utf-8')) > 72:
        password = password[:72]
    
    # Generate salt and hash password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    # Bcrypt has a 72-byte limit, so truncate if necessary
    if len(plain_password.encode('utf-8')) > 72:
        plain_password = plain_password[:72]
    
    try:
        # Convert hash back to bytes if it's a string
        if isinstance(hashed_password, str):
            hashed_password = hashed_password.encode('utf-8')
        
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password)
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(seconds=settings.jwt_expiration)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm
    )
    return encoded_jwt


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Get current authenticated user"""
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")

    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Get user from database
    query = select(User).where(User.username == username)
    result = await db.execute(query)
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    return user


# API Endpoints
@router.post("/register", response_model=Token)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user"""
    # Check if user already exists
    query = select(User).where(
        (User.username == user_data.username) | (User.email == user_data.email)
    )
    result = await db.execute(query)
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(
            status_code=400, detail="Username or email already registered"
        )

    # Create new user
    hashed_password = hash_password(user_data.password)
    user = User(
        username=user_data.username, email=user_data.email, password_hash=hashed_password
    )

    db.add(user)
    await db.commit()
    await db.refresh(user)

    # Create default trading configuration
    trading_config = TradingConfig(
        user_id=user.id,
        strategy_name="day_trading",
        risk_percentage=1.0,
        trading_pair="BTCUSDT",
        is_active=False,
        is_test_mode=True,
    )

    db.add(trading_config)
    await db.commit()

    # Create access token
    access_token = create_access_token(data={"sub": user.username})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            created_at=user.created_at,
        ),
    }


@router.post("/login", response_model=Token)
async def login(login_data: UserLogin, db: AsyncSession = Depends(get_db)):
    """Login user"""
    # Get user from database
    query = select(User).where(User.username == login_data.username)
    result = await db.execute(query)
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    # Handle password verification with better error handling
    try:
        # Debug info
        logger.info(f"Attempting login for user: {user.username}")
        logger.info(f"Password length: {len(login_data.password)} chars, {len(login_data.password.encode('utf-8'))} bytes")
        logger.info(f"Stored hash length: {len(user.password_hash) if user.password_hash else 0} chars")
        
        password_match = verify_password(login_data.password, user.password_hash)
        if not password_match:
            raise HTTPException(status_code=401, detail="Incorrect username or password")
    except ValueError as e:
        if "password cannot be longer than 72 bytes" in str(e):
            logger.error(f"Password too long for user {user.username}: {len(login_data.password.encode('utf-8'))} bytes")
            raise HTTPException(
                status_code=400, 
                detail="Password is too long. Please use a shorter password or contact support."
            )
        else:
            logger.error(f"Password verification error for user {user.username}: {e}")
            raise HTTPException(
                status_code=401, 
                detail="Password verification failed. Please try again or contact support."
            )
    except Exception as e:
        logger.error(f"Unexpected error during login for user {user.username}: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error during authentication."
        )

    # Create access token
    access_token = create_access_token(data={"sub": user.username})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            created_at=user.created_at,
        ),
    }


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        created_at=current_user.created_at,
    )


@router.post("/reset-password")
async def reset_password(
    username: str, 
    new_password: str, 
    db: AsyncSession = Depends(get_db)
):
    """Reset password for a user (temporary endpoint for development)"""
    # Validate new password
    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters long")
    if len(new_password.encode('utf-8')) > 72:
        raise HTTPException(status_code=400, detail="Password is too long (max 72 bytes)")
    
    # Get user from database
    query = select(User).where(User.username == username)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Hash and update password
    user.password_hash = hash_password(new_password)
    await db.commit()
    
    return {"message": "Password reset successfully"}
